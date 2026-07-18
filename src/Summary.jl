# Summary.jl — portable model summary / comparison layer for BIM round-trip stress testing.
#
# A summary is a Dict{String,Any} of coarse, backend-agnostic invariants of an introspected
# model (the NamedTuple produced by e.g. KhepriRevit's introspect_model, or any NamedTuple
# with the same field names). It is serialized to a plain-text, line-oriented format:
#
#   count.<category> = <int>       categories: levels, walls, curtain_walls, floors, ceilings,
#                                  roofs, columns, beams, doors, windows, stairs, railings,
#                                  fixtures, groups, group_instances, fallback_meshes
#   level_elevations = v1,v2,...   sorted ascending, rounded to 3 decimals
#   total_wall_length = <float>    sum of path_length over non-curtain walls, 3 decimals
#   total_slab_area = <float>      best-effort shoelace area of floor slabs (outer - holes)
#   bbox = x0,y0,z0,x1,y1,z1       from wall path vertices + slab vertices + column cb +
#                                  fixture loc; the line (and key) is omitted when no geometry
#
# model_summary only READS proxy fields (w.path, s.region, l.height, ...) — it never realizes
# or renders shapes, so it is safe on models built under with_introspection and on live ones.
# compare_summaries yields PASS/FAIL/WARN verdict lines for a src vs rebuilt model pair, with
# allowances for Revit-template artifacts (:template_levels, :stair_railings).

export model_summary, summary_string, write_summary, read_summary, compare_summaries

const summary_count_categories =
  (:levels, :walls, :curtain_walls, :floors, :ceilings, :roofs, :columns, :beams,
   :doors, :windows, :stairs, :stairs_in_groups, :railings, :fixtures, :groups,
   :group_instances, :fallback_meshes)

# 3-decimal rounding, normalizing -0.0 so the serialized form is canonical.
_summary_round3(v) =
  let r = round(Float64(v), digits=3)
    r == 0.0 ? 0.0 : r
  end

# World coordinates of a Loc; degrade to raw coordinates if in_world is unavailable.
_summary_world_xyz(p) =
  try
    let q = in_world(p)
      (Float64(cx(q)), Float64(cy(q)), Float64(cz(q)))
    end
  catch
    (Float64(cx(p)), Float64(cy(p)), Float64(cz(p)))
  end

_summary_path_vertices(path) =
  [_summary_world_xyz(v) for v in path_vertices(path)]

# Unsigned shoelace area of a closed vertex loop projected on the world XY plane.
_summary_shoelace(vs) =
  length(vs) < 3 ? 0.0 :
  abs(sum(let q = vs[mod1(i + 1, length(vs))], p = vs[i]
        p[1]*q[2] - q[1]*p[2]
      end for i in 1:length(vs))) / 2

# A slab's area container may be a Region (paths, first = outer, rest = holes) or a bare path.
_summary_region_paths(r::Region) = r.paths
_summary_region_paths(p) = [p]

_summary_slab_region(s) =
  hasproperty(s, :region) ? s.region :
  hasproperty(s, :contour) ? s.contour :
  nothing

_summary_slab_area(s) =
  try
    let r = _summary_slab_region(s)
      isnothing(r) ? 0.0 :
        let areas = [_summary_shoelace(_summary_path_vertices(p)) for p in _summary_region_paths(r)]
          isempty(areas) ? 0.0 : max(0.0, areas[1] - sum(areas[2:end], init=0.0))
        end
    end
  catch
    0.0
  end

_summary_wall_length(w) =
  try
    hasproperty(w, :path) ? Float64(path_length(w.path)) : 0.0
  catch
    0.0
  end

"""
    model_summary(model) -> Dict{String,Any}

Compute the coarse invariants of an introspected BIM model NamedTuple (fields `levels`,
`walls`, `floors`, ... — every field is optional and defaults to empty). Reads proxy fields
only; never realizes shapes. Keys: `"count.<category>"` (Int) for each of
`$(join(summary_count_categories, ", "))`, `"level_elevations"` (sorted `Vector{Float64}`),
`"total_wall_length"`, `"total_slab_area"` (Float64), and `"bbox"` (`NTuple{6,Float64}`,
absent when the model contributes no geometry).
"""
function model_summary(model)
  fld(s) = hasproperty(model, s) ? getproperty(model, s) : []
  levels = fld(:levels)
  walls_all = fld(:walls)
  floors = fld(:floors)
  columns = fld(:columns)
  fixtures = fld(:fixtures)
  groups = fld(:groups)
  ncurtain = count(is_curtain_wall, walls_all)
  plain_walls = [w for w in walls_all if !is_curtain_wall(w)]
  group_insts(g) =
    hasproperty(g, :instances) ? g.instances :
    (g isa Tuple && length(g) >= 3) ? g[3] : []
  group_membs(g) =
    hasproperty(g, :members) ? g.members :
    (g isa Tuple && length(g) >= 2) ? g[2] : []
  s = Dict{String,Any}(
    "count.levels" => length(levels),
    "count.walls" => length(plain_walls),
    "count.curtain_walls" => ncurtain,
    "count.floors" => length(floors),
    "count.ceilings" => length(fld(:ceilings)),
    "count.roofs" => length(fld(:roofs)),
    "count.columns" => length(columns),
    "count.beams" => length(fld(:beams)),
    "count.doors" => sum(w -> hasproperty(w, :doors) ? length(w.doors) : 0, plain_walls; init=0),
    "count.windows" => sum(w -> hasproperty(w, :windows) ? length(w.windows) : 0, plain_walls; init=0),
    "count.stairs" => length(fld(:stairs)),
    # Stairs living INSIDE groups: Revit auto-generates railings for them too, so the
    # :stair_railings allowance must know about them (informational WARN-only in comparison).
    "count.stairs_in_groups" =>
      sum(g -> count(m -> m isa Stair, group_membs(g)), groups; init=0),
    "count.railings" => length(fld(:railings)),
    "count.fixtures" => length(fixtures),
    "count.groups" => length(groups),
    "count.group_instances" => sum(g -> length(group_insts(g)), groups; init=0),
    "count.fallback_meshes" => length(fld(:fallback_meshes)),
    "level_elevations" =>
      sort([_summary_round3(l.height) for l in levels if hasproperty(l, :height)]),
    "total_wall_length" =>
      _summary_round3(sum(_summary_wall_length, plain_walls; init=0.0)),
    "total_slab_area" =>
      _summary_round3(sum(_summary_slab_area, floors; init=0.0)))
  pts = NTuple{3,Float64}[]
  for w in walls_all
    try
      hasproperty(w, :path) && append!(pts, _summary_path_vertices(w.path))
    catch
    end
  end
  for f in floors
    try
      let r = _summary_slab_region(f)
        isnothing(r) ||
          foreach(p -> append!(pts, _summary_path_vertices(p)), _summary_region_paths(r))
      end
    catch
    end
  end
  for c in columns
    try
      hasproperty(c, :cb) && push!(pts, _summary_world_xyz(c.cb))
    catch
    end
  end
  for f in fixtures
    try
      hasproperty(f, :loc) && push!(pts, _summary_world_xyz(f.loc))
    catch
    end
  end
  isempty(pts) ||
    (s["bbox"] = (_summary_round3(minimum(p[1] for p in pts)),
                  _summary_round3(minimum(p[2] for p in pts)),
                  _summary_round3(minimum(p[3] for p in pts)),
                  _summary_round3(maximum(p[1] for p in pts)),
                  _summary_round3(maximum(p[2] for p in pts)),
                  _summary_round3(maximum(p[3] for p in pts))))
  s
end

_summary_value_str(v::Integer) = string(v)
_summary_value_str(v::Real) = string(Float64(v))
_summary_value_str(v::AbstractVector) = join(map(x -> string(Float64(x)), v), ",")
_summary_value_str(v::Tuple) = join(map(x -> string(Float64(x)), v), ",")

"""
    summary_string(s::Dict) -> String

Canonical line-oriented serialization of a summary: `key = value` lines, keys sorted.
"""
summary_string(s::Dict) =
  join(["$k = $(_summary_value_str(s[k]))" for k in sort(collect(keys(s)))], "\n") * "\n"

write_summary(path::AbstractString, s::Dict) =
  let dir = dirname(path)
    isempty(dir) || mkpath(dir)
    open(io -> write(io, summary_string(s)), path, "w")
    path
  end

_summary_parse_value(k, v) =
  startswith(k, "count.") ? parse(Int, v) :
  k == "level_elevations" ?
    [parse(Float64, strip(x)) for x in split(v, ',') if !isempty(strip(x))] :
  k == "bbox" ?
    let xs = [parse(Float64, strip(x)) for x in split(v, ',')]
      length(xs) == 6 ? (xs[1], xs[2], xs[3], xs[4], xs[5], xs[6]) :
        error("bbox line must have 6 coordinates, got $(length(xs))")
    end :
  parse(Float64, v)

"""
    read_summary(path::AbstractString) -> Dict{String,Any}

Parse a summary file written by `write_summary`. Blank lines and `#` comments are skipped.
"""
function read_summary(path::AbstractString)
  d = Dict{String,Any}()
  for line in eachline(path)
    let ln = strip(line)
      (isempty(ln) || startswith(ln, "#")) && continue
      let m = match(r"^([^=]+?)\s*=\s*(.*)$", ln)
        isnothing(m) && continue
        let k = String(strip(m[1]))
          d[k] = _summary_parse_value(k, String(strip(m[2])))
        end
      end
    end
  end
  d
end

"""
    compare_summaries(src::Dict, rebuilt::Dict;
                      allow=Symbol[], count_tol=Dict{String,Int}(),
                      length_rtol=0.01, area_rtol=0.05, elev_atol=0.002)
      -> (ok = Bool, lines = Vector{String})

Compare the summary of a source model against the summary of its round-trip rebuild.
Each verdict line is prefixed `"PASS "`, `"FAIL "`, or `"WARN "`; `ok` is true iff no line
FAILed. `allow` grants Revit-template deviations:

- `:template_levels` — the rebuilt model may contain up to 2 extra levels whose elevations
  are not in src (the blank Revit template ships Level 1/Level 2); excess beyond 2, or
  missing src levels, still FAIL.
- `:stair_railings` — rebuilt railing count may exceed src by up to 2 per src stair (Revit
  auto-generates stair railings).

`count.fallback_meshes` differences are always WARN, never FAIL (rebuild re-introspection
typically runs without OBJ export). `count_tol` maps a count key (`"count.columns"` or bare
`"columns"`) to an allowed absolute deviation.
"""
function compare_summaries(src::Dict, rebuilt::Dict;
                           allow=Symbol[], count_tol=Dict{String,Int}(),
                           length_rtol=0.01, area_rtol=0.05, elev_atol=0.002)
  lines = String[]
  pass(msg) = push!(lines, "PASS " * msg)
  fail(msg) = push!(lines, "FAIL " * msg)
  warn(msg) = push!(lines, "WARN " * msg)
  cnt(d, k) = Int(get(d, k, 0))
  for cat in summary_count_categories
    let k = "count.$cat",
        sv = cnt(src, k),
        rv = cnt(rebuilt, k),
        tol = max(get(count_tol, k, 0), get(count_tol, String(cat), 0))
      if cat == :fallback_meshes
        sv == rv ? pass("$k: $sv == $rv") :
          warn("$k: src=$sv rebuilt=$rv (informational: fallback meshes depend on OBJ export)")
      elseif cat == :levels && :template_levels in allow && rv > sv
        rv - sv <= 2 ?
          warn("$k: src=$sv rebuilt=$rv (+$(rv - sv) template level(s) allowed)") :
          fail("$k: src=$sv rebuilt=$rv (exceeds +2 template-level allowance)")
      elseif cat == :stairs_in_groups
        # Informational: group membership is compared only loosely (member sets live inside
        # group factories, not as top-level counts).
        sv == rv ? pass("$k: $sv == $rv") : warn("$k: src=$sv rebuilt=$rv (informational)")
      elseif cat == :railings && :stair_railings in allow && rv > sv
        let excess = rv - sv,
            allowed = 2 * (cnt(src, "count.stairs") + cnt(src, "count.stairs_in_groups"))
          excess <= allowed + tol ?
            warn("$k: src=$sv rebuilt=$rv (+$excess stair-generated railing(s) allowed)") :
            fail("$k: src=$sv rebuilt=$rv (excess $excess > allowed $allowed)")
        end
      else
        abs(sv - rv) <= tol ?
          (sv == rv ? pass("$k: $sv == $rv") :
             warn("$k: src=$sv rebuilt=$rv (within tolerance $tol)")) :
          fail("$k: src=$sv rebuilt=$rv")
      end
    end
  end
  let se = Float64.(get(src, "level_elevations", Float64[])),
      unmatched = Float64.(get(rebuilt, "level_elevations", Float64[])),
      missing_elevs = Float64[]
    for e in se
      let i = findfirst(x -> abs(x - e) <= elev_atol, unmatched)
        isnothing(i) ? push!(missing_elevs, e) : deleteat!(unmatched, i)
      end
    end
    isempty(missing_elevs) ||
      fail("level_elevations: missing in rebuilt: $(join(missing_elevs, ","))")
    if !isempty(unmatched)
      (:template_levels in allow && length(unmatched) <= 2) ?
        warn("level_elevations: extra in rebuilt: $(join(unmatched, ",")) (template levels allowed)") :
        fail("level_elevations: extra in rebuilt: $(join(unmatched, ","))")
    end
    isempty(missing_elevs) && isempty(unmatched) &&
      pass("level_elevations: all $(length(se)) matched (atol=$elev_atol)")
  end
  relchk(key, rtol) =
    let sv = get(src, key, nothing), rv = get(rebuilt, key, nothing)
      if isnothing(sv) && isnothing(rv)
        nothing
      elseif isnothing(sv) || isnothing(rv)
        fail("$key: src=$sv rebuilt=$rv")
      else
        let svf = Float64(sv), rvf = Float64(rv),
            tol = rtol * max(abs(svf), abs(rvf), 1e-9)
          abs(svf - rvf) <= tol ?
            pass("$key: $svf ~= $rvf (rtol=$rtol)") :
            fail("$key: src=$svf rebuilt=$rvf (delta=$(abs(svf - rvf)) > $tol)")
        end
      end
    end
  relchk("total_wall_length", length_rtol)
  relchk("total_slab_area", area_rtol)
  let sb = get(src, "bbox", nothing), rb = get(rebuilt, "bbox", nothing)
    if isnothing(sb) && isnothing(rb)
      nothing
    elseif isnothing(sb)
      warn("bbox: absent in src, present in rebuilt")
    elseif isnothing(rb)
      fail("bbox: present in src, absent in rebuilt")
    else
      let diag = max(sqrt(sum((Float64(sb[i+3]) - Float64(sb[i]))^2 for i in 1:3)), 1e-9),
          tol = max(length_rtol * diag, 0.01),
          dmax = maximum(abs(Float64(sb[i]) - Float64(rb[i])) for i in 1:6)
        dmax <= tol ?
          pass("bbox: max coordinate delta $(round(dmax, digits=4)) <= $(round(tol, digits=4))") :
          fail("bbox: max coordinate delta $(round(dmax, digits=4)) > $(round(tol, digits=4))")
      end
    end
  end
  (ok = !any(startswith("FAIL "), lines), lines = lines)
end
