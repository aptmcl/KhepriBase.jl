# CodeGen.jl — Portable BIM-model → Julia-program code generation.
#
# This is the backend-agnostic half of the "introspect a CAD/BIM model and emit an equivalent
# Khepri program" capability. It operates purely on portable Khepri BIM shapes (Wall, Slab, Column,
# …) and `Expr`, and was lifted verbatim out of KhepriRevit (Revit.jl) so that any backend can reuse
# it. The only backend-specific decisions are isolated behind two hooks:
#
#   b_native_family_expr(b, var, meta)  — the `set_backend_family(...)` line(s) a backend emits for a
#                                         reconstructed family (nothing ⇒ no mapping).
#   b_codegen_module(b)                 — the module named in the emitted `using` line.
#
# Architecture (inspired by Tomás Grelha da Cunha's MSc thesis, "Refactoring for Dynamic Languages:
# The Julia Case", IST 2020):
#   1. Introspect : backend model → Khepri shape objects   (backend-specific; see Introspection.jl)
#   2. meta_program: shapes → raw Julia Expr (AST)
#   3. Transform  : refactoring passes Expr → Expr (extract variables, loop rerolling, …)
#   4. Print      : pretty-print final Expr → source text

# ─── Backend hooks (defaults) ─────────────────────────────────────────────────

# The native `set_backend_family(...)` expression for a reconstructed family; `nothing` ⇒ emit none.
# `meta` is the introspection metadata NamedTuple stored for the family. Backends with a native family
# system (Revit) override this; geometry-only backends leave the default.
b_native_family_expr(b::Backend, var, meta) = nothing

# The module named in the emitted `using` line. Default derives it from the backend's package; socket
# backends (whose concrete type lives in KhepriBase) must override.
b_codegen_module(b::Backend) = nameof(parentmodule(typeof(b)))

# ─── Utilities ────────────────────────────────────────────────────────────────

sanitize_name(name) =
  let s = replace(lowercase(strip(name)), r"[^a-z0-9_]" => "_"),
      s = replace(s, r"_+" => "_"),
      s = strip(s, '_')
    isempty(s) ? "unnamed" : (isdigit(s[1]) ? "_" * s : s)
  end

# Detect regular grid pattern in a set of values
function detect_regular_spacing(vals; tol=0.01)
  sorted = sort(unique(vals))
  length(sorted) < 2 && return nothing
  spacing = sorted[2] - sorted[1]
  spacing < tol && return nothing
  for i in 2:length(sorted)-1
    abs((sorted[i+1] - sorted[i]) - spacing) > tol && return nothing
  end
  (first=sorted[1], step=spacing, last=sorted[end], count=length(sorted))
end

# Bottom-up map over an Expr tree
map_expr(f, e::Expr) = f(Expr(e.head, map(x -> map_expr(f, x), e.args)...))
map_expr(f, x) = f(x)

# Collect all sub-expressions matching a predicate
collect_exprs(pred, x) = pred(x) ? [x] : []
collect_exprs(pred, e::Expr) =
  let result = pred(e) ? [e] : []
    for a in e.args
      append!(result, collect_exprs(pred, a))
    end
    result
  end

# Compare two Exprs structurally, returning list of (path, val1, val2) diffs.
# path is a vector of arg indices.
expr_diff(e1, e2) = e1 == e2 ? [] : [([],  e1, e2)]
expr_diff(e1::Expr, e2::Expr) =
  if e1.head != e2.head || length(e1.args) != length(e2.args)
    [([], e1, e2)]
  else
    let diffs = []
      for (i, (a1, a2)) in enumerate(zip(e1.args, e2.args))
        for (path, v1, v2) in expr_diff(a1, a2)
          push!(diffs, (vcat([i], path), v1, v2))
        end
      end
      diffs
    end
  end

# Replace the value at a path in an Expr with a new value
expr_replace_at(e, path, val) =
  isempty(path) ? val :
  let args = collect(e.args)
    args[path[1]] = expr_replace_at(args[path[1]], path[2:end], val)
    Expr(e.head, args...)
  end

# The family constructor names known to the codegen passes (dedup/naming/section grouping).
const _family_function_names = Set([:wall_family, :curtain_wall_family, :slab_family, :ceiling_family,
                    :column_family, :beam_family, :door_family, :window_family,
                    :roof_family, :family_element_family, :table_family, :chair_family,
                    :toilet_family, :sink_family, :closet_family,
                    :stair_family, :railing_family])

# Family metadata is carried explicitly in `model.family_meta` (an IdDict family → FamilyMeta) built
# by introspection, not in module globals — see family_expr_map below.

# `is_curtain_wall` is the @defproxy-generated predicate for CurtainWall (Shapes.jl); reused as-is.

# Deduplicate slabs whose boundary vertices are the same set (different order)
_dedup_slabs(members) =
  let seen_slab_verts = Set{Set{Tuple{Float64,Float64,Float64}}}(),
      result = []
    for m in members
      if m isa Slab
        let verts = m.region isa Region ?
              (m.region.paths[1] isa ClosedPolygonalPath ?
                Set((round(cx(v), digits=4), round(cy(v), digits=4), round(cz(v), digits=4))
                    for v in m.region.paths[1].vertices) :
                Set{Tuple{Float64,Float64,Float64}}()) :
              Set{Tuple{Float64,Float64,Float64}}()
          if isempty(verts) || verts ∉ seen_slab_verts
            push!(seen_slab_verts, verts)
            push!(result, m)
          end
        end
      else
        push!(result, m)
      end
    end
    result
  end

_shape_family_category(m) =
  m isa Wall ? (is_curtain_wall(m) ? :curtain_wall : :wall) :
  m isa Slab ? :slab :
  m isa Column ? :column :
  m isa FreeColumn ? :column :
  m isa Beam ? :beam :
  m isa Ceiling ? :ceiling :
  m isa Roof ? :roof :
  m isa Stair ? :stair :
  m isa Railing ? :railing :
  m isa Table ? :table :
  m isa Chair ? :chair :
  m isa Toilet ? :toilet :
  m isa Sink ? :sink :
  m isa Closet ? :closet :
  m isa FamilyElement ? :family_element : :other

# ─── Phase 2: Model → Expr ────────────────────────────────────────────────────

# Map each reconstructed family's meta_program Expr to its metadata, from the model's `family_meta`
# carrier (IdDict family → FamilyMeta). Because families are reconstructed *named by their type*
# (family_key), distinct native types produce distinct Exprs here, so this map does NOT collapse
# them the way the former Expr-keyed global did. Consumed by extract_families (variable naming) and
# add_backend_families (native family mapping).
family_expr_map(model) =
  let fmap = Dict{Expr, FamilyMeta}(),
      fm = hasproperty(model, :family_meta) ? model.family_meta : ()
    for (fam, meta) in fm
      fmap[meta_program(fam)] = meta
    end
    fmap
  end

function model_to_expr(model)
  stmts = Expr[]
  for l in model.levels
    push!(stmts, meta_program(l))
  end
  for w in model.walls
    push!(stmts, meta_program(w))
  end
  for f in model.floors
    push!(stmts, meta_program(f))
  end
  for c in model.columns
    push!(stmts, meta_program(c))
  end
  for bm in model.beams
    push!(stmts, meta_program(bm))
  end
  for ce in model.ceilings
    push!(stmts, meta_program(ce))
  end
  for rf in model.roofs
    push!(stmts, meta_program(rf))
  end
  for fx in model.fixtures
    push!(stmts, meta_program(fx))
  end
  for st in model.stairs
    push!(stmts, meta_program(st))
  end
  for rl in model.railings
    push!(stmts, meta_program(rl))
  end
  # Groups: emit group definitions and group_instance placements
  for g in model.groups
    let grp_name = sanitize_name("group_$(g.type_name)"),
        grp_var = Symbol(grp_name),
        factory_var = Symbol("$(grp_name)_factory"),
        # Compute group origin from first instance location
        origin = g.instances[1],
        # Build member shape expressions, translated to be group-relative
        member_stmts = Expr[]
      for m in g.members
        push!(member_stmts, meta_program(m))
      end
      body_expr = Expr(:block, member_stmts...)
      translated_body = translate_xyz_expr(body_expr, cx(origin), cy(origin), cz(origin))
      # Emit: factory function
      push!(stmts, Expr(:function, Expr(:call, factory_var), translated_body))
      # Emit: grp_var = group("name", factory=factory_var)
      push!(stmts, Expr(:(=), grp_var,
        Expr(:call, :group,
          Expr(:parameters,
            Expr(:kw, :factory, factory_var)),
          grp_name)))
      # Emit: group_instance(grp_var, xyz(...))
      for loc in g.instances
        push!(stmts, Expr(:call, :group_instance, grp_var,
          Expr(:call, :xyz,
            meta_program(cx(loc)), meta_program(cy(loc)), meta_program(cz(loc)))))
      end
    end
  end
  # Mesh-fallback tail (Phase 6): emit an obj_model placement for each element no parametric reader
  # consumed (curtain panels, MEP, topography, in-place, sloped/degenerate), so nothing is silently lost.
  if hasproperty(model, :fallback_meshes)
    for m in model.fallback_meshes
      push!(stmts, meta_program(m))
    end
  end
  Expr(:block, stmts...)
end

# ── Geometric round-trip (non-BIM backends: AutoCAD, …) ──────────────────────
# Emit a geometry-only program: one meta_program statement per reconstructed shape. Skips shapes that
# can't round-trip to reproducible code — `Unknown` (opaque native handles, e.g. an AutoCAD Solid3d the
# CAD API cannot classify back to box/sphere/…), which would otherwise emit a dead `unknown(handle)`
# call. Needs no BIM structure; the standard passes then refine it (extract_levels/families no-op with
# none present, while loop_rerolling still rerolls repeated geometry such as arrays into for-loops).
shapes_to_expr(shapes) =
  let keep = [s for s in shapes if !is_unknown(s)],
      dropped = length(shapes) - length(keep)
    dropped > 0 && @info "shapes_to_expr: skipped $dropped non-reproducible shape(s) (e.g. AutoCAD Solid3d → unknown handle)"
    Expr(:block, [meta_program(s) for s in keep]...)
  end

# Run the full codegen pipeline on an already-reconstructed set of geometric shapes (e.g. from
# existing_shapes on a geometric backend). Same passes as the BIM generate_khepri_code; the family passes
# no-op without families. Returns the source; also writes it to output_path when a path is given.
function shapes_to_khepri_code(shapes, b; output_path=nothing)
  let raw_expr = shapes_to_expr(shapes),
      fmap = Dict{Expr, FamilyMeta}(),
      passes = [extract_levels,
                extract_families(fmap),
                add_backend_families(b, fmap),
                loop_rerolling,
                detect_level_repetition,
                add_geometric_header(b)],
      code = expr_to_string(foldl((e, pass) -> pass(e), passes, init=raw_expr))
    output_path === nothing || open(io -> write(io, code), output_path, "w")
    code
  end
end

# Translate all xyz/xy calls in an Expr tree by subtracting an offset.
# Skips door/window positions (wall-relative, not world coords).
translate_xyz_expr(x, dx, dy, dz) = x
translate_xyz_expr(e::Expr, dx, dy, dz) =
  if (e.head == :kw && e.args[1] in (:doors, :windows, :family)) ||
     (e.head == :call && (e.args[1] in (:add_door, :add_window) || e.args[1] in _family_function_names))
    # These carry LOCAL coordinates — wall-relative opening positions and family-local frame
    # profiles — not world positions, so they must not be shifted to group-relative coordinates.
    e
  elseif e.head == :call && e.args[1] == :xyz && length(e.args) == 4 &&
     all(a -> a isa Real, e.args[2:4])
    Expr(:call, :xyz,
      meta_program(e.args[2] - dx),
      meta_program(e.args[3] - dy),
      meta_program(e.args[4] - dz))
  elseif e.head == :call && e.args[1] == :xy && length(e.args) == 3 &&
         all(a -> a isa Real, e.args[2:3])
    Expr(:call, :xy,
      meta_program(e.args[2] - dx),
      meta_program(e.args[3] - dy))
  else
    Expr(e.head, [translate_xyz_expr(a, dx, dy, dz) for a in e.args]...)
  end

# ─── Phase 3: Transform Passes ────────────────────────────────────────────────

# 3.1 Extract levels: deduplicate level(h) calls into named variables
function extract_levels(e::Expr)
  level_calls = collect_exprs(
    x -> x isa Expr && x.head == :call && x.args[1] == :level && length(x.args) == 2,
    e)
  unique_levels = unique(level_calls)
  isempty(unique_levels) && return e
  # Sort by height value
  sorted = sort(unique_levels, by=x -> x.args[2] isa Real ? x.args[2] : 0.0)
  assignments = Expr[]
  replacements = Dict{Expr, Symbol}()
  for (i, lc) in enumerate(sorted)
    var = Symbol("level_$(i-1)")
    push!(assignments, Expr(:(=), var, lc))
    replacements[lc] = var
  end
  body = map_expr(x -> get(replacements, x, x), e)
  # Filter out bare symbols left over from replacing standalone level() calls
  level_syms = Set(values(replacements))
  body_stmts = filter(s -> !(s isa Symbol && s in level_syms), body.args)
  Expr(:block, assignments..., body_stmts...)
end

# 3.2 Extract families: deduplicate family constructors into named variables
extract_families(fmap) = function (e::Expr)
  family_fns = _family_function_names
  family_calls = collect_exprs(
    x -> x isa Expr && x.head == :call && x.args[1] in family_fns,
    e)
  unique_families = unique(family_calls)
  isempty(unique_families) && return e
  assignments = Expr[]
  replacements = Dict{Expr, Symbol}()
  for fc in unique_families
    # Use introspection metadata for descriptive variable names
    meta = get(fmap, fc, nothing)
    var = if meta !== nothing
      Symbol(sanitize_name("$(meta.category)_$(meta.family_name)_$(meta.type_name)"))
    else
      let cat = replace(string(fc.args[1]), "_family" => "")
        Symbol("$(cat)_fam_$(length(assignments)+1)")
      end
    end
    push!(assignments, Expr(:(=), var, fc))
    replacements[fc] = var
  end
  stmts = e.head == :block ? e.args : [e]
  # Find where levels end and elements start
  first_non_assign = findfirst(
    s -> !(s isa Expr && s.head == :(=) && s.args[2] isa Expr &&
           s.args[2].head == :call && s.args[2].args[1] == :level),
    stmts)
  insert_pos = first_non_assign === nothing ? length(stmts) + 1 : first_non_assign
  body = map_expr(x -> get(replacements, x, x), Expr(:block, stmts...))
  Expr(:block, body.args[1:insert_pos-1]..., assignments..., body.args[insert_pos:end]...)
end

# Build `@isdefined(backend) && set_backend_family(var, backend, native)`: a cross-backend family
# mapping that applies only when its backend is actually loaded. This lets one generated program carry
# mappings for several backends at once (a Revit `.rfa` and an extracted OBJ for KhepriThreejs) yet run
# on each without referencing — or evaluating the native constructor of — the others. `&&` short-circuits
# so e.g. `revit_file_family(...)` is never evaluated in a KhepriThreejs session (where it's undefined).
guarded_backend_family_expr(var, backend::Symbol, native) =
  Expr(:&&,
       Expr(:macrocall, Symbol("@isdefined"), nothing, backend),
       Expr(:call, :set_backend_family, var, backend, native))

# 3.3 Add backend family mappings
# Uses _family_expr_meta populated by model_to_expr; the actual native family expression is
# supplied by the backend via b_native_family_expr(b, var, meta).
add_backend_families(b, fmap) = (e::Expr) ->
  let stmts = e.head == :block ? collect(e.args) : [e],
      family_fns = _family_function_names,
      new_stmts = Any[]
    for s in stmts
      push!(new_stmts, s)
      if s isa Expr && s.head == :(=) &&
         s.args[2] isa Expr && s.args[2].head == :call &&
         s.args[2].args[1] in family_fns
        let var = s.args[1],
            rhs = s.args[2],
            meta = get(fmap, rhs, nothing)
          if meta !== nothing
            let backend_call = b_native_family_expr(b, var, meta)
              # A backend may map a family to several backends at once (e.g. Revit `.rfa` + extracted
              # OBJ for KhepriThreejs), so accept a list of statements as well as a single one.
              if backend_call isa AbstractVector
                append!(new_stmts, backend_call)
              elseif backend_call !== nothing
                push!(new_stmts, backend_call)
              end
            end
          end
        end
      end
    end
    Expr(:block, new_stmts...)
  end

# 3.4 Loop rerolling: detect repeated shape calls forming grid patterns
loop_rerolling(e::Expr) =
  let stmts = e.head == :block ? collect(e.args) : [e],
      result = Any[],
      # Group consecutive statements by their "template" (shape call with same structure)
      i = 1
    while i <= length(stmts)
      let s = stmts[i]
        if s isa Expr && s.head == :call
          # Try to find a run of structurally similar calls
          let run_end = i,
              template = s
            for j in (i+1):length(stmts)
              let sj = stmts[j]
                if sj isa Expr && sj.head == :call && sj.args[1] == template.args[1] &&
                   length(sj.args) == length(template.args)
                  let diffs = expr_diff(template, sj)
                    # Only consider if all diffs are at leaf values (numbers, symbols)
                    all(d -> !(d[2] isa Expr) && !(d[3] isa Expr), diffs) || break
                    run_end = j
                  end
                else
                  break
                end
              end
            end
            if run_end - i + 1 >= 4  # Need at least 4 similar statements
              let run = stmts[i:run_end],
                  rerolled = _try_reroll(run)
                if rerolled !== nothing
                  push!(result, rerolled)
                  i = run_end + 1
                  continue
                end
              end
            end
          end
        end
        push!(result, s)
        i += 1
      end
    end
    Expr(:block, result...)
  end

function _try_reroll(stmts)
  length(stmts) < 4 && return nothing
  template = stmts[1]
  all_diffs = [expr_diff(template, s) for s in stmts[2:end]]
  isempty(all_diffs) && return nothing
  ref_paths = Set([d[1] for d in all_diffs[1]])
  all(ds -> Set([d[1] for d in ds]) == ref_paths, all_diffs) || return nothing
  paths = sort(collect(ref_paths))
  isempty(paths) && return nothing
  # Collect value sequences for each varying path
  value_lists = Dict{Vector{Int}, Vector{Any}}()
  for p in paths
    value_lists[p] = [_get_diff_value(template, p)]
  end
  for ds in all_diffs
    for (p, _, v2) in ds
      push!(value_lists[p], v2)
    end
  end
  # Try to detect 2D grid pattern (two varying paths)
  if length(paths) == 2
    _try_reroll_2d(template, paths, value_lists)
  elseif length(paths) == 1
    _try_reroll_1d(template, paths[1], value_lists[paths[1]])
  else
    nothing
  end
end

_get_diff_value(e::Expr, path) =
  isempty(path) ? e : _get_diff_value(e.args[path[1]], path[2:end])
_get_diff_value(x, path) = isempty(path) ? x : error("path too deep")

function _try_reroll_1d(template, path, values)
  range = _try_make_range(values)
  range === nothing && return nothing
  let var = gensym_short(:i),
      body = expr_replace_at(template, path, var)
    Expr(:for, Expr(:(=), var, range), Expr(:block, body))
  end
end

# Detect if v1, v2 form a nested grid pattern, returning path info
function _detect_nesting(v1, v2, n, p1, p2)
  let inner_unique = unique(v2), m = length(inner_unique)
    if n % m == 0
      let k = n ÷ m
        if all(i -> v2[i] == inner_unique[((i-1) % m) + 1], 1:n)
          let outer_unique = [v1[(i-1)*m + 1] for i in 1:k]
            if all(i -> all(j -> v1[(i-1)*m + j] == outer_unique[i], 1:m), 1:k)
              return (p1, outer_unique, p2, inner_unique)
            end
          end
        end
      end
    end
    # Try swapped
    let inner_unique = unique(v1), m = length(inner_unique)
      if n % m == 0
        let k = n ÷ m
          if all(i -> v1[i] == inner_unique[((i-1) % m) + 1], 1:n)
            let outer_unique = [v2[(i-1)*m + 1] for i in 1:k]
              if all(i -> all(j -> v2[(i-1)*m + j] == outer_unique[i], 1:m), 1:k)
                return (p2, outer_unique, p1, inner_unique)
              end
            end
          end
        end
      end
    end
    nothing
  end
end

function _try_reroll_2d(template, paths, value_lists)
  let p1 = paths[1], p2 = paths[2],
      v1 = value_lists[p1], v2 = value_lists[p2],
      n = length(v1),
      nested = _detect_nesting(v1, v2, n, p1, p2)
    nested === nothing && return nothing
    let (outer_path, outer_vals, inner_path, inner_vals) = nested,
        outer_range = _try_make_range(outer_vals),
        inner_range = _try_make_range(inner_vals)
      (outer_range === nothing || inner_range === nothing) && return nothing
      let outer_var = gensym_short(:x),
          inner_var = gensym_short(:y),
          body = expr_replace_at(
                   expr_replace_at(template, outer_path, outer_var),
                   inner_path, inner_var)
        Expr(:for, Expr(:(=), outer_var, outer_range),
             Expr(:block,
                  Expr(:for, Expr(:(=), inner_var, inner_range),
                       Expr(:block, body))))
      end
    end
  end
end

gensym_short(base::Symbol) =
  let names = Dict(:x => :x, :y => :y, :z => :z, :i => :i, :j => :j)
    get(names, base, base)
  end

# Try to convert a value list to a range expression
function _try_make_range(vals)
  # Only numeric leaves can form a range/step; a non-numeric differing leaf (e.g. obj_model file paths,
  # or family names) means this group can't be rerolled — signal rejection so it emits individually.
  all(v -> v isa Real, vals) || return nothing
  n = length(vals)
  n < 2 && return Expr(:vect, vals...)
  sorted = sort(vals)
  spacing = sorted[2] - sorted[1]
  if spacing > 0 && all(i -> abs((sorted[i] - sorted[i-1]) - spacing) < 0.001, 2:n)
    let first_v = meta_program(sorted[1]),
        step_v = meta_program(spacing),
        last_v = meta_program(sorted[end])
      if step_v == 1
        Expr(:call, :(:), first_v, last_v)
      else
        Expr(:call, :(:), first_v, step_v, last_v)
      end
    end
  else
    Expr(:vect, map(meta_program, sorted)...)
  end
end

# 3.5 Detect level repetition: identical floor plans across levels → for loop
detect_level_repetition(e::Expr) =
  let stmts = e.head == :block ? collect(e.args) : [e]
    # Group shape calls by their level reference
    # This is a complex analysis — for now, return unchanged
    # TODO: Implement when models with repeated floor plans are available
    e
  end

# 3.6 Add header
add_header(b) = (e::Expr) ->
  Expr(:block,
       Expr(:toplevel_comment, "Auto-generated Khepri code from Revit model"),
       Expr(:toplevel_comment, "Generated on: $(Dates.now())"),
       Expr(:using, Expr(:., b_codegen_module(b))),
       stmts_of(e)...)

# Header for the geometric round-trip (own provenance line so the BIM add_header — and its goldens —
# stay untouched).
add_geometric_header(b) = (e::Expr) ->
  Expr(:block,
       Expr(:toplevel_comment, "Auto-generated Khepri code (geometric round-trip via $(b_codegen_module(b)))"),
       Expr(:toplevel_comment, "Generated on: $(Dates.now())"),
       Expr(:using, Expr(:., b_codegen_module(b))),
       stmts_of(e)...)

stmts_of(e::Expr) = e.head == :block ? e.args : [e]

# ─── Phase 4: Pretty-Printing ─────────────────────────────────────────────────

function expr_to_string(e::Expr; indent=0)
  let ind = "  " ^ indent,
      io = IOBuffer()
    _print_block(io, e, indent)
    String(take!(io))
  end
end

function _print_block(io, e::Expr, indent)
  let stmts = e.head == :block ? e.args : [e],
      prev_section = :none
    for (i, s) in enumerate(stmts)
      let cur_section = _stmt_section(s)
        # Add blank line between sections
        if i > 1 && cur_section != prev_section && prev_section != :none
          println(io)
        end
        _print_stmt(io, s, indent)
        println(io)
        prev_section = cur_section
      end
    end
  end
end

_stmt_section(s) =
  if s isa Expr
    if s.head == :toplevel_comment
      :header
    elseif s.head == :using
      :header
    elseif s.head == :(=) && s.args[2] isa Expr
      let rhs = s.args[2]
        if rhs.head == :call && rhs.args[1] == :level
          :levels
        elseif rhs.head == :call && rhs.args[1] in _family_function_names
          :families
        else
          :elements
        end
      end
    elseif s.head == :call && s.args[1] == :set_backend_family
      :families
    elseif s.head == :&& && s.args[2] isa Expr && s.args[2].head == :call &&
           s.args[2].args[1] == :set_backend_family
      :families
    elseif s.head == :function
      :groups
    elseif s.head == :(=) && s.args[2] isa Expr && s.args[2].head == :call &&
           s.args[2].args[1] == :group
      :groups
    elseif s.head == :call && s.args[1] == :group_instance
      :groups
    elseif s.head == :for
      :elements
    elseif s.head == :call
      :elements
    else
      :other
    end
  else
    :other
  end

# Check if an Expr contains a :do block (for pretty-printing with do...end)
_has_do_block(e::Expr) = e.head == :call && any(a -> a isa Expr && a.head == :do, e.args) ||
                          e.head == :do ||
                          (e.head == :call && any(a -> a isa Expr && _has_do_block(a), e.args))
_has_do_block(x) = false

# Print a call expression that contains a do block:
# fn(args..., inner_fn() do\n  body\nend)
function _print_do_call(io, e::Expr, indent)
  let ind = "  " ^ indent
    if e.head == :call
      # Find do-block argument
      let do_idx = findfirst(a -> a isa Expr && a.head == :do, e.args),
          fn = e.args[1],
          regular_args = [a for (i, a) in enumerate(e.args[2:end]) if i + 1 != do_idx]
        if do_idx !== nothing
          let do_expr = e.args[do_idx],
              do_call = do_expr.args[1],  # the inner function call (e.g., collecting_shapes)
              do_lambda = do_expr.args[2], # the -> with body
              do_body = do_lambda.args[2]  # the block body
            if isempty(regular_args)
              print(io, "$(_expr_str(fn))($(_expr_str(do_call))) do")
            else
              print(io, "$(_expr_str(fn))($(join(map(_expr_str, regular_args), ", ")), $(_expr_str(do_call))) do")
            end
            println(io)
            _print_block(io, do_body, indent + 1)
            print(io, "$(ind)end")
          end
        else
          print(io, _expr_str(e))
        end
      end
    else
      print(io, _expr_str(e))
    end
  end
end

function _print_stmt(io, s::Expr, indent)
  let ind = "  " ^ indent
    if s.head == :toplevel_comment
      print(io, "$(ind)# $(s.args[1])")
    elseif s.head == :using
      print(io, "$(ind)using $(join(map(_expr_str, s.args), ", "))")
    elseif s.head == :(=)
      let rhs = s.args[2]
        if _has_do_block(rhs)
          print(io, "$(ind)$(s.args[1]) = ")
          _print_do_call(io, rhs, indent)
        else
          print(io, "$(ind)$(s.args[1]) = $(_expr_str(rhs))")
        end
      end
    elseif s.head == :function
      let sig = s.args[1],
          body = s.args[2]
        print(io, "$(ind)function $(_expr_str(sig))")
        println(io)
        _print_block(io, body, indent + 1)
        print(io, "$(ind)end")
      end
    elseif s.head == :for
      let binding = s.args[1],
          body = s.args[2]
        print(io, "$(ind)for $(_expr_str(binding))")
        println(io)
        _print_block(io, body, indent + 1)
        print(io, "$(ind)end")
      end
    elseif s.head == :call
      _print_call(io, s, ind)
    else
      print(io, "$(ind)$(_expr_str(s))")
    end
  end
end

_print_stmt(io, s, indent) = print(io, "  " ^ indent, _expr_str(s))

function _print_call(io, e::Expr, ind)
  let fn = e.args[1],
      args = e.args[2:end],
      # Extract keyword args from :parameters block if present
      params_block = findfirst(a -> a isa Expr && a.head == :parameters, args),
      extra_kw = params_block !== nothing ? args[params_block].args : Expr[],
      rest = params_block !== nothing ? [args[1:params_block-1]..., args[params_block+1:end]...] : args,
      pos_args = filter(a -> !(a isa Expr && a.head == :kw), rest),
      kw_args = vcat(filter(a -> a isa Expr && a.head == :kw, rest), extra_kw),
      short = _expr_str(e)
    if length(short) + length(ind) <= 80
      print(io, "$(ind)$(short)")
    else
      # Multi-line format
      print(io, "$(ind)$(fn)(")
      let parts = vcat(map(_expr_str, pos_args), map(_kw_str, kw_args)),
          first_line = "$(fn)("
        for (i, p) in enumerate(parts)
          if i == 1
            print(io, p)
          else
            print(io, ",\n$(ind)  $(p)")
          end
        end
        print(io, ")")
      end
    end
  end
end

_kw_str(e::Expr) = "$(e.args[1])=$(_expr_str(e.args[2]))"

function _expr_str(e::Expr)
  if e.head == :call && e.args[1] === Symbol(":") && length(e.args) in (3, 4)
    # Range operator is infix: (:)(a,b) → a:b, (:)(a,step,b) → a:step:b (a for-loop reroll range).
    join(map(_expr_str, e.args[2:end]), ":")
  elseif e.head == :call
    let fn = e.args[1],
        args = e.args[2:end],
        # Extract keyword args from :parameters block if present
        params_block = findfirst(a -> a isa Expr && a.head == :parameters, args),
        extra_kw = params_block !== nothing ? args[params_block].args : Expr[],
        rest = params_block !== nothing ? [args[1:params_block-1]..., args[params_block+1:end]...] : args,
        pos = filter(a -> !(a isa Expr && a.head == :kw), rest),
        kw = vcat(filter(a -> a isa Expr && a.head == :kw, rest), extra_kw),
        parts = vcat(map(_expr_str, pos), map(_kw_str, kw))
      "$(fn)($(join(parts, ", ")))"
    end
  elseif e.head == :(=)
    "$(_expr_str(e.args[1])) = $(_expr_str(e.args[2]))"
  elseif e.head == :kw
    "$(e.args[1])=$(_expr_str(e.args[2]))"
  elseif e.head == :vect
    "[$(join(map(_expr_str, e.args), ", "))]"
  elseif e.head == :tuple
    "($(join(map(_expr_str, e.args), ", ")))"
  elseif e.head == :ref
    "$(_expr_str(e.args[1]))[$(join(map(_expr_str, e.args[2:end]), ", "))]"
  elseif e.head == :.
    join(map(a -> a isa QuoteNode ? string(a.value) : _expr_str(a), e.args), ".")
  elseif e.head == :macrocall
    let macro_name = string(e.args[1])
      if macro_name == "@raw_str"
        "raw\"$(e.args[end])\""
      elseif macro_name == "@isdefined"
        # Parenthesize so `@isdefined(revit) && …` doesn't greedily consume the `&&` as the macro arg.
        "@isdefined($(_expr_str(e.args[end])))"
      else
        string(e)
      end
    end
  elseif e.head == :&&
    "$(_expr_str(e.args[1])) && $(_expr_str(e.args[2]))"
  elseif e.head == :using
    "using $(join(map(_expr_str, e.args), ", "))"
  elseif e.head == :do
    # do block: args[1] is the call, args[2] is the -> lambda
    let call_str = _expr_str(e.args[1]),
        body = e.args[2].args[2],
        body_stmts = body.head == :block ? body.args : [body]
      "$(call_str) do; $(join(map(_expr_str, body_stmts), "; ")); end"
    end
  elseif e.head == :block
    join(map(_expr_str, e.args), "; ")
  else
    string(e)
  end
end

_expr_str(s::Symbol) = string(s)
_expr_str(x::Real) = string(meta_program(x))
_expr_str(x::Int) = string(x)
_expr_str(x::AbstractString) = repr(x)
_expr_str(x::Bool) = string(x)
_expr_str(x::QuoteNode) = string(x.value)
_expr_str(::Nothing) = "nothing"
_expr_str(x) = string(x)
