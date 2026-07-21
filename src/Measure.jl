#=
MeasureBackend — a verification backend that realizes shapes through KhepriBase's
DEFAULT lowering (the exact code path mesh backends execute) down to `b_trig` and
records world-space triangles, from which per-shape and per-statement measurements
are derived: AABB, surface area, area-weighted centroid, and signed volume (whose
SIGN gives chirality for closed shells — the only cheap detector of mirrored
placements, since bbox/centroid/size/angle metrics are all mirror-invariant).

It powers two verification loops (see stress/README.md):
- the PASS-EQUIVALENCE ORACLE: the raw `model_to_expr` output and the transformed
  program must realize the same geometry — fully headless, no Revit, localizes
  codegen-pass regressions;
- PER-ELEMENT CONFORMANCE against introspection-time ground-truth ledgers.

The backend is deliberately NOT registered as a current backend on load; create
one explicitly with `measure_backend()`.
=#

struct MEASKey end
const MEASId = Int

mutable struct MeasureBackend <: Backend{MEASKey, MEASId}
  triangles::Dict{MEASId, NTuple{3, Loc}}   # id → world-space triangle
  groups::Dict{MEASId, Vector{MEASId}}      # grouped ref → member ids (b_surface_mesh)
  order::Vector{MEASId}                     # creation order (ids are monotonic)
  next_id::Int
  transaction::Parameter{Transaction}
  refs::References{MEASKey, MEASId}
end

measure_backend() = MeasureBackend(
  Dict{MEASId, NTuple{3, Loc}}(),
  Dict{MEASId, Vector{MEASId}}(),
  MEASId[],
  1,
  Parameter{Transaction}(AutoCommitTransaction()),
  References{MEASKey, MEASId}())

_meas_next_ref!(b::MeasureBackend) =
  let id = b.next_id
    b.next_id += 1
    push!(b.order, id)
    id
  end

backend_name(b::MeasureBackend) = "Measure"
void_ref(b::MeasureBackend) = 0
new_refs(b::MeasureBackend) = MEASId[]
current_transaction(b::MeasureBackend) = b.transaction

# The single geometry sink: everything the default lowering produces funnels here.
b_trig(b::MeasureBackend, p1, p2, p3) =
  let id = _meas_next_ref!(b)
    b.triangles[id] = (in_world(p1), in_world(p2), in_world(p3))
    id
  end

# The default b_surface_mesh returns a per-face REF VECTOR, which nests one level
# too deep for shape ref containers when composed through the per-usemtl OBJ path.
# Group the face refs under a single grouped ref instead; measurement expands it.
b_surface_mesh(b::MeasureBackend, vertices, faces, mat) =
  let ids = MEASId[]
    for face in faces
      let fv = vertices[face .+ 1]
        for k in 2:(length(fv) - 1)
          push!(ids, b_trig(b, fv[1], fv[k], fv[k + 1]))
        end
      end
    end
    let gid = _meas_next_ref!(b)
      b.groups[gid] = ids
      gid
    end
  end

_expand_ids(b::MeasureBackend, ids) =
  let out = MEASId[]
    for id in ids
      haskey(b.groups, id) ? append!(out, _expand_ids(b, b.groups[id])) : push!(out, id)
    end
    unique!(out)
    out
  end

# Deletion / layers / materials: measurement bookkeeping only.
b_delete_ref(b::MeasureBackend, r::MEASId) =
  (delete!(b.triangles, r); filter!(!=(r), b.order); nothing)
b_delete_refs(b::MeasureBackend, rs::Vector{MEASId}) =
  (foreach(r -> b_delete_ref(b, r), rs); nothing)
b_delete_all_shape_refs(b::MeasureBackend) =
  (empty!(b.triangles); empty!(b.groups); empty!(b.order); nothing)
b_layer(b::MeasureBackend, name, visible, color) = _meas_next_ref!(b)
b_current_layer_ref(b::MeasureBackend) = 0
b_current_layer_ref(b::MeasureBackend, r) = nothing
b_delete_all_shapes_in_layer(b::MeasureBackend, layer) = nothing
b_get_material(b::MeasureBackend, spec::Nothing) = 0
b_get_material(b::MeasureBackend, spec) = 0
b_get_material(b::MeasureBackend, ::BackendDefault) = 0
b_material(b::MeasureBackend, name, base_color) = _meas_next_ref!(b)
b_set_view(b::MeasureBackend, camera, target, lens, aperture) = nothing
b_realistic_sky(b::MeasureBackend, date, latitude, longitude, meridian, altitude, azimuth, turbidity, sun) = nothing
b_render_and_save_view(b::MeasureBackend, path) = path

export measure_backend, MeasureBackend, reset_measure!, measure_watermark, measured_shapes,
       measure_ids_since, measure_triangles, Measurement, measured_statements,
       match_measurements

reset_measure!(b::MeasureBackend) =
  begin
    empty!(b.triangles)
    empty!(b.groups)
    empty!(b.order)
    empty!(b.refs.shapes)
    empty!(b.refs.materials)
    empty!(b.refs.layers)
    empty!(b.refs.annotations)
    empty!(b.refs.families)
    empty!(b.refs.levels)
    b.next_id = 1
    b
  end

# Statement bracketing: ids are monotonic, so the shapes a statement created are
# exactly the ids at-or-after the pre-eval watermark.
measure_watermark(b::MeasureBackend) = b.next_id
measure_ids_since(b::MeasureBackend, w) = MEASId[id for id in b.order if id >= w]

#=
Measurement of a triangle set. `signed_volume` is computed about the centroid via
the divergence theorem: its magnitude is meaningful only for closed shells, but
its SIGN flips under mirroring for any consistently-oriented mesh — the chirality
detector. `area` and the area-weighted centroid are stable for open surfaces too.
=#
struct Measurement
  n_trigs::Int
  aabb_min::Loc
  aabb_max::Loc
  centroid::Loc
  area::Float64
  signed_volume::Float64
end

_tri_area_centroid(t) =
  let (p1, p2, p3) = t,
      v1 = p2 - p1, v2 = p3 - p1,
      a = norm(cross(v1, v2)) / 2
    (a, xyz((cx(p1) + cx(p2) + cx(p3)) / 3,
            (cy(p1) + cy(p2) + cy(p3)) / 3,
            (cz(p1) + cz(p2) + cz(p3)) / 3))
  end

measure_triangles(b::MeasureBackend, ids=b.order) =
  let ts = [b.triangles[id] for id in _expand_ids(b, ids) if haskey(b.triangles, id)]
    if isempty(ts)
      Measurement(0, u0(), u0(), u0(), 0.0, 0.0)
    else
      let lo = [Inf, Inf, Inf], hi = [-Inf, -Inf, -Inf],
          asum = 0.0, cw = [0.0, 0.0, 0.0]
        for t in ts
          for p in t
            for (k, v) in enumerate((cx(p), cy(p), cz(p)))
              lo[k] = min(lo[k], v)
              hi[k] = max(hi[k], v)
            end
          end
          let (a, c) = _tri_area_centroid(t)
            asum += a
            cw[1] += a * cx(c); cw[2] += a * cy(c); cw[3] += a * cz(c)
          end
        end
        let c = asum < 1e-12 ?
                  xyz((lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2, (lo[3] + hi[3]) / 2) :
                  xyz(cw[1] / asum, cw[2] / asum, cw[3] / asum),
            sv = sum(ts) do t
              let (p1, p2, p3) = t,
                  a = p1 - c, bb = p2 - c, cc = p3 - c
                dot(a, cross(bb, cc)) / 6
              end
            end
          Measurement(length(ts), xyz(lo...), xyz(hi...), c, asum, sv)
        end
      end
    end
  end

#=
Evaluate a statement list on a MeasureBackend inside a fresh sandbox module,
bracketing each top-level statement with an id watermark, so each row reports
the geometry THAT statement created (a rerolled for-loop or a group_instance
reports its whole expansion — the natural replay granularity).

`stmts` accepts a source string, a :block/:toplevel Expr, or a statement vector.
Backend-family guards (`@isdefined(revit) && ...`) are inert unless the sandbox
defines those names. `using` statements are skipped (the sandbox already has
KhepriBase). Errors are contained per-statement and reported in the row.
=#
_stmt_list(src::AbstractString) =
  [e for e in Meta.parseall(src).args if !(e isa LineNumberNode)]
_stmt_list(e::Expr) =
  e.head in (:block, :toplevel) ?
    [x for x in e.args if !(x isa LineNumberNode)] : [e]
_stmt_list(v::AbstractVector) = collect(v)

# Per-SHAPE measurements after a full evaluation: the equivalence unit that
# survives loop rerolling and function extraction (a rerolled for-loop registers
# one shape per iteration in b.refs.shapes, exactly like the unrolled original).
measured_shapes(b::MeasureBackend) =
  [(shape=s, measurement=measure_triangles(b, ref_values(b, r)))
   for (s, r) in b.refs.shapes]

function measured_statements(stmts; b::MeasureBackend=measure_backend(),
                             env::Module=Module(gensym(:MeasureSandbox)))
  Core.eval(env, :(using KhepriBase))
  prev = current_backends()
  current_backend(b)
  rows = NamedTuple[]
  try
    for (i, e) in enumerate(_stmt_list(stmts))
      # :toplevel_comment is a printer-only pseudo-node from the header pass;
      # :inline_comment (round_parameters) wraps a real statement — evaluate that.
      e isa Expr && e.head in (:using, :toplevel_comment) && continue
      e isa Expr && e.head == :inline_comment && (e = e.args[1])
      let w = measure_watermark(b),
          err = nothing
        try
          Core.eval(env, e)
        catch ex
          err = ex
        end
        push!(rows, (index=i, expr=e,
                     measurement=measure_triangles(b, measure_ids_since(b, w)),
                     error=err))
      end
    end
  finally
    current_backends(prev)
  end
  rows
end

#=
Greedy nearest-centroid matching between two measurement lists (small models;
no need for optimal assignment). Two measurements match when their centroids
are within `tol_pos` and their AABB sizes within `tol_size` per axis; chirality
(sign of signed_volume) must agree when both magnitudes exceed `vol_floor`.
Returns matched pairs with deltas plus the unmatched residue on each side —
unmatched entries are findings (dropped/duplicated/misplaced geometry).
=#
function match_measurements(as::Vector{Measurement}, bs::Vector{Measurement};
                            tol_pos=1e-3, tol_size=1e-3, vol_floor=1e-9)
  bfree = trues(length(bs))
  pairs = Tuple{Int, Int, Float64}[]
  for (i, a) in enumerate(as)
    best, bd = 0, Inf
    for (j, m) in enumerate(bs)
      bfree[j] || continue
      let d = norm(m.centroid - a.centroid)
        d < bd && (bd = d; best = j)
      end
    end
    if best != 0 && bd <= tol_pos
      let m = bs[best],
          da = [abs((cx(m.aabb_max) - cx(m.aabb_min)) - (cx(a.aabb_max) - cx(a.aabb_min))),
                abs((cy(m.aabb_max) - cy(m.aabb_min)) - (cy(a.aabb_max) - cy(a.aabb_min))),
                abs((cz(m.aabb_max) - cz(m.aabb_min)) - (cz(a.aabb_max) - cz(a.aabb_min)))],
          chirality_ok = abs(a.signed_volume) < vol_floor || abs(m.signed_volume) < vol_floor ||
                         sign(a.signed_volume) == sign(m.signed_volume)
        if maximum(da) <= tol_size && chirality_ok
          bfree[best] = false
          push!(pairs, (i, best, bd))
        end
      end
    end
  end
  matched_a = Set(p[1] for p in pairs)
  (pairs=pairs,
   unmatched_a=[i for i in 1:length(as) if !(i in matched_a)],
   unmatched_b=[j for j in 1:length(bs) if bfree[j]])
end
