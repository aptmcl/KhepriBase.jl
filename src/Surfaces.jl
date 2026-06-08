export SurfaceGeometry,
       SurfacePatch,
       TensorProductSurface,
       PlaneSurface,
       plane_surface,
       BezierSurface,
       bezier_surface,
       BSplineSurface,
       bspline_surface,
       NurbsSurface,
       nurbs_surface,
       TrimmedSurface,
       trimmed_surface,
       surface_domain,
       surface_points,
       surface_boundary,
       surface_trims,
       surface_trim_uv_points,
       surface_trim_points,
       trimmed_surface_contains,
       normal_at,
       tessellate_surface

abstract type SurfacePatch{ClosedU,ClosedV} <: SurfaceGeometry end
abstract type TensorProductSurface{ClosedU,ClosedV} <: SurfacePatch{ClosedU,ClosedV} end

struct PlaneSurface <: SurfacePatch{false,false}
  frame::Loc
end

struct BezierSurface{ClosedU,ClosedV} <: TensorProductSurface{ClosedU,ClosedV}
  control_points::Matrix{Loc}
  degree_u::Int
  degree_v::Int
end

struct BSplineSurface{ClosedU,ClosedV,Rational} <: TensorProductSurface{ClosedU,ClosedV}
  control_points::Matrix{Loc}
  degree_u::Int
  degree_v::Int
  knots_u::Vector{Float64}
  knots_v::Vector{Float64}
  weights::Union{Nothing,Matrix{Float64}}
  domain::Tuple{Float64,Float64,Float64,Float64}
end

const NurbsSurface{ClosedU,ClosedV} = BSplineSurface{ClosedU,ClosedV,true}

struct TrimmedSurface{S<:SurfaceGeometry} <: SurfaceGeometry
  base::S
  outer::ClosedPath
  holes::Vector{<:ClosedPath}
end

plane_surface(frame::Loc=u0()) = PlaneSurface(frame)

function loc_matrix(points::Matrix{<:Loc})
  reshape(Loc[points...], size(points))
end

function loc_matrix(rows::Vector{<:Vector{<:Loc}})
  isempty(rows) && throw(ArgumentError("Surface control points cannot be empty."))
  n = length(rows[1])
  n > 0 || throw(ArgumentError("Surface control point rows cannot be empty."))
  all(row -> length(row) == n, rows) ||
    throw(ArgumentError("Surface control point rows must all have the same length."))
  mat = Matrix{Loc}(undef, length(rows), n)
  for i in eachindex(rows), j in eachindex(rows[i])
    mat[i, j] = rows[i][j]
  end
  mat
end

validate_surface_grid(cps::Matrix{Loc}) =
  min(size(cps)...) >= 2 ? cps :
    throw(ArgumentError("A surface needs at least a 2x2 control-point grid."))

function bezier_surface(points; closed_u::Bool=false, closed_v::Bool=false)
  cps = validate_surface_grid(loc_matrix(points))
  BezierSurface{closed_u,closed_v}(cps, size(cps, 1) - 1, size(cps, 2) - 1)
end

append_periodic_u(cps::Matrix, degree::Integer) =
  vcat(cps, cps[1:degree, :])
append_periodic_v(cps::Matrix, degree::Integer) =
  hcat(cps, cps[:, 1:degree])

function bspline_surface(points; degree_u::Integer=3, degree_v::Integer=3,
                         closed_u::Bool=false, closed_v::Bool=false,
                         knots_u=nothing, knots_v=nothing, domain=nothing)
  degree_u, degree_v = Int(degree_u), Int(degree_v)
  degree_u >= 1 && degree_v >= 1 ||
    throw(ArgumentError("B-spline surface degrees must be at least 1."))
  cps = validate_surface_grid(loc_matrix(points))
  size(cps, 1) > degree_u ||
    throw(ArgumentError("A degree $degree_u B-spline surface needs at least $(degree_u + 1) control rows."))
  size(cps, 2) > degree_v ||
    throw(ArgumentError("A degree $degree_v B-spline surface needs at least $(degree_v + 1) control columns."))
  closed_u && knots_u === nothing && (cps = append_periodic_u(cps, degree_u))
  closed_v && knots_v === nothing && (cps = append_periodic_v(cps, degree_v))
  ku = knots_u === nothing ?
    default_bspline_knots(size(cps, 1), degree_u, closed_u) :
    normalize_bspline_knots(knots_u, size(cps, 1), degree_u)
  kv = knots_v === nothing ?
    default_bspline_knots(size(cps, 2), degree_v, closed_v) :
    normalize_bspline_knots(knots_v, size(cps, 2), degree_v)
  dom = domain === nothing ?
    (bspline_domain(ku, degree_u, size(cps, 1))...,
     bspline_domain(kv, degree_v, size(cps, 2))...) :
    (Float64(domain[1]), Float64(domain[2]), Float64(domain[3]), Float64(domain[4]))
  dom[1] < dom[2] && dom[3] < dom[4] ||
    throw(ArgumentError("B-spline surface domain must be increasing in both directions."))
  BSplineSurface{closed_u,closed_v,false}(cps, degree_u, degree_v, ku, kv, nothing, dom)
end

function nurbs_surface(points; degree_u::Integer=3, degree_v::Integer=3,
                       closed_u::Bool=false, closed_v::Bool=false,
                       knots_u=nothing, knots_v=nothing, weights=nothing, domain=nothing)
  degree_u, degree_v = Int(degree_u), Int(degree_v)
  degree_u >= 1 && degree_v >= 1 ||
    throw(ArgumentError("NURBS surface degrees must be at least 1."))
  cps = validate_surface_grid(loc_matrix(points))
  ws = weights === nothing ? fill(1.0, size(cps)) : Float64.(weights)
  size(ws) == size(cps) ||
    throw(ArgumentError("NURBS surface weights must match the control-point grid."))
  size(cps, 1) > degree_u ||
    throw(ArgumentError("A degree $degree_u NURBS surface needs at least $(degree_u + 1) control rows."))
  size(cps, 2) > degree_v ||
    throw(ArgumentError("A degree $degree_v NURBS surface needs at least $(degree_v + 1) control columns."))
  if closed_u && knots_u === nothing
    cps = append_periodic_u(cps, degree_u)
    ws = append_periodic_u(ws, degree_u)
  end
  if closed_v && knots_v === nothing
    cps = append_periodic_v(cps, degree_v)
    ws = append_periodic_v(ws, degree_v)
  end
  ku = knots_u === nothing ?
    default_bspline_knots(size(cps, 1), degree_u, closed_u) :
    normalize_bspline_knots(knots_u, size(cps, 1), degree_u)
  kv = knots_v === nothing ?
    default_bspline_knots(size(cps, 2), degree_v, closed_v) :
    normalize_bspline_knots(knots_v, size(cps, 2), degree_v)
  dom = domain === nothing ?
    (bspline_domain(ku, degree_u, size(cps, 1))...,
     bspline_domain(kv, degree_v, size(cps, 2))...) :
    (Float64(domain[1]), Float64(domain[2]), Float64(domain[3]), Float64(domain[4]))
  dom[1] < dom[2] && dom[3] < dom[4] ||
    throw(ArgumentError("NURBS surface domain must be increasing in both directions."))
  BSplineSurface{closed_u,closed_v,true}(cps, degree_u, degree_v, ku, kv, ws, dom)
end

trimmed_surface(base::SurfaceGeometry, outer::ClosedPath, holes::ClosedPath...) =
  TrimmedSurface(base, outer, ClosedPath[holes...])

plane_surface(region::Region) =
  plane_surface(loc_from_o_vz(path_start(outer_path(region)),
                              planar_path_normal(outer_path(region))))
trimmed_surface(region::Region) =
  trimmed_surface(plane_surface(region), outer_path(region), inner_paths(region)...)

surface_domain(::PlaneSurface) = (0.0, 1.0, 0.0, 1.0)
surface_domain(::BezierSurface) = (0.0, 1.0, 0.0, 1.0)
surface_domain(s::BSplineSurface) = s.domain
surface_domain(s::TrimmedSurface) = surface_domain(s.base)

surface_boundary(s::TrimmedSurface) = s.outer
surface_boundary(r::Region) = outer_path(r)
surface_trims(s::TrimmedSurface) = s.holes
surface_trims(r::Region) = inner_paths(r)

# For non-planar trimmed surfaces, trim paths live in the base surface's
# parameter space: each path vertex's x/y coordinates are interpreted as u/v.
surface_trim_uv_points(path::ClosedPath) =
  [(Float64(raw_point(p)[1]), Float64(raw_point(p)[2])) for p in path_vertices(path)]

surface_trim_uv_points(s::TrimmedSurface{PlaneSurface}, path::ClosedPath=s.outer) =
  [(Float64(in_cs(p, s.base.frame.cs).x), Float64(in_cs(p, s.base.frame.cs).y))
   for p in path_vertices(path)]

surface_trim_uv_points(s::TrimmedSurface, path::ClosedPath=s.outer) =
  surface_trim_uv_points(path)

surface_trim_points(s::TrimmedSurface, path::ClosedPath=s.outer) =
  s.base isa PlaneSurface ?
    path_vertices(path) :
    [surface_location(s.base, u, v) for (u, v) in surface_trim_uv_points(s, path)]

function _trim_uv_bounds(points::Vector{Tuple{Float64,Float64}}, fallback)
  isempty(points) && return fallback
  us, vs = first.(points), last.(points)
  (minimum(us), maximum(us), minimum(vs), maximum(vs))
end

surface_domain(s::TrimmedSurface{PlaneSurface}) =
  _trim_uv_bounds(surface_trim_uv_points(s), surface_domain(s.base))

surface_location(s::PlaneSurface, u::Real, v::Real) =
  s.frame + vx(u, s.frame.cs) + vy(v, s.frame.cs)

surface_location(s::BezierSurface, u::Real, v::Real) =
  let u = clamp(Float64(u), 0.0, 1.0),
      v = clamp(Float64(v), 0.0, 1.0),
      cols = [bezier_eval(s.control_points[:, j], u) for j in axes(s.control_points, 2)]
    bezier_eval(cols, v)
  end

surface_parameter(t::Real, t0::Real, t1::Real, closed::Bool) =
  closed ? t0 + mod(Float64(t) - t0, t1 - t0) : clamp(Float64(t), t0, t1)

surface_weights(s::BSplineSurface{U,V,false}) where {U,V} = nothing
surface_weights(s::BSplineSurface{U,V,true}) where {U,V} = s.weights

function surface_location(s::BSplineSurface{ClosedU,ClosedV}, u::Real, v::Real) where {ClosedU,ClosedV}
  u0, u1, v0, v1 = surface_domain(s)
  u = surface_parameter(u, u0, u1, ClosedU)
  v = surface_parameter(v, v0, v1, ClosedV)
  span_u = nurbs_find_span(s.knots_u, s.degree_u, u, size(s.control_points, 1))
  span_v = nurbs_find_span(s.knots_v, s.degree_v, v, size(s.control_points, 2))
  nu = nurbs_basis_functions(span_u, u, s.degree_u, s.knots_u)
  nv = nurbs_basis_functions(span_v, v, s.degree_v, s.knots_v)
  weights = surface_weights(s)
  x = 0.0
  y = 0.0
  z = 0.0
  denom = 0.0
  for a in 0:s.degree_u, b in 0:s.degree_v
    i = span_u - s.degree_u + a
    j = span_v - s.degree_v + b
    w = isnothing(weights) ? 1.0 : weights[i, j]
    coeff = nu[a + 1] * nv[b + 1] * w
    p = raw_point(s.control_points[i, j])
    x += coeff * p[1]
    y += coeff * p[2]
    z += coeff * p[3]
    denom += coeff
  end
  abs(denom) > eps(Float64) ||
    throw(DomainError((u, v), "Surface evaluation produced a zero denominator."))
  xyz(x / denom, y / denom, z / denom, world_cs)
end

surface_location(s::TrimmedSurface, u::Real, v::Real) =
  surface_location(s.base, u, v)

#=
Finite-difference step for surface tangents.

`surface_tangent` has no closed-form derivative for the general
`SurfaceGeometry` (NURBS, trimmed, mesh-backed patches), so it estimates the
partial derivative by a central difference of `surface_location`. The step
`h` must be small enough that the secant approximates the tangent, yet large
enough that subtracting two nearby surface points does not lose all its
significant digits to catastrophic cancellation.

`h` is a fraction of the parameter-domain extent in the chosen direction
(`(u1 - u0)` or `(v1 - v0)`), so it is expressed in units of the surface's
parameter domain and scales automatically to domains of any size. The
default fraction 1e-6 sits near the classic optimum for a central difference
in Float64: truncation error grows as h^2 while rounding error grows as
eps/h, and their balance lands a few orders of magnitude below 1, so 1e-6 of
the domain keeps both error terms small for smooth patches.

The relative step is floored at `sqrt(eps(Float64))` (~1.5e-8) so that on a
degenerate or near-zero-width domain the step never collapses to a value
where `u +/- h == u` in floating point, which would make the secant
identically zero. Raise this when a surface is so finely parameterised that a
1e-6 relative step still falls inside floating-point noise; lower it for
highly curved patches where the truncation error of a larger step dominates.

See also: zero_vector_tolerance (used by `normal_at`/`location_at` to reject
the degenerate tangents this step can still produce), parallelism_tolerance.
=#

"Relative central-difference step (fraction of the parameter-domain extent) used by `surface_tangent`; floored at `sqrt(eps(Float64))`."
const surface_fd_relative_step = Parameter(1e-6)
export surface_fd_relative_step

function surface_tangent(s::SurfaceGeometry, u::Real, v::Real, axis::Symbol)
  u0, u1, v0, v1 = surface_domain(s)
  if axis === :u
    h = max((u1 - u0) * surface_fd_relative_step(), sqrt(eps(Float64)))
    a, b = max(u0, Float64(u) - h), min(u1, Float64(u) + h)
    b == a && return vx(0)
    (surface_location(s, b, v) - surface_location(s, a, v)) / (b - a)
  elseif axis === :v
    h = max((v1 - v0) * surface_fd_relative_step(), sqrt(eps(Float64)))
    a, b = max(v0, Float64(v) - h), min(v1, Float64(v) + h)
    b == a && return vy(0)
    (surface_location(s, u, b) - surface_location(s, u, a)) / (b - a)
  else
    throw(ArgumentError("Unknown surface tangent axis $(axis)."))
  end
end

normal_at(s::SurfaceGeometry, u::Real, v::Real) =
  let tu = surface_tangent(s, u, v, :u),
      tv = surface_tangent(s, u, v, :v),
      n = cross(tu, tv)
    norm(n) <= zero_vector_tolerance() ? vz(1) : unitized(n)
  end

location_at(s::SurfaceGeometry, u::Real, v::Real) =
  let p = surface_location(s, u, v),
      tu = surface_tangent(s, u, v, :u),
      tv = surface_tangent(s, u, v, :v)
    norm(tu) <= zero_vector_tolerance() || norm(tv) <= zero_vector_tolerance() ||
      norm(cross(tu, tv)) <= parallelism_tolerance() ?
      loc_from_o_vz(p, normal_at(s, u, v)) :
      loc_from_o_vx_vy(p, tu, tv)
  end

surface_points(s::BezierSurface; mode::Symbol=:control, u_count::Integer=path_smoothness_segments(), v_count::Integer=path_smoothness_segments()) =
  mode === :control ? s.control_points :
  mode === :sample ? sample_surface_points(s, u_count, v_count) :
  throw(ArgumentError("Unknown surface point mode $(mode). Use :control or :sample."))
surface_points(s::BSplineSurface; mode::Symbol=:control, u_count::Integer=path_smoothness_segments(), v_count::Integer=path_smoothness_segments()) =
  mode === :control ? s.control_points :
  mode === :sample ? sample_surface_points(s, u_count, v_count) :
  throw(ArgumentError("Unknown surface point mode $(mode). Use :control or :sample."))
surface_points(s::SurfaceGeometry; mode::Symbol=:sample, u_count::Integer=path_smoothness_segments(), v_count::Integer=path_smoothness_segments()) =
  mode === :sample ? sample_surface_points(s, u_count, v_count) :
  throw(ArgumentError("$(typeof(s)) has no control-point grid."))

#=
Closedness predicates for the sampling/tessellation grid.

A `SurfacePatch{ClosedU,ClosedV}` carries, in its type, whether each parameter
direction is periodic (a "seam" where `u1` maps back to `u0` — see
`surface_parameter`, which wraps via `mod`). Sampling and tessellation must know
this to avoid emitting a duplicated seam row/column. Any surface that does not
expose these type parameters (e.g. `Mesh`, `TrimmedSurface`) is treated as open;
`TrimmedSurface` has its own tessellation method that culls faces by trim region
and must never wrap, so the open default is the correct fallback for it too.
See also: surface_parameter, sample_surface_points, tessellate_surface.
=#
surface_closed_u(::SurfaceGeometry) = false
surface_closed_v(::SurfaceGeometry) = false
surface_closed_u(::SurfacePatch{ClosedU,ClosedV}) where {ClosedU,ClosedV} = ClosedU
surface_closed_v(::SurfacePatch{ClosedU,ClosedV}) where {ClosedU,ClosedV} = ClosedV

#=
Half-open sampling along closed (periodic) directions.

`range(t0, t1; length=count+1)` samples the CLOSED interval `[t0,t1]` inclusive,
which is correct for an open surface but wrong for a closed one: there the seam
makes `t1` map to the same physical point as `t0` (`surface_parameter` wraps via
`mod`), so the first and last rows/columns coincide. Tessellating that grid
produces a redundant strip of duplicated/zero-area quads along the seam.

For a closed direction we therefore drop the duplicated endpoint and keep only
the `count` interior-plus-start samples over the HALF-OPEN interval `[t0,t1)`.
Because we reuse the same `count+1` partition and merely discard its last point,
every retained sample is byte-identical to the corresponding open-surface sample
(no resampling drift). The omitted point is re-supplied at tessellation time by
wrapping the face indices modulo `count`, which closes the seam with a single
band of non-degenerate quads. See also: tessellate_surface, surface_closed_u.
=#
sample_axis(t0::Real, t1::Real, count::Integer, closed::Bool) =
  let ts = collect(range(t0, t1; length=Int(count) + 1))
    closed ? ts[1:end-1] : ts
  end

function sample_surface_points(s::SurfaceGeometry, u_count::Integer, v_count::Integer)
  u0, u1, v0, v1 = surface_domain(s)
  us = sample_axis(u0, u1, u_count, surface_closed_u(s))
  vs = sample_axis(v0, v1, v_count, surface_closed_v(s))
  [surface_location(s, u, v) for u in us, v in vs]
end

function _point_on_uv_segment(u::Float64, v::Float64, a::Tuple{Float64,Float64}, b::Tuple{Float64,Float64}, tol::Float64)
  ax, ay = a
  bx, by = b
  dx, dy = bx - ax, by - ay
  len2 = dx * dx + dy * dy
  if len2 <= tol * tol
    return hypot(u - ax, v - ay) <= tol
  end
  t = ((u - ax) * dx + (v - ay) * dy) / len2
  -tol <= t <= 1 + tol || return false
  t = clamp(t, 0.0, 1.0)
  hypot(u - (ax + t * dx), v - (ay + t * dy)) <= tol
end

function _point_in_uv_polygon(u::Float64, v::Float64, polygon::Vector{Tuple{Float64,Float64}})
  length(polygon) >= 3 || return false
  tol = coincidence_tolerance()
  inside = false
  previous = polygon[end]
  for current in polygon
    if _point_on_uv_segment(u, v, previous, current, tol)
      return true
    end
    x0, y0 = previous
    x1, y1 = current
    if (y0 > v) != (y1 > v)
      x_intersection = x0 + (v - y0) * (x1 - x0) / (y1 - y0)
      u < x_intersection && (inside = !inside)
    end
    previous = current
  end
  inside
end

trimmed_surface_contains(s::TrimmedSurface, u::Real, v::Real) =
  let u = Float64(u),
      v = Float64(v),
      outer = surface_trim_uv_points(s, s.outer)
    _point_in_uv_polygon(u, v, outer) &&
      !any(hole -> _point_in_uv_polygon(u, v, surface_trim_uv_points(s, hole)), s.holes)
  end

function tessellate_surface(s::SurfaceGeometry; u_count::Integer=path_smoothness_segments(), v_count::Integer=path_smoothness_segments())
  pts = sample_surface_points(s, u_count, v_count)
  nu, nv = size(pts)
  closed_u, closed_v = surface_closed_u(s), surface_closed_v(s)
  #=
  Cell counts and wrapped indexing close the seam without a duplicate row.

  An open direction has `n` samples and `n-1` cells between consecutive rows.
  A closed direction has `n` samples (the duplicated endpoint was dropped) and
  `n` cells: the extra cell bridges the last sample back to the first. We wrap
  the "next" index with `mod1`, so on the seam cell `i+1` (or `j+1`) refers to
  sample 1 again instead of running off the grid. For open directions the
  cell count is `n-1` and no wrap ever fires, reproducing the original mesh
  byte-for-byte. See also: sample_surface_points, surface_closed_u.
  =#
  ncu = closed_u ? nu : nu - 1
  ncv = closed_v ? nv : nv - 1
  wu(i) = closed_u ? mod1(i, nu) : i
  wv(j) = closed_v ? mod1(j, nv) : j
  idx(i, j) = (j - 1) * nu + (i - 1)
  faces = [[idx(wu(i), wv(j)), idx(wu(i + 1), wv(j)),
            idx(wu(i + 1), wv(j + 1)), idx(wu(i), wv(j + 1))]
           for j in 1:ncv for i in 1:ncu]
  mesh(vec(pts), faces)
end

tessellate_surface(m::Mesh; kwargs...) = m

function tessellate_surface(s::TrimmedSurface; u_count::Integer=path_smoothness_segments(), v_count::Integer=path_smoothness_segments())
  u_count, v_count = max(Int(u_count), 1), max(Int(v_count), 1)
  u0, u1, v0, v1 = surface_domain(s)
  us = collect(range(u0, u1; length=u_count + 1))
  vs = collect(range(v0, v1; length=v_count + 1))
  pts = [surface_location(s.base, u, v) for u in us, v in vs]
  nu, nv = size(pts)
  idx(i, j) = (j - 1) * nu + (i - 1)
  faces = Vector{Vector{Int}}()
  for j in 1:nv-1, i in 1:nu-1
    uc = (us[i] + us[i + 1]) / 2
    vc = (vs[j] + vs[j + 1]) / 2
    if trimmed_surface_contains(s, uc, vc)
      push!(faces, [idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])
    end
  end
  mesh(vec(pts), faces)
end
