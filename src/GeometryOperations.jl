export GeometryOperationOptions,
       GeometryOperationResult,
       IntersectionElement,
       IntersectionSet,
       PointIntersection,
       CurveIntersection,
       RegionIntersection,
       ProjectionResult,
       ClosestPointsResult,
       MultiRegion,
       multi_region,
       GeometryCollection,
       geometry_collection,
       InfiniteLine,
       infinite_line,
       intersections,
       section,
       boolean,
       split,
       trim,
       project,
       closest_points,
       intersection_points,
       intersection_curves,
       intersection_regions,
       UnsupportedGeometryOperation

"""
Abstract supertype for result objects returned by geometry-computation
operations.
"""
abstract type GeometryOperationResult end

"""
Abstract supertype for individual elements inside an [`IntersectionSet`](@ref).
Concrete subtypes include [`PointIntersection`](@ref),
[`CurveIntersection`](@ref), and [`RegionIntersection`](@ref).
"""
abstract type IntersectionElement end

"""
    UnsupportedGeometryOperation(operation, operands)

Exception thrown when KhepriBase cannot compute a requested geometric
operation locally and no selected backend advertises support for it.
"""
struct UnsupportedGeometryOperation <: Exception
  operation::Symbol
  operands::Tuple
end

showerror(io::IO, e::UnsupportedGeometryOperation) =
  print(io, "Unsupported geometry operation $(e.operation) for operand types ",
        join(string.(typeof.(e.operands)), ", "))

"""
    GeometryOperationOptions(; tolerance, overlap_tolerance, method, backend)

Options shared by geometric operations such as [`intersections`](@ref),
[`section`](@ref), [`project`](@ref), and [`closest_points`](@ref).

`method` can be `:auto`, `:local`, or `:backend`. In `:auto` mode, a backend
implementation is used when the selected backend advertises support; otherwise
KhepriBase uses its local analytic implementation when one exists.
"""
@kwdef struct GeometryOperationOptions
  tolerance::Float64 = coincidence_tolerance()
  overlap_tolerance::Float64 = tolerance
  method::Symbol = :auto
  backend::Union{Nothing,Backend} = nothing
end

"""
    InfiniteLine(point, direction)
    infinite_line(point=u0(), direction=uvx())

Unbounded 3D line represented by a point and a unit direction vector. It is
used for results that cannot be represented as a finite `Path`, such as the
section of two non-parallel planes.
"""
struct InfiniteLine <: GeometryElement
  point::Loc
  direction::Vec
end

infinite_line(point::Loc=u0(), direction::Vec=uvx()) =
  InfiniteLine(point, unitized(direction))

"""
    MultiRegion(regions)
    multi_region(regions...)

Collection of independent planar regions. This is the correct result container
for planar boolean operations that produce several disconnected regions, or
regions with different hole structure.
"""
struct MultiRegion <: SurfaceGeometry
  regions::Vector{Region}
end

multi_region(regions::Region...) = MultiRegion(Region[regions...])
multi_region(regions::Vector{Region}) = MultiRegion(regions)

"""
    GeometryCollection(geometries)
    geometry_collection(geometries...)

Heterogeneous geometry result container. Use this when an operation can
produce values of different dimensionality or unrelated concrete types.
"""
struct GeometryCollection <: GeometryOperationResult
  geometries::Vector{Any}
end

geometry_collection(geometries...) = GeometryCollection(Any[geometries...])

"""
    PointIntersection(point; parameters=(;), kind=:transversal)

Intersection event whose geometric result is a point. `parameters` stores
operand-specific parameters when available, conventionally under `:first` and
`:second`. `kind` classifies the event, for example `:transversal`, `:tangent`,
`:endpoint`, or `:coincident`.
"""
struct PointIntersection <: IntersectionElement
  point::Loc
  parameters::NamedTuple
  kind::Symbol
end

PointIntersection(point::Loc; parameters=(;), kind::Symbol=:transversal) =
  PointIntersection(point, parameters, kind)

"""
    CurveIntersection(curve; parameters=(;), curve_on_first=nothing,
                      curve_on_second=nothing, kind=:section)

Intersection event whose result is a curve. `curve` is usually a `Path`, but it
can also be an unbounded value such as [`InfiniteLine`](@ref). Surface/surface
backends may also fill `curve_on_first` and `curve_on_second` with parameter
space curves on the two operands.
"""
struct CurveIntersection <: IntersectionElement
  curve::Any
  parameters::NamedTuple
  curve_on_first::Any
  curve_on_second::Any
  kind::Symbol
end

CurveIntersection(curve; parameters=(;), curve_on_first=nothing,
                  curve_on_second=nothing, kind::Symbol=:section) =
  CurveIntersection(curve, parameters, curve_on_first, curve_on_second, kind)

"""
    RegionIntersection(region; parameters=(;), kind=:overlap)

Intersection event whose result is a filled planar region or [`MultiRegion`](@ref).
"""
struct RegionIntersection <: IntersectionElement
  region::Union{Region,MultiRegion}
  parameters::NamedTuple
  kind::Symbol
end

RegionIntersection(region::Union{Region,MultiRegion}; parameters=(;),
                   kind::Symbol=:overlap) =
  RegionIntersection(region, parameters, kind)

"""
    IntersectionSet(a, b, elements; tolerance, method, exactness)

Result of [`intersections`](@ref) or [`section`](@ref). It is iterable and
contains zero or more [`IntersectionElement`](@ref) values. Results carry the
operands, tolerance, method (`:analytic`, `:backend`, etc.), and exactness
metadata because geometric operations may be exact, toleranced, or approximate.
"""
struct IntersectionSet{A,B} <: GeometryOperationResult
  operands::Tuple{A,B}
  elements::Vector{IntersectionElement}
  tolerance::Float64
  method::Symbol
  exactness::Symbol
end

IntersectionSet(a, b, elements::Vector{<:IntersectionElement};
                tolerance::Real=coincidence_tolerance(),
                method::Symbol=:analytic,
                exactness::Symbol=:toleranced) =
  IntersectionSet((a, b), IntersectionElement[elements...], Float64(tolerance), method, exactness)

Base.length(r::IntersectionSet) = length(r.elements)
Base.isempty(r::IntersectionSet) = isempty(r.elements)
Base.iterate(r::IntersectionSet, state...) = iterate(r.elements, state...)
Base.getindex(r::IntersectionSet, i::Integer) = r.elements[i]

"""
    intersection_points(result)

Return the locations from all [`PointIntersection`](@ref) elements in an
[`IntersectionSet`](@ref).
"""
intersection_points(r::IntersectionSet) =
  Loc[e.point for e in r.elements if e isa PointIntersection]

"""
    intersection_curves(result)

Return the curve payloads from all [`CurveIntersection`](@ref) elements in an
[`IntersectionSet`](@ref).
"""
intersection_curves(r::IntersectionSet) =
  Any[e.curve for e in r.elements if e isa CurveIntersection]

"""
    intersection_regions(result)

Return the region payloads from all [`RegionIntersection`](@ref) elements in an
[`IntersectionSet`](@ref).
"""
intersection_regions(r::IntersectionSet) =
  Union{Region,MultiRegion}[e.region for e in r.elements if e isa RegionIntersection]

"""
    ProjectionResult

Result of [`project`](@ref). Stores the source object, target object, projected
geometry, parameter metadata, projection distance, and method/exactness metadata.
"""
struct ProjectionResult <: GeometryOperationResult
  source::Any
  target::Any
  geometry::Any
  parameters::NamedTuple
  distance::Float64
  method::Symbol
  exactness::Symbol
end

"""
    ClosestPointsResult

Result of [`closest_points`](@ref). Stores the closest point on the first
operand, the closest point on the second operand, their parameters when
available, the distance between them, and method/exactness metadata.
"""
struct ClosestPointsResult <: GeometryOperationResult
  first::Any
  second::Any
  first_parameter::Any
  second_parameter::Any
  distance::Float64
  method::Symbol
  exactness::Symbol
end

public supports_geometry_operation,
       b_intersections,
       b_section,
       b_boolean_geometry,
       b_project_geometry,
       b_split_geometry,
       b_trim_geometry,
       b_closest_points

"""
    supports_geometry_operation(backend, operation, args...)

Return `true` when `backend` provides a native implementation for a
result-valued geometry operation and operand combination. `operation` is a
symbol such as `:intersections`, `:section`, or `:project`.
"""
supports_geometry_operation(::Backend, ::Symbol, args...) = false

"""
    b_intersections(backend, a, b, opts)

Backend hook for [`intersections`](@ref). Backends that can compute native
curve/curve, curve/surface, surface/surface, or solid/solid intersections
should return an [`IntersectionSet`](@ref) using KhepriBase result types.
"""
b_intersections(::Backend, a, b, opts::GeometryOperationOptions) =
  throw(UnsupportedGeometryOperation(:intersections, (a, b)))

"""
    b_section(backend, a, b, opts)

Backend hook for lower-dimensional section geometry, typically curves produced
by intersecting surfaces, planes, solids, or Breps.
"""
b_section(::Backend, a, b, opts::GeometryOperationOptions) =
  throw(UnsupportedGeometryOperation(:section, (a, b)))

"""
    b_boolean_geometry(backend, op, a, b, opts)

Backend hook for result-valued geometric booleans. Unlike shape CSG proxies,
this hook should return explicit Khepri geometry such as `Region`,
[`MultiRegion`](@ref), or [`GeometryCollection`](@ref).
"""
b_boolean_geometry(::Backend, op::Symbol, a, b, opts::GeometryOperationOptions) =
  throw(UnsupportedGeometryOperation(Symbol(:boolean_, op), (a, b)))

"""
    b_project_geometry(backend, a, target, opts)

Backend hook for [`project`](@ref).
"""
b_project_geometry(::Backend, a, target, opts::GeometryOperationOptions) =
  throw(UnsupportedGeometryOperation(:project, (a, target)))

"""
    b_split_geometry(backend, a, cutters, opts)

Backend hook for [`split`](@ref).
"""
b_split_geometry(::Backend, a, cutters, opts::GeometryOperationOptions) =
  throw(UnsupportedGeometryOperation(:split, (a, cutters...)))

"""
    b_trim_geometry(backend, a, cutters, opts)

Backend hook for [`trim`](@ref).
"""
b_trim_geometry(::Backend, a, cutters, opts::GeometryOperationOptions) =
  throw(UnsupportedGeometryOperation(:trim, (a, cutters...)))

"""
    b_closest_points(backend, a, b, opts)

Backend hook for [`closest_points`](@ref).
"""
b_closest_points(::Backend, a, b, opts::GeometryOperationOptions) =
  throw(UnsupportedGeometryOperation(:closest_points, (a, b)))

geometry_operation_options(; tolerance=coincidence_tolerance(),
                           overlap_tolerance=tolerance,
                           method::Symbol=:auto,
                           backend::Union{Nothing,Backend}=nothing) =
  GeometryOperationOptions(Float64(tolerance), Float64(overlap_tolerance), method, backend)

_operation_backend(opts::GeometryOperationOptions) =
  isnothing(opts.backend) && has_current_backend() ? top_backend() : opts.backend

function _with_backend_or_local(op::Symbol, local_fn::Function, backend_fn::Function, args, opts)
  b = _operation_backend(opts)
  if opts.method in (:auto, :backend) && !isnothing(b) &&
     supports_geometry_operation(b, op, args...)
    return backend_fn(b)
  elseif opts.method == :backend
    throw(UnsupportedGeometryOperation(op, args))
  else
    return local_fn()
  end
end

"""
    intersections(a, b; tolerance=coincidence_tolerance(),
                  overlap_tolerance=tolerance, method=:auto, backend=nothing)

Compute explicit geometric intersections between two objects. The result is an
[`IntersectionSet`](@ref), which may contain points, curves, regions, or be
empty. This operation is for geometric incidence; shape CSG still uses
`intersection`, `union`, and `subtraction`.
"""
intersections(a, b; kwargs...) =
  let opts = geometry_operation_options(; kwargs...)
    _with_backend_or_local(:intersections,
      () -> _analytic_intersections(a, b, opts),
      backend -> b_intersections(backend, a, b, opts),
      (a, b), opts)
  end

"""
    section(a, b; kwargs...)

Return the curve-valued part of [`intersections`](@ref), useful for operations
such as plane/plane, plane/solid, surface/surface, or Brep/Brep sectioning.
"""
section(a, b; kwargs...) =
  let opts = geometry_operation_options(; kwargs...)
    _with_backend_or_local(:section,
      () -> _analytic_section(a, b, opts),
      backend -> b_section(backend, a, b, opts),
      (a, b), opts)
  end

"""
    boolean(op, a, b; kwargs...)

Compute explicit result geometry for a set operation. `op` is typically
`:union`, `:intersection`, `:difference`, or `:symmetric_difference`. This is
separate from lazy shape CSG proxies and currently depends on local or backend
support for the operand types.
"""
boolean(op::Symbol, a, b; kwargs...) =
  let opts = geometry_operation_options(; kwargs...)
    _with_backend_or_local(Symbol(:boolean_, op),
      () -> _analytic_boolean(op, a, b, opts),
      backend -> b_boolean_geometry(backend, op, a, b, opts),
      (a, b), opts)
  end

"""
    split(a::GeometryElement, cutters...; kwargs...)

Split a geometric object by one or more cutters. The local implementation
currently supports splitting `LinePath` values at point intersections.
"""
function split(a::GeometryElement, cutters...; kwargs...)
  opts = geometry_operation_options(; kwargs...)
  _with_backend_or_local(:split,
    () -> _analytic_split(a, cutters, opts),
    backend -> b_split_geometry(backend, a, cutters, opts),
    (a, cutters...), opts)
end

"""
    trim(a::GeometryElement, cutters...; kwargs...)

Trim a geometric object by one or more cutters. This is a backend/local-kernel
extension point; unsupported operand combinations throw
[`UnsupportedGeometryOperation`](@ref).
"""
trim(a::GeometryElement, cutters...; kwargs...) =
  let opts = geometry_operation_options(; kwargs...)
    _with_backend_or_local(:trim,
      () -> _analytic_trim(a, cutters, opts),
      backend -> b_trim_geometry(backend, a, cutters, opts),
      (a, cutters...), opts)
  end

"""
    project(a, target; kwargs...)

Project geometry onto a target object. The local implementation currently
supports projecting points and line segments onto `PlaneSurface`.
"""
project(a, target; kwargs...) =
  let opts = geometry_operation_options(; kwargs...)
    _with_backend_or_local(:project,
      () -> _analytic_project(a, target, opts),
      backend -> b_project_geometry(backend, a, target, opts),
      (a, target), opts)
  end

"""
    closest_points(a, b; kwargs...)

Compute closest points between two geometric objects. The result is a
[`ClosestPointsResult`](@ref). The local implementation currently supports
points and line segments.
"""
closest_points(a, b; kwargs...) =
  let opts = geometry_operation_options(; kwargs...)
    _with_backend_or_local(:closest_points,
      () -> _analytic_closest_points(a, b, opts),
      backend -> b_closest_points(backend, a, b, opts),
      (a, b), opts)
  end

distance(a::GeometryElement, b::GeometryElement; kwargs...) =
  closest_points(a, b; kwargs...).distance
distance(a::Loc, b::GeometryElement; kwargs...) =
  closest_points(a, b; kwargs...).distance
distance(a::GeometryElement, b::Loc; kwargs...) =
  closest_points(a, b; kwargs...).distance

_empty_intersections(a, b, opts::GeometryOperationOptions; method=:analytic, exactness=:toleranced) =
  IntersectionSet(a, b, IntersectionElement[]; tolerance=opts.tolerance, method=method, exactness=exactness)

_unsupported(op::Symbol, a, b) =
  throw(UnsupportedGeometryOperation(op, (a, b)))

_reverse_parameters(params::NamedTuple) =
  (first=get(params, :second, nothing), second=get(params, :first, nothing))

_reverse_element(e::PointIntersection) =
  PointIntersection(e.point; parameters=_reverse_parameters(e.parameters), kind=e.kind)
_reverse_element(e::CurveIntersection) =
  CurveIntersection(e.curve; parameters=_reverse_parameters(e.parameters),
                    curve_on_first=e.curve_on_second,
                    curve_on_second=e.curve_on_first,
                    kind=e.kind)
_reverse_element(e::RegionIntersection) =
  RegionIntersection(e.region; parameters=_reverse_parameters(e.parameters), kind=e.kind)

_reverse_result(r::IntersectionSet) =
  IntersectionSet(r.operands[2], r.operands[1], _reverse_element.(r.elements);
                  tolerance=r.tolerance, method=r.method, exactness=r.exactness)

function _dedupe_intersection_elements(elements::Vector{IntersectionElement}, tol::Real)
  result = IntersectionElement[]
  for e in elements
    duplicate = e isa PointIntersection &&
      any(r -> r isa PointIntersection && distance(e.point, r.point) <= tol, result)
    duplicate || push!(result, e)
  end
  result
end

_point_on_segment_parameter(path::LinePath, p::Loc) =
  let p0 = in_world(path.p0),
      p1 = in_world(path.p1),
      q = in_world(p),
      d = p1 - p0,
      len = path_length(path)
    len <= zero_vector_tolerance() ? 0.0 : clamp(dot(q - p0, d) / dot(d, d), 0.0, 1.0) * len
  end

_line_point_at_fraction(path::LinePath, t::Real) =
  in_world(path.p0) + (in_world(path.p1) - in_world(path.p0)) * t

function _line_line_parameters(a::LinePath, b::LinePath)
  p = in_world(a.p0)
  q = in_world(b.p0)
  r = in_world(a.p1) - p
  s = in_world(b.p1) - q
  w = p - q
  aa = dot(r, r)
  bb = dot(r, s)
  cc = dot(s, s)
  dd = dot(r, w)
  ee = dot(s, w)
  det = aa * cc - bb * bb
  det < parallelism_tolerance() && return nothing
  ((bb * ee - cc * dd) / det, (aa * ee - bb * dd) / det)
end

function _line_overlap(a::LinePath, b::LinePath, opts::GeometryOperationOptions)
  p = in_world(a.p0)
  q0 = in_world(b.p0)
  q1 = in_world(b.p1)
  r = in_world(a.p1) - p
  s = q1 - q0
  norm(cross(r, s)) < parallelism_tolerance() || return nothing
  norm(cross(q0 - p, r)) <= opts.overlap_tolerance * max(norm(r), 1.0) || return false
  rr = dot(r, r)
  rr <= zero_vector_tolerance() && return false
  t0 = dot(q0 - p, r) / rr
  t1 = dot(q1 - p, r) / rr
  lo = max(0.0, min(t0, t1))
  hi = min(1.0, max(t0, t1))
  hi < lo - opts.overlap_tolerance && return false
  (lo, hi)
end

function _analytic_intersections(a::LinePath, b::LinePath, opts::GeometryOperationOptions)
  if (overlap = _line_overlap(a, b, opts)) !== nothing
    overlap === false && return _empty_intersections(a, b, opts)
    lo, hi = overlap
    if abs(hi - lo) * path_length(a) <= opts.overlap_tolerance
      p = _line_point_at_fraction(a, (lo + hi) / 2)
      return IntersectionSet(a, b, [PointIntersection(p;
        parameters=(first=lo * path_length(a), second=_point_on_segment_parameter(b, p)),
        kind=:endpoint)]; tolerance=opts.tolerance)
    else
      p0 = _line_point_at_fraction(a, lo)
      p1 = _line_point_at_fraction(a, hi)
      return IntersectionSet(a, b, [CurveIntersection(line_path(p0, p1);
        parameters=(first=(lo * path_length(a), hi * path_length(a)),
                    second=(_point_on_segment_parameter(b, p0), _point_on_segment_parameter(b, p1))),
        kind=:overlap)]; tolerance=opts.tolerance)
    end
  end
  params = _line_line_parameters(a, b)
  isnothing(params) && return _empty_intersections(a, b, opts)
  ta, tb = params
  if -opts.tolerance <= ta <= 1 + opts.tolerance &&
     -opts.tolerance <= tb <= 1 + opts.tolerance
    pa = _line_point_at_fraction(a, clamp(ta, 0.0, 1.0))
    pb = _line_point_at_fraction(b, clamp(tb, 0.0, 1.0))
    if distance(pa, pb) <= opts.tolerance
      p = intermediate_loc(pa, pb)
      return IntersectionSet(a, b, [PointIntersection(p;
        parameters=(first=clamp(ta, 0.0, 1.0) * path_length(a),
                    second=clamp(tb, 0.0, 1.0) * path_length(b)),
        kind=:transversal)]; tolerance=opts.tolerance)
    end
  end
  _empty_intersections(a, b, opts)
end

_circle_center(path::Union{ArcPath,CircularPath}) = path.center
_circle_radius(path::Union{ArcPath,CircularPath}) = path.radius

function _circle_plane_relation(a::Union{ArcPath,CircularPath}, b::Union{ArcPath,CircularPath}, opts)
  ca = in_world(_circle_center(a))
  cb = in_world(_circle_center(b))
  na = unitized(in_world(vz(1, _circle_center(a).cs)))
  nb = unitized(in_world(vz(1, _circle_center(b).cs)))
  abs(abs(dot(na, nb)) - 1.0) > opts.tolerance && return :skew
  abs(dot(na, cb - ca)) <= opts.tolerance ? :coplanar : :parallel_distinct
end

_normalize_angle(a::Real) = mod(Float64(a), 2π)

function _angle_delta_ccw(from::Real, to::Real)
  mod(Float64(to) - Float64(from), 2π)
end

function _path_parameter_for_angle(path::CircularPath, angle::Real)
  _normalize_angle(angle)
end

function _path_parameter_for_angle(path::ArcPath, angle::Real)
  path.amplitude >= 0 ?
    _angle_delta_ccw(path.start_angle, angle) :
    -_angle_delta_ccw(angle, path.start_angle)
end

_arc_intervals(path::CircularPath, tol) = [(0.0, 2π)]

function _arc_intervals(path::ArcPath, tol)
  abs(path.amplitude) >= 2π - tol && return [(0.0, 2π)]
  a0 = _normalize_angle(path.start_angle)
  a1 = _normalize_angle(path.start_angle + path.amplitude)
  if path.amplitude >= 0
    a0 <= a1 ? [(a0, a1)] : [(a0, 2π), (0.0, a1)]
  else
    a1 <= a0 ? [(a1, a0)] : [(a1, 2π), (0.0, a0)]
  end
end

function _angle_in_path(path::Union{ArcPath,CircularPath}, angle::Real, tol::Real)
  a = _normalize_angle(angle)
  for (lo, hi) in _arc_intervals(path, tol)
    (lo - tol <= a <= hi + tol ||
     (lo == 0.0 && abs(a - 2π) <= tol) ||
     (hi == 2π && abs(a) <= tol)) && return true
  end
  false
end

function _dedupe_points(points::Vector{Loc}, tol::Real)
  result = Loc[]
  for p in points
    any(q -> distance(p, q) <= tol, result) || push!(result, p)
  end
  result
end

function _line_circle_points(line::LinePath, circle::Union{ArcPath,CircularPath}, opts)
  cs = _circle_center(circle).cs
  p0 = in_cs(line.p0, cs)
  p1 = in_cs(line.p1, cs)
  c = _circle_center(circle)
  dx = p1.x - p0.x
  dy = p1.y - p0.y
  dz = p1.z - p0.z
  z0 = p0.z - c.z
  z1 = p1.z - c.z
  ts = Float64[]
  if abs(z0) <= opts.tolerance && abs(z1) <= opts.tolerance
    fx = p0.x - c.x
    fy = p0.y - c.y
    a = dx*dx + dy*dy
    b = 2 * (fx*dx + fy*dy)
    q = fx*fx + fy*fy - _circle_radius(circle)^2
    disc = b*b - 4*a*q
    if disc >= -opts.tolerance
      if abs(disc) <= opts.tolerance
        push!(ts, -b / (2*a))
      else
        root = sqrt(max(0.0, disc))
        push!(ts, (-b - root) / (2*a))
        push!(ts, (-b + root) / (2*a))
      end
    end
  elseif abs(dz) > opts.tolerance
    t = -z0 / dz
    x = p0.x + dx*t
    y = p0.y + dy*t
    abs(hypot(x - c.x, y - c.y) - _circle_radius(circle)) <= opts.tolerance &&
      push!(ts, t)
  end
  points = Loc[]
  for t in ts
    -opts.tolerance <= t <= 1 + opts.tolerance || continue
    p = xyz(p0.x + dx*t, p0.y + dy*t, p0.z + dz*t, cs)
    angle = atan(p.y - c.y, p.x - c.x)
    _angle_in_path(circle, angle, opts.tolerance) && push!(points, p)
  end
  _dedupe_points(points, opts.tolerance)
end

function _analytic_intersections(line::LinePath, circle::Union{ArcPath,CircularPath}, opts::GeometryOperationOptions)
  points = _line_circle_points(line, circle, opts)
  kind = length(points) == 1 ? :tangent : :transversal
  elements = IntersectionElement[
    PointIntersection(p;
      parameters=(first=_point_on_segment_parameter(line, p),
                  second=_path_parameter_for_angle(circle, atan(in_cs(p, circle.center.cs).y - circle.center.y,
                                                               in_cs(p, circle.center.cs).x - circle.center.x))),
      kind=kind)
    for p in points
  ]
  IntersectionSet(line, circle, elements; tolerance=opts.tolerance)
end

_analytic_intersections(circle::Union{ArcPath,CircularPath}, line::LinePath, opts::GeometryOperationOptions) =
  _reverse_result(_analytic_intersections(line, circle, opts))

function _same_circle(a::Union{ArcPath,CircularPath}, b::Union{ArcPath,CircularPath}, opts)
  ca = _circle_center(a)
  cb = in_cs(_circle_center(b), ca.cs)
  distance(ca, cb) <= opts.tolerance && abs(_circle_radius(a) - _circle_radius(b)) <= opts.tolerance
end

function _circle_circle_points(a::Union{ArcPath,CircularPath}, b::Union{ArcPath,CircularPath}, opts)
  cs = _circle_center(a).cs
  c1 = _circle_center(a)
  c2 = in_cs(_circle_center(b), cs)
  abs(c1.z - c2.z) <= opts.tolerance || return Loc[]
  r1 = _circle_radius(a)
  r2 = _circle_radius(b)
  dx = c2.x - c1.x
  dy = c2.y - c1.y
  d = hypot(dx, dy)
  (d > r1 + r2 + opts.tolerance ||
   d < abs(r1 - r2) - opts.tolerance ||
   d <= opts.tolerance) && return Loc[]
  aa = (r1*r1 - r2*r2 + d*d) / (2*d)
  h2 = r1*r1 - aa*aa
  h = sqrt(max(0.0, h2))
  mx = c1.x + aa*dx/d
  my = c1.y + aa*dy/d
  candidates = h <= opts.tolerance ?
    [xyz(mx, my, c1.z, cs)] :
    [xyz(mx - dy*h/d, my + dx*h/d, c1.z, cs),
     xyz(mx + dy*h/d, my - dx*h/d, c1.z, cs)]
  Loc[p for p in candidates
      if _angle_in_path(a, atan(p.y - c1.y, p.x - c1.x), opts.tolerance) &&
         _angle_in_path(b, atan(in_cs(p, _circle_center(b).cs).y - _circle_center(b).y,
                                in_cs(p, _circle_center(b).cs).x - _circle_center(b).x), opts.tolerance)]
end

function _overlap_angle_intervals(a::Union{ArcPath,CircularPath}, b::Union{ArcPath,CircularPath}, opts)
  intervals = Tuple{Float64,Float64}[]
  for (alo, ahi) in _arc_intervals(a, opts.tolerance), (blo, bhi) in _arc_intervals(b, opts.tolerance)
    lo = max(alo, blo)
    hi = min(ahi, bhi)
    hi >= lo - opts.tolerance && push!(intervals, (lo, hi))
  end
  intervals
end

function _analytic_intersections(a::Union{ArcPath,CircularPath}, b::Union{ArcPath,CircularPath}, opts::GeometryOperationOptions)
  plane_relation = _circle_plane_relation(a, b, opts)
  plane_relation == :skew && _unsupported(:intersections, a, b)
  plane_relation == :parallel_distinct && return _empty_intersections(a, b, opts)
  if _same_circle(a, b, opts)
    elements = IntersectionElement[]
    for (lo, hi) in _overlap_angle_intervals(a, b, opts)
      if abs(hi - lo) * _circle_radius(a) <= opts.overlap_tolerance
        p = xyz(a.center.x + a.radius*cos((lo + hi)/2),
                a.center.y + a.radius*sin((lo + hi)/2),
                a.center.z, a.center.cs)
        push!(elements, PointIntersection(p;
          parameters=(first=_path_parameter_for_angle(a, (lo + hi)/2),
                      second=_path_parameter_for_angle(b, (lo + hi)/2)),
          kind=:endpoint))
      elseif abs(hi - lo - 2π) <= opts.tolerance && a isa CircularPath
        push!(elements, CurveIntersection(a;
          parameters=(first=(0.0, 2π), second=(0.0, 2π)), kind=:overlap))
      else
        push!(elements, CurveIntersection(arc_path(a.center, a.radius, lo, hi - lo);
          parameters=(first=(_path_parameter_for_angle(a, lo), _path_parameter_for_angle(a, hi)),
                      second=(_path_parameter_for_angle(b, lo), _path_parameter_for_angle(b, hi))),
          kind=:overlap))
      end
    end
    return IntersectionSet(a, b, elements; tolerance=opts.tolerance)
  end
  points = _circle_circle_points(a, b, opts)
  kind = length(points) == 1 ? :tangent : :transversal
  elements = IntersectionElement[
    PointIntersection(p;
      parameters=(first=_path_parameter_for_angle(a, atan(in_cs(p, a.center.cs).y - a.center.y,
                                                     in_cs(p, a.center.cs).x - a.center.x)),
                  second=_path_parameter_for_angle(b, atan(in_cs(p, b.center.cs).y - b.center.y,
                                                      in_cs(p, b.center.cs).x - b.center.x))),
      kind=kind)
    for p in points
  ]
  IntersectionSet(a, b, elements; tolerance=opts.tolerance)
end

function _analytic_intersections(a::Path, b::Path, opts::GeometryOperationOptions)
  pieces_a = path_pieces(a)
  pieces_b = path_pieces(b)
  (length(pieces_a) == 1 && pieces_a[1] === a &&
   length(pieces_b) == 1 && pieces_b[1] === b) &&
    _unsupported(:intersections, a, b)
  elements = IntersectionElement[]
  for pa in pieces_a, pb in pieces_b
    append!(elements, _analytic_intersections(pa, pb, opts).elements)
  end
  IntersectionSet(a, b, _dedupe_intersection_elements(elements, opts.tolerance);
                  tolerance=opts.tolerance)
end

function _analytic_intersections(line::LinePath, surface::PlaneSurface, opts::GeometryOperationOptions)
  cs = surface.frame.cs
  p0 = in_cs(line.p0, cs)
  p1 = in_cs(line.p1, cs)
  z0 = p0.z - surface.frame.z
  z1 = p1.z - surface.frame.z
  if abs(z0) <= opts.tolerance && abs(z1) <= opts.tolerance
    return IntersectionSet(line, surface, [CurveIntersection(line;
      parameters=(first=path_domain(line), second=nothing), kind=:overlap)];
      tolerance=opts.tolerance)
  elseif z0 * z1 > 0
    return _empty_intersections(line, surface, opts)
  elseif abs(z1 - z0) <= opts.tolerance
    return _empty_intersections(line, surface, opts)
  else
    t = -z0 / (z1 - z0)
    -opts.tolerance <= t <= 1 + opts.tolerance || return _empty_intersections(line, surface, opts)
    p = xyz(p0.x + (p1.x - p0.x)*t,
            p0.y + (p1.y - p0.y)*t,
            surface.frame.z, cs)
    return IntersectionSet(line, surface, [PointIntersection(p;
      parameters=(first=clamp(t, 0.0, 1.0) * path_length(line),
                  second=(u=p.x - surface.frame.x, v=p.y - surface.frame.y)),
      kind=:transversal)]; tolerance=opts.tolerance)
  end
end

_analytic_intersections(surface::PlaneSurface, line::LinePath, opts::GeometryOperationOptions) =
  _reverse_result(_analytic_intersections(line, surface, opts))

function _circle_plane_points(path::Union{ArcPath,CircularPath}, surface::PlaneSurface, opts)
  circle_plane = plane_surface(path.center)
  p1, n1 = _plane_equation(circle_plane)
  p2, n2 = _plane_equation(surface)
  d = LinearAlgebra.cross(n1, n2)
  denom = dot(d, d)
  if denom <= parallelism_tolerance()
    return abs(dot(n1, p2 - p1)) <= opts.tolerance ? :coplanar : Loc[]
  end
  c1 = dot(n1, p1)
  c2 = dot(n2, p2)
  line_point = (LinearAlgebra.cross(d, n2) * c1 + LinearAlgebra.cross(n1, d) * c2) / denom
  c = in_world(path.center)
  center = SVector(c.x, c.y, c.z)
  t = dot(center - line_point, d) / denom
  closest = line_point + d * t
  dist2 = sum(abs2, closest - center)
  radius = Float64(path.radius)
  dist2 > (radius + opts.tolerance)^2 && return Loc[]
  h2 = max(radius^2 - dist2, 0.0)
  dir = d / sqrt(denom)
  candidates = h2 <= opts.tolerance^2 ?
    [closest] :
    [closest - dir * sqrt(h2), closest + dir * sqrt(h2)]
  pts = Loc[]
  for p in candidates
    loc = xyz(p[1], p[2], p[3])
    local_p = in_cs(loc, path.center.cs)
    angle = atan(local_p.y - path.center.y, local_p.x - path.center.x)
    _angle_in_path(path, angle, opts.tolerance) && push!(pts, loc)
  end
  _dedupe_points(pts, opts.tolerance)
end

function _analytic_intersections(path::Union{ArcPath,CircularPath}, surface::PlaneSurface, opts::GeometryOperationOptions)
  points = _circle_plane_points(path, surface, opts)
  points === :coplanar && return IntersectionSet(path, surface, [CurveIntersection(path;
    parameters=(first=path_domain(path), second=nothing), kind=:overlap)];
    tolerance=opts.tolerance)
  kind = length(points) == 1 ? :tangent : :transversal
  elements = IntersectionElement[
    PointIntersection(p;
      parameters=(first=_path_parameter_for_angle(path, atan(in_cs(p, path.center.cs).y - path.center.y,
                                                        in_cs(p, path.center.cs).x - path.center.x)),
                  second=(u=in_cs(p, surface.frame.cs).x - surface.frame.x,
                          v=in_cs(p, surface.frame.cs).y - surface.frame.y)),
      kind=kind)
    for p in points
  ]
  IntersectionSet(path, surface, elements; tolerance=opts.tolerance)
end

_analytic_intersections(surface::PlaneSurface, path::Union{ArcPath,CircularPath}, opts::GeometryOperationOptions) =
  _reverse_result(_analytic_intersections(path, surface, opts))

function _analytic_intersections(path::Path, surface::PlaneSurface, opts::GeometryOperationOptions)
  pieces = path_pieces(path)
  length(pieces) == 1 && pieces[1] === path && _unsupported(:intersections, path, surface)
  elements = IntersectionElement[]
  for piece in pieces
    append!(elements, _analytic_intersections(piece, surface, opts).elements)
  end
  IntersectionSet(path, surface, _dedupe_intersection_elements(elements, opts.tolerance);
                  tolerance=opts.tolerance)
end

_analytic_intersections(surface::PlaneSurface, path::Path, opts::GeometryOperationOptions) =
  _reverse_result(_analytic_intersections(path, surface, opts))

function _trimmed_plane_contains_point(surface::TrimmedSurface{PlaneSurface}, p::Loc, opts)
  q = in_cs(p, surface.base.frame.cs)
  trimmed_surface_contains(surface, q.x - surface.base.frame.x, q.y - surface.base.frame.y)
end

function _analytic_intersections(line::LinePath, surface::TrimmedSurface{PlaneSurface}, opts::GeometryOperationOptions)
  r = _analytic_intersections(line, surface.base, opts)
  elements = IntersectionElement[e for e in r.elements
    if !(e isa PointIntersection) || _trimmed_plane_contains_point(surface, e.point, opts)]
  IntersectionSet(line, surface, elements; tolerance=opts.tolerance,
                  method=r.method, exactness=r.exactness)
end

_analytic_intersections(surface::TrimmedSurface{PlaneSurface}, line::LinePath, opts::GeometryOperationOptions) =
  _reverse_result(_analytic_intersections(line, surface, opts))

function _plane_equation(surface::PlaneSurface)
  p = in_world(surface.frame)
  n = unitized(in_world(vz(1, surface.frame.cs)))
  (SVector(p.x, p.y, p.z), SVector(n.x, n.y, n.z))
end

function _analytic_intersections(a::PlaneSurface, b::PlaneSurface, opts::GeometryOperationOptions)
  p1, n1 = _plane_equation(a)
  p2, n2 = _plane_equation(b)
  d = LinearAlgebra.cross(n1, n2)
  denom = dot(d, d)
  if denom <= parallelism_tolerance()
    return abs(dot(n1, p2 - p1)) <= opts.tolerance ?
      IntersectionSet(a, b, IntersectionElement[]; tolerance=opts.tolerance, exactness=:coincident_unrepresented) :
      _empty_intersections(a, b, opts)
  end
  c1 = dot(n1, p1)
  c2 = dot(n2, p2)
  point = (LinearAlgebra.cross(d, n2) * c1 + LinearAlgebra.cross(n1, d) * c2) / denom
  line = infinite_line(xyz(point[1], point[2], point[3]), vxyz(d[1], d[2], d[3]))
  IntersectionSet(a, b, [CurveIntersection(line; kind=:unbounded_section)];
                  tolerance=opts.tolerance)
end

_analytic_intersections(a, b, opts::GeometryOperationOptions) =
  _unsupported(:intersections, a, b)

function _analytic_section(a, b, opts::GeometryOperationOptions)
  r = _analytic_intersections(a, b, opts)
  IntersectionSet(a, b, IntersectionElement[e for e in r.elements if e isa CurveIntersection];
                  tolerance=r.tolerance, method=r.method, exactness=r.exactness)
end

function _polygon_signed_area(pts)
  n = length(pts)
  sum(pts[i].x * pts[mod1(i + 1, n)].y - pts[mod1(i + 1, n)].x * pts[i].y
      for i in 1:n) / 2
end

_clip_inside(p, a, b, orientation, tol) =
  orientation * ((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)) >= -tol

function _clip_line_intersection(s, e, a, b)
  den = (e.x - s.x) * (b.y - a.y) - (e.y - s.y) * (b.x - a.x)
  abs(den) <= parallelism_tolerance() && return e
  t = ((a.x - s.x) * (b.y - a.y) - (a.y - s.y) * (b.x - a.x)) / den
  xy(s.x + t * (e.x - s.x), s.y + t * (e.y - s.y), s.cs)
end

function _convex_polygon_intersection(subject, clipper, tol)
  (isempty(subject) || isempty(clipper)) && return Loc[]
  orientation = _polygon_signed_area(clipper) >= 0 ? 1 : -1
  output = Loc[subject...]
  for i in eachindex(clipper)
    input = output
    output = Loc[]
    isempty(input) && break
    a = clipper[i]
    b = clipper[mod1(i + 1, length(clipper))]
    s = input[end]
    for e in input
      e_inside = _clip_inside(e, a, b, orientation, tol)
      s_inside = _clip_inside(s, a, b, orientation, tol)
      if e_inside
        s_inside || push!(output, _clip_line_intersection(s, e, a, b))
        push!(output, e)
      elseif s_inside
        push!(output, _clip_line_intersection(s, e, a, b))
      end
      s = e
    end
  end
  result = Loc[]
  for p in output
    (isempty(result) || distance(p, result[end]) > tol) && push!(result, p)
  end
  length(result) > 1 && distance(result[1], result[end]) <= tol && pop!(result)
  result
end

function _analytic_boolean(op::Symbol, a::Region, b::Region, opts::GeometryOperationOptions)
  op in (:intersection, :intersect) || _unsupported(Symbol(:boolean_, op), a, b)
  (isempty(inner_paths(a)) && isempty(inner_paths(b))) ||
    throw(UnsupportedGeometryOperation(:boolean_intersection_with_holes, (a, b)))
  pts = _convex_polygon_intersection(path_vertices(outer_path(a)),
                                     path_vertices(outer_path(b)),
                                     opts.tolerance)
  length(pts) < 3 && return multi_region()
  region(closed_polygonal_path(pts))
end

_analytic_boolean(op::Symbol, a, b, opts::GeometryOperationOptions) =
  _unsupported(Symbol(:boolean_, op), a, b)

function _unique_sorted_parameters(params, tol)
  sorted = sort(Float64[p for p in params])
  result = Float64[]
  for p in sorted
    if isempty(result) || abs(p - result[end]) > tol
      push!(result, p)
    end
  end
  result
end

function _analytic_split(path::LinePath, cutters, opts::GeometryOperationOptions)
  params = Float64[0.0, path_length(path)]
  for cutter in cutters
    for e in intersections(path, cutter; tolerance=opts.tolerance,
                           overlap_tolerance=opts.overlap_tolerance,
                           method=:local).elements
      if e isa PointIntersection && haskey(e.parameters, :first)
        p = e.parameters.first
        p isa Real && opts.tolerance < p < path_length(path) - opts.tolerance && push!(params, p)
      end
    end
  end
  params = _unique_sorted_parameters(params, opts.tolerance)
  path_set([line_path(location_at(path, params[i]), location_at(path, params[i + 1]))
            for i in 1:length(params)-1]...)
end

_analytic_split(a::GeometryElement, cutters, opts::GeometryOperationOptions) =
  throw(UnsupportedGeometryOperation(:split, (a, cutters...)))

_analytic_trim(a::GeometryElement, cutters, opts::GeometryOperationOptions) =
  throw(UnsupportedGeometryOperation(:trim, (a, cutters...)))

function _analytic_project(p::Loc, surface::PlaneSurface, opts::GeometryOperationOptions)
  q = in_cs(p, surface.frame.cs)
  projected = xyz(q.x, q.y, surface.frame.z, surface.frame.cs)
  ProjectionResult(p, surface, projected,
                   (target=(u=projected.x - surface.frame.x, v=projected.y - surface.frame.y),),
                   distance(p, projected), :analytic, :toleranced)
end

function _analytic_project(path::LinePath, surface::PlaneSurface, opts::GeometryOperationOptions)
  p0 = _analytic_project(path.p0, surface, opts).geometry
  p1 = _analytic_project(path.p1, surface, opts).geometry
  projected = line_path(p0, p1)
  ProjectionResult(path, surface, projected, (;), max(distance(path.p0, p0), distance(path.p1, p1)),
                   :analytic, :toleranced)
end

_analytic_project(a, target, opts::GeometryOperationOptions) =
  throw(UnsupportedGeometryOperation(:project, (a, target)))

_analytic_closest_points(a::Loc, b::Loc, opts::GeometryOperationOptions) =
  ClosestPointsResult(a, b, nothing, nothing, distance(a, b), :analytic, :exact)

function _analytic_closest_points(p::Loc, line::LinePath, opts::GeometryOperationOptions)
  p0 = in_world(line.p0)
  p1 = in_world(line.p1)
  q = in_world(p)
  d = p1 - p0
  len2 = dot(d, d)
  t = len2 <= zero_vector_tolerance() ? 0.0 : clamp(dot(q - p0, d) / len2, 0.0, 1.0)
  lp = _line_point_at_fraction(line, t)
  ClosestPointsResult(p, lp, nothing, t * path_length(line), distance(p, lp),
                      :analytic, :toleranced)
end

_analytic_closest_points(line::LinePath, p::Loc, opts::GeometryOperationOptions) =
  let r = _analytic_closest_points(p, line, opts)
    ClosestPointsResult(r.second, r.first, r.second_parameter, r.first_parameter,
                        r.distance, r.method, r.exactness)
  end

function _analytic_closest_points(a::LinePath, b::LinePath, opts::GeometryOperationOptions)
  params = _line_line_parameters(a, b)
  if isnothing(params)
    candidates = [_analytic_closest_points(a.p0, b, opts),
                  _analytic_closest_points(a.p1, b, opts),
                  _analytic_closest_points(b.p0, a, opts),
                  _analytic_closest_points(b.p1, a, opts)]
    best = candidates[argmin([c.distance for c in candidates])]
    if best.first === b.p0 || best.first === b.p1
      return ClosestPointsResult(best.second, best.first, best.second_parameter,
                                 best.first_parameter, best.distance, best.method, best.exactness)
    else
      return best
    end
  end
  ta, tb = params
  ta = clamp(ta, 0.0, 1.0)
  tb = clamp(tb, 0.0, 1.0)
  pa = _line_point_at_fraction(a, ta)
  pb = _line_point_at_fraction(b, tb)
  ClosestPointsResult(pa, pb, ta * path_length(a), tb * path_length(b),
                      distance(pa, pb), :analytic, :toleranced)
end

_analytic_closest_points(a, b, opts::GeometryOperationOptions) =
  throw(UnsupportedGeometryOperation(:closest_points, (a, b)))
