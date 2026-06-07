export empty_path,
       point_path,
       #open_path,
       #closed_path,
       open_path_ops,
       closed_path_ops,
       ContourPath,
       SurfaceGeometry,
       PrimitivePath,
       PrimitiveOpenPath,
       PrimitiveClosedPath,
       CompositePath,
       composite_path,
       path_pieces,
       single_arc_path,
       single_arc_segment,
       LinePath,
       line_path,
       EllipticArcPath,
       elliptic_arc_path,
       BezierSpan,
       path_points,
       path_center,
       path_radius,
       path_radii,
       arc_start_angle,
       arc_amplitude,
       arc_end_angle,
       LineOp,
       ArcOp,
       CloseOp,
       ArcPath,
       arc_path,
       CircularPath,
       circular_path,
       EllipticPath,
       elliptic_path,
       RectangularPath,
       rectangular_path,
       centered_rectangular_path,
       OpenPolygonalPath,
       open_polygonal_path,
       ClosedPolygonalPath,
       closed_polygonal_path,
       PolygonalPath,
       polygonal_path,
       SplinePath,
       InterpolatingSplinePath,
       ControlSplinePath,
       BezierPath,
       BSplinePath,
       OpenInterpolatingSplinePath,
       OpenSplinePath,
       open_interpolating_spline_path,
       open_spline_path,
       ClosedInterpolatingSplinePath,
       ClosedSplinePath,
       closed_interpolating_spline_path,
       closed_spline_path,
       interpolating_spline_path,
       spline_path,
       OpenBezierPath,
       open_bezier_path,
       ClosedBezierPath,
       closed_bezier_path,
       bezier_path,
       OpenBSplinePath,
       open_bspline_path,
       ClosedBSplinePath,
       closed_bspline_path,
       bspline_path,
       OpenNurbsPath,
       open_nurbs_path,
       ClosedNurbsPath,
       closed_nurbs_path,
       NurbsPath,
       nurbs_path,
       control_point_curve_path,
       PathSet,
       path_set,
       OpenPathSequence,
       open_path_sequence,
       ClosedPathSequence,
       closed_path_sequence,
       PathSequence,
       path_sequence,
       translate,
       stroke,
       fill,
       path_length,
       length_at_parameter,
       parameter_at_length,
       divide_path_parameters_by_count,
       divide_path_by_count,
       divide_curve_parameters,
       divide_curve,
       location_at,
       location_at_length,
       path_start,
       path_end,
       path_domain,
       planar_path_normal,
       subpath,
       subpaths,
       join_paths,
       subtract_paths,
       path_vertices,
       path_frames,
       path_interpolated_frames,
       curve_interpolator,
       mirrored_path,
       mirrored_on_x,
       mirrored_on_y,
       mirrored_on_z,
       ClosedPath,
       OpenPath,
       Region,
       region,
       outer_path,
       inner_paths,
       is_closed_path,
       is_smooth_path,
       coincident_path_location,
       closed_offsetted_path,
       closed_path_for_height

#=
Coincidence tolerance.

Geometric construction frequently produces two locations that are
mathematically the same point but differ by floating-point noise.
Evaluating a spline at its endpoint and reading its last control
point should give the same Loc; in practice the basis-function
evaluation introduces drift of order 1e-14 to 1e-12 metres. Exact
`p1 == p2` comparison therefore fails on values the rest of the
system treats as equal, and we need a threshold below which two
locations count as the same point.

Compared against `distance(p1, p2)`, in metres (Khepri's canonical
unit). The default 1e-10 m (100 fm) sits well below any architectural
scale but well above the rounding of typical path operations. Users
working from measured input, or at a sub-millimetre scale where
Float64 precision becomes a noticeable fraction of the smallest
meaningful feature, may want to loosen this:

    with(coincidence_tolerance, 1e-6) do
      ...
    end

See also: truss_node_coincidence_tolerance (structural-analysis
variant at 1e-6 m, reflecting the coarser provenance of truss input
geometry).
=#

"Distance below which two locations are treated as the same point. `distance(a, b) < coincidence_tolerance()`. [metres]"
const coincidence_tolerance = Parameter(1e-10)
export coincidence_tolerance

coincident_path_location(p1::Loc, p2::Loc) = distance(p1, p2) < coincidence_tolerance()

#=
Polygonal smoothness budget.

Smooth curves (arcs, circles, ellipses, splines) get flattened to polylines
in three places: when sent to backends that don't speak curves natively,
when computing `path_vertices` for region construction, and inside
`surface(...)` whenever a boundary contains a curved subpath. The polyline
must be dense enough to render as smooth at viewing scale yet sparse enough
to keep downstream operations cheap.

This parameter sets the polyline segment count used to flatten one full
revolution (2π of parameter span) of a smooth curve. A circle and an
ellipse both use exactly this many segments; an arc of amplitude θ uses
`ceil(θ/(2π) · n)` segments, clamped to ≥ 2. For splines that already
carry at least `n+1` user-supplied control points the points are used
directly so the conversion is exact and paired splines (e.g. two parallel
boundaries of a thin strip) align by construction; sparser splines are
uniformly resampled to `n+1` vertices.

The default 64 puts each segment at ~5.6° of parameter span, which renders
as visually smooth at typical architectural scales while keeping vertex
counts modest. Override locally for high-resolution renders, or globally
for fast preview at the cost of visible facets:

    with(path_smoothness_segments, 256) do
      render_view("hi-res")
    end

Why uniform sampling rather than adaptive midpoint subdivision? Adaptive
schemes based on chord-fit collinearity terminate spuriously on periodic
curves whose period nests the probe parameters on the chord — `sin(x)` on
`[0, 2π·N]` is the canonical failure case. An earlier implementation made
matters worse by jittering the secondary probe randomly, which caused two
parallel splines bounding a thin strip to be sampled at non-corresponding
points and produce self-intersecting polygons that backends rejected.
Uniform sampling at a fixed density is reliable, paired smooth boundaries
align by construction, and the density is exposed to the user for explicit
control.

See also: `coincidence_tolerance` (governs whether the endpoints of a
closed sample loop are treated as the same point), `collinearity_tolerance`
(used by the residual adaptive paths in `path_interpolated_frames` for
sweep-style frame placement).
=#

"Polyline segments per 2π of parameter span when flattening smooth curves to polylines. [count]"
const path_smoothness_segments = Parameter(64)
export path_smoothness_segments

"Absolute tolerance used by numerical path-length integration and inverse arc-length lookup. [metres]"
const curve_length_tolerance = Parameter(1e-8)
export curve_length_tolerance


abstract type PrimitivePath{Closed} <: ContourPath{Closed} end
const PrimitiveOpenPath = PrimitivePath{false}
const PrimitiveClosedPath = PrimitivePath{true}

struct LinePath <: PrimitiveOpenPath
  p0::Loc
  p1::Loc
end

line_path(p0::Loc=u0(), p1::Loc=ux()) =
  LinePath(p0, p1)

struct EllipticArcPath <: PrimitiveOpenPath
  center::Loc
  r1::Real
  r2::Real
  start_angle::Real
  amplitude::Real
end

elliptic_arc_path(Center::Loc=u0(), R1::Real=1, R2::Real=1,
                  StartAngle::Real=0, Amplitude::Real=pi*1.0;
                  center::Loc=Center, r1::Real=R1, r2::Real=R2,
                  start_angle::Real=StartAngle, amplitude::Real=Amplitude) =
  EllipticArcPath(center, r1, r2, start_angle, amplitude)

struct BezierSpan
  control_points::Locs
end

struct CompositePath{Closed} <: ContourPath{Closed}
  pieces::Vector{OpenPath}
end

# Compatibility names for older path-sequence callers. They are aliases to
# the single composite data structure, not independent container types.
const OpenPathSequence = CompositePath{false}
const ClosedPathSequence = CompositePath{true}
const PathSequence = CompositePath



struct EmptyPath <: Path
end

empty_path() = EmptyPath()
is_empty_path(path::Path) = false
is_empty_path(path::EmptyPath) = true

struct PointPath <: Path
    location::Loc
end

point_path(loc::Loc=u0()) = PointPath(loc)

getindex(p::Path, i::Real) = location_at_length(p, i)
firstindex(p::Path) = 0
lastindex(p::Path) = path_length(p)
#getindex(p::Path, i::ClosedInterval) = subpath(p, i.left, i.right)
path_start(p::Path) = location_at_length(p, 0)
path_end(p::Path) = location_at_length(p, path_length(p))


is_closed_path(path::Path) = false
is_closed_path(path::ClosedPath) = true
is_open_path(path::Path) = ! is_closed_path(path)
# To avoid duplicate convertions, we also deal with Locs
#= A point list with fewer than two vertices is not a closed path: an empty
list has no endpoints to compare (and would raise a BoundsError on pts[1]),
and a lone point trivially satisfies pts[1] == pts[end] yet encloses no loop.
Reporting such a degenerate list as closed is what drives an empty
vertices[1:end-1] into closed_polygonal_path (-> BoundsError). =#
is_closed_path(pts::Locs) =
  length(pts) >= 2 && coincident_path_location(pts[1], pts[end])

# Default path domain, probably only makes sense for Splines.
path_domain(path::Path) = (0, 1)

# Generic map_division using path default domain divided into equal parts
map_division(f::Function, path::Path, n::Integer) =
  map_division(ϕ->f(location_at(path, ϕ)), path_domain(path)..., n, is_open_path(path))
# Even more generic map_division, with default domain and divided "naturaly"
map_division(f::Function, path::Path) =
  f.(path_interpolated_frames(path))

path_interpolated_frames(path::Path, (t0, t1)=path_domain(path), tol=collinearity_tolerance(), min_recursion=1) =
  let p0 = location_at(path, t0),
      p1 = location_at(path, t1),
      tm = (t0 + t1)/2.0,
      pm = location_at(path, tm)
    min_recursion < 0 &&
    collinear_points(p0, pm, p1, tol) &&
    let tr = # t0 + (t1 - t0)/(1.99+rand()*0.02),
             tm + (t1 - t0)*0.001,  # fixed near-midpoint offset; was randomized
        pr = location_at(path, tr) # To avoid coincidences
      collinear_points(p0, pr, p1, tol)
    end ?
      [p0, pm, p1] :
      [path_interpolated_frames(path, (t0, tm), tol, min_recursion - 1)...,
       path_interpolated_frames(path, (tm, t1), tol, min_recursion - 1)[2:end]...]
  end

## Line path
path_start(path::LinePath) = path.p0
path_end(path::LinePath) = path.p1
path_domain(path::LinePath) = (0, path_length(path))
location_at(path::LinePath, d::Real) = location_at_length(path, d)
planar_path_normal(path::LinePath) = uvz(path.p0.cs)

## Arc path
struct ArcPath <: PrimitiveOpenPath
    center::Loc
    radius::Real
    start_angle::Real
    amplitude::Real
end
arc_path(center::Loc=u0(), radius::Real=1, start_angle::Real=0, amplitude::Real=pi*1.0) =
  # pi*1.0 just to circunvent a Julia bug https://github.com/JuliaLang/julia/issues/31949.
  false ? #amplitude < 0 ?
    ArcPath(center, radius, start_angle + amplitude, - amplitude) :
    ArcPath(center, radius, start_angle, amplitude)
path_domain(path::ArcPath) = (0, path.amplitude)
location_at(path::ArcPath, ϕ::Real) =
  let s = sign(path.amplitude*1.0), # just to circunvent a Julia bug https://github.com/JuliaLang/julia/issues/31949.
      ϕ = ϕ*s
    loc_from_o_vx_vy(add_pol(path.center, path.radius, path.start_angle + ϕ),
                     vpol(1, path.start_angle + ϕ + π, path.center.cs),
                     vz(1, path.center.cs))
  end
planar_path_normal(path::ArcPath) = uvz(path.center.cs)

## Circular path
struct CircularPath <: PrimitiveClosedPath
    center::Loc
    radius::Real
end
circular_path(Center::Loc=u0(), Radius::Real=1; center::Loc=Center, radius::Real=Radius) =
  CircularPath(center, radius)
path_domain(path::CircularPath) = (0, 2π)
location_at(path::CircularPath, ϕ::Real) =
  loc_from_o_vx_vy(add_pol(path.center, path.radius, ϕ),
                   vpol(1, ϕ + π, path.center.cs),
                   vz(1, path.center.cs))
planar_path_normal(path::CircularPath) = uvz(path.center.cs)

## Elliptic path
struct EllipticPath <: PrimitiveClosedPath
  center::Loc
  r1::Real
  r2::Real
end
elliptic_path(Center::Loc=u0(), R1::Real=1, R2::Real=1; center::Loc=Center, r1::Real=R1, r2::Real=R2) =
  EllipticPath(center, r1, r2)
path_domain(path::EllipticPath) = (0, 2π)
location_at(path::EllipticPath, ϕ::Real) =
  loc_from_o_vx_vy(add_xy(path.center, path.r1*cos(ϕ), path.r2*sin(ϕ)),
                   vxy(cos(ϕ + π), sin(ϕ + π), path.center.cs),
                   vz(1, path.center.cs))
planar_path_normal(path::EllipticPath) = uvz(path.center.cs)

path_domain(path::EllipticArcPath) = (0, path.amplitude)
location_at(path::EllipticArcPath, ϕ::Real) =
  let s = sign(path.amplitude*1.0),
      α = path.start_angle + ϕ*s
    loc_from_o_vx_vy(add_xy(path.center, path.r1*cos(α), path.r2*sin(α)),
                     vxy(cos(α + π), sin(α + π), path.center.cs),
                     vz(1, path.center.cs))
  end
planar_path_normal(path::EllipticArcPath) = uvz(path.center.cs)

path_center(path::Union{ArcPath,CircularPath,EllipticPath,EllipticArcPath}) = path.center
path_radius(path::Union{ArcPath,CircularPath}) = path.radius
path_radii(path::Union{EllipticPath,EllipticArcPath}) = (path.r1, path.r2)
arc_start_angle(path::Union{ArcPath,EllipticArcPath}) = path.start_angle
arc_amplitude(path::Union{ArcPath,EllipticArcPath}) = path.amplitude
arc_end_angle(path::Union{ArcPath,EllipticArcPath}) = path.start_angle + path.amplitude

## Rectangular path
struct RectangularPath <: PrimitiveClosedPath
    corner::Loc
    dx::Real
    dy::Real
end
rectangular_path(corner::Loc=u0(), dx::Real=1, dy::Real=1) =
  RectangularPath(corner, dx, dy)
centered_rectangular_path(p, dx, dy) =
  rectangular_path(p-vxy(dx/2, dy/2), dx, dy)
path_domain(path::RectangularPath) = (0, path_length(path))
location_at(path::RectangularPath, d::Real) = location_at_length(path, d)
planar_path_normal(path::RectangularPath) = uvz(path.corner.cs)

struct OpenPolygonalPath <: PrimitiveOpenPath
    vertices::Locs
end
open_polygonal_path(vertices=[u0(), x(), xy(), y()]) =
  OpenPolygonalPath(vertices)

struct ClosedPolygonalPath <: PrimitiveClosedPath
    vertices::Locs
end
closed_polygonal_path(vertices=[u0(), x(), xy(), y()]) =
  ClosedPolygonalPath(ensure_no_repeated_locations(vertices))

PolygonalPath = Union{OpenPolygonalPath, ClosedPolygonalPath}

#= A polygonal path needs at least two vertices to span a segment. A single
vertex with coincident endpoints would otherwise route into
closed_polygonal_path(vertices[1:end-1]) on an empty list and raise a
BoundsError deep inside ensure_no_repeated_locations; fail early and clearly. =#
polygonal_path(vertices::Locs=[u0(), x(), xy(), y(), u0()]) =
  length(vertices) < 2 ?
    throw(ArgumentError("A polygonal path needs at least two vertices.")) :
  coincident_path_location(vertices[1], vertices[end]) ?
    closed_polygonal_path(vertices[1:end-1]) :
    open_polygonal_path(vertices)
polygonal_path(v::Loc, vs...) = polygonal_path([v, vs...])

ensure_no_repeated_locations(locs) =
    begin
        @assert ! coincident_path_location(locs[1], locs[end])
        locs
    end

path_start(path::PolygonalPath) = path.vertices[1]
path_end(path::OpenPolygonalPath) = path.vertices[end]
path_end(path::ClosedPath) = path_start(path)
path_domain(path::PolygonalPath) = (0, path_length(path))
location_at(path::PolygonalPath, ϕ::Real) = location_at_length(path, ϕ)
planar_path_normal(path::PolygonalPath) = vertices_normal(path_vertices(path))

#=
When the two paths meet end-to-start, the shared vertex is dropped to avoid a
zero-length segment. The collinearity test is a further optimisation: if the
vertex before the join (p1.vertices[end-1]), the shared vertex, and the vertex
after the join (p2.vertices[2]) are collinear, the shared vertex itself is
redundant and removed too. That test indexes [end-1] and [2], so a single-vertex
path (e.g. an open_polygonal_path with one location, which the constructor does
not forbid) would index vertices[0] / vertices[2] and raise a BoundsError. Guard
it with the length checks: when either path is too short there is no middle
vertex to elide, so fall through to plain concatenation keeping the shared vertex.
=#
join_paths(p1::OpenPolygonalPath, p2::OpenPolygonalPath) =
    coincident_path_location(p1.vertices[end], p2.vertices[1]) ?
      length(p1.vertices) >= 2 && length(p2.vertices) >= 2 &&
        collinear_points(p1.vertices[end-1], p1.vertices[end], p2.vertices[2]) ?
        polygonal_path([p1.vertices[1:end-1]..., p2.vertices[2:end]...]) :
        polygonal_path([p1.vertices[1:end-1]..., p2.vertices...]) :
      polygonal_path([p1.vertices..., p2.vertices...])

join_paths(p1::ArcPath, p2::OpenPolygonalPath) =
  path_sequence(p1, p2)

# Splines

abstract type SplinePath{Closed} <: ContourPath{Closed} end
abstract type InterpolatingSplinePath{Closed} <: SplinePath{Closed} end
abstract type ControlSplinePath{Closed} <: SplinePath{Closed} end
abstract type BezierPath{Closed} <: ControlSplinePath{Closed} end
abstract type BSplinePath{Closed,Rational} <: ControlSplinePath{Closed} end

struct OpenInterpolatingSplinePath <: InterpolatingSplinePath{false}
    vertices::Locs
    v0::Union{Bool,Vec}
    v1::Union{Bool,Vec}
    interpolator::ParametricSpline
    OpenInterpolatingSplinePath(vs, v0, v1) = new(vs, v0, v1, curve_interpolator(vs, false))
end
const OpenSplinePath = OpenInterpolatingSplinePath
open_interpolating_spline_path(vertices=[u0(), x(), xy(), y()], v0=false, v1=false) =
  OpenInterpolatingSplinePath(vertices, v0, v1)
open_spline_path(vertices=[u0(), x(), xy(), y()], v0=false, v1=false) =
  open_interpolating_spline_path(vertices, v0, v1)

struct ClosedInterpolatingSplinePath <: InterpolatingSplinePath{true}
    vertices::Locs
    interpolator::ParametricSpline
    ClosedInterpolatingSplinePath(vs) = new(vs, curve_interpolator(vs, true))
end
const ClosedSplinePath = ClosedInterpolatingSplinePath
closed_interpolating_spline_path(vertices=[u0(), x(), xy(), y()]) =
  let vertices = ensure_no_repeated_locations(vertices)
    ClosedInterpolatingSplinePath(vertices)
  end
closed_spline_path(vertices=[u0(), x(), xy(), y()]) =
  closed_interpolating_spline_path(vertices)

interpolating_spline_path(vertices::Locs) =
  coincident_path_location(vertices[1], vertices[end]) ?
    closed_interpolating_spline_path(vertices[1:end-1]) :
    open_interpolating_spline_path(vertices)
interpolating_spline_path(v::Loc, vs...) = interpolating_spline_path([v, vs...])
spline_path(vertices::Locs) = interpolating_spline_path(vertices)
spline_path(v::Loc, vs...) = spline_path([v, vs...])


location_at(path::InterpolatingSplinePath, ϕ::Real) =
  location_at(path.interpolator, ϕ)
location_at(interpolator::ParametricSpline, ϕ) =
  let f = interpolator(ϕ),
      d1 = derivative(interpolator, ϕ), #Interpolations.gradient(interpolator, ϕ)[1],
      d2 = derivative(interpolator, ϕ, nu=2),#u0 = path.vertices[1],
      p = xyz(f[1], f[2], f[3], world_cs),
      t = vxyz(d1[1], d1[2], d1[3], world_cs),
      n = vxyz(d2[1], d2[2], d2[3], world_cs)#,
    # n is the spline's second derivative (curvature). When it's too small
    # to define a reliable normal, fall back to a tangent-only frame.
    norm(n) <= parallelism_tolerance() ? loc_from_o_vz(p, t) : loc_from_o_vx_vy(p, n, cross(t, n))
  end

map_division(func::Function, path::OpenSplinePath, n::Integer) =
  let interpol = path.interpolator,
      fs = map_division(interpol, 0.0, 1.0, n, true),
      f = fs[1],
      d1s = map_division(t->derivative(interpol, t), 0.0, 1.0, n, true),
      d1 = d1s[1],
      d2 = derivative(interpol, 0.0, nu=2),
      p = xyz(f[1], f[2], f[3], world_cs),
      t = vxyz(d1[1], d1[2], d1[3], world_cs),
      n = vxyz(d2[1], d2[2], d2[3], world_cs)
    func.(rotation_minimizing_frames(
            # Same curvature-too-small guard as above.
            norm(n) <= parallelism_tolerance() ? loc_from_o_vz(p, t) : loc_from_o_vx_vy(p, n, cross(t, n)),
            map(p->xyz(p[1], p[2], p[3], world_cs), fs),
            map(t->vxyz(t[1], t[2], t[3], world_cs), d1s)))
  end

map_division(f::Function, path::ClosedSplinePath, n::Integer) =
  let interpolator = path.interpolator,
      ps = map_division(interpolator, 0.0, 1.0, n, false),
      vts = map_division(t->derivative(interpolator, t), 0.0, 1.0, n, false)
      #vns = map_division(t->Interpolations.hessian(interpolator, t)[1], 0.0, 1.0, n)
    f.(rotation_minimizing_frames(
         path.vertices[1],
         map(p->xyz(p[1], p[2], p[3], world_cs), ps),
         map(vy->vxyz(vy[1], vy[2], vy[3], world_cs), vts)))
  end

path_start(path::InterpolatingSplinePath) = path.vertices[1]
path_end(path::OpenSplinePath) = path.vertices[end]
path_end(path::ClosedSplinePath) = path.vertices[1]

convert(::Type{ClosedSplinePath}, path::Path) =
  closed_spline_path(path_vertices(path))

# Bezier paths

struct OpenBezierPath <: BezierPath{false}
    spans::Vector{BezierSpan}
end

struct ClosedBezierPath <: BezierPath{true}
    spans::Vector{BezierSpan}
end

_bezier_start(seg::BezierSpan) = seg.control_points[1]
_bezier_end(seg::BezierSpan) = seg.control_points[end]

function _validate_bezier_spans(spans::Vector{BezierSpan}, closed::Bool)
  isempty(spans) && throw(ArgumentError("A Bezier path needs at least one span."))
  for seg in spans
    length(seg.control_points) >= 2 ||
      throw(ArgumentError("A Bezier span needs at least two control points."))
  end
  for i in 1:max(length(spans) - 1, 0)
    coincident_path_location(_bezier_end(spans[i]), _bezier_start(spans[i + 1])) ||
      throw(ArgumentError("Bezier spans are disconnected at index $(i)."))
  end
  closed && !coincident_path_location(_bezier_end(spans[end]), _bezier_start(spans[1])) &&
    throw(ArgumentError("A closed Bezier path must end where it starts."))
  spans
end

open_bezier_path(spans::Vector{BezierSpan}) =
  OpenBezierPath(_validate_bezier_spans(spans, false))
open_bezier_path(span::BezierSpan, spans::BezierSpan...) =
  open_bezier_path(BezierSpan[span, spans...])
open_bezier_path(control_points::Locs=[u0(), x(), xy(), y()]) =
  open_bezier_path(BezierSpan(control_points))

closed_bezier_path(spans::Vector{BezierSpan}) =
  ClosedBezierPath(_validate_bezier_spans(spans, true))
closed_bezier_path(span::BezierSpan, spans::BezierSpan...) =
  closed_bezier_path(BezierSpan[span, spans...])
closed_bezier_path(control_points::Locs=[u0(), x(), xy(), u0()]) =
  closed_bezier_path(BezierSpan(control_points))

bezier_path(spans::Vector{BezierSpan}; closed::Bool=false) =
  closed ? closed_bezier_path(spans) : open_bezier_path(spans)
bezier_path(control_points::Locs=[u0(), x(), xy(), y()]; closed::Bool=false) =
  closed ? closed_bezier_path(control_points) : open_bezier_path(control_points)

function bezier_eval(values, t::Real)
  vs = collect(values)
  for k in 1:max(length(vs) - 1, 0)
    for i in 1:length(vs)-k
      vs[i] = vs[i] + (vs[i + 1] - vs[i]) * t
    end
  end
  vs[1]
end

bezier_derivative_control_points(seg::BezierSpan) =
  let n = length(seg.control_points) - 1
    [(seg.control_points[i + 1] - seg.control_points[i]) * n for i in 1:n]
  end

function location_at(seg::BezierSpan, t::Real)
  t = clamp(Float64(t), 0.0, 1.0)
  p = bezier_eval(seg.control_points, t)
  dvs = bezier_derivative_control_points(seg)
  tangent = isempty(dvs) ? vz(0, p.cs) : bezier_eval(dvs, t)
  norm(tangent) <= zero_vector_tolerance() ? p : loc_from_o_vz(p, tangent)
end

path_domain(path::BezierPath) = (0.0, Float64(length(path.spans)))
function _bezier_segment_at_parameter(path::BezierPath, t::Real)
  n = length(path.spans)
  u = is_closed_path(path) && n > 0 ? mod(Float64(t), n) : clamp(Float64(t), 0.0, n)
  idx = min(floor(Int, u) + 1, n)
  (path.spans[idx], idx == n && u == n ? 1.0 : u - (idx - 1))
end
location_at(path::BezierPath, t::Real) =
  let (seg, local_t) = _bezier_segment_at_parameter(path, t)
    location_at(seg, local_t)
  end
path_start(path::BezierPath) = _bezier_start(path.spans[1])
path_end(path::OpenBezierPath) = _bezier_end(path.spans[end])
path_end(path::ClosedBezierPath) = path_start(path)
path_length(path::BezierPath) = length_at_parameter(path, path_domain(path)[2])
location_at_length(path::BezierPath, d::Real) =
  location_at(path, parameter_at_length(path, d))

# B-spline and NURBS paths

struct OpenBSplinePath <: BSplinePath{false,false}
    control_points::Vector{Loc}
    degree::Int
    knots::Vector{Float64}
    domain::Tuple{Float64,Float64}
end

struct ClosedBSplinePath <: BSplinePath{true,false}
    control_points::Vector{Loc}
    degree::Int
    knots::Vector{Float64}
    domain::Tuple{Float64,Float64}
end

struct OpenNurbsPath <: BSplinePath{false,true}
    control_points::Vector{Loc}
    degree::Int
    knots::Vector{Float64}
    weights::Vector{Float64}
    domain::Tuple{Float64,Float64}
end

struct ClosedNurbsPath <: BSplinePath{true,true}
    control_points::Vector{Loc}
    degree::Int
    knots::Vector{Float64}
    weights::Vector{Float64}
    domain::Tuple{Float64,Float64}
end

NonRationalBSplinePath = Union{OpenBSplinePath, ClosedBSplinePath}
NurbsPath = Union{OpenNurbsPath, ClosedNurbsPath}

function default_bspline_knots(n::Integer, degree::Integer, periodic::Bool)
  n > degree || throw(ArgumentError("A B-spline path needs at least degree + 1 control points."))
  if periodic
    collect(0.0:Float64(n + degree))
  else
    interior = n - degree - 1
    vcat(fill(0.0, degree + 1),
         [i / (interior + 1) for i in 1:interior],
         fill(1.0, degree + 1))
  end
end
default_nurbs_knots(n::Integer, degree::Integer, periodic::Bool) =
  default_bspline_knots(n, degree, periodic)

function normalize_bspline_knots(knots, n::Integer, degree::Integer)
  ks = Float64.(collect(knots))
  expected = n + degree + 1
  if length(ks) == expected
    ks
  elseif length(ks) == expected - 2
    # Rhino exposes the knot vector without the two superfluous end knots.
    [ks[1], ks..., ks[end]]
  else
    throw(ArgumentError("Expected $(expected) NURBS knots ($(expected - 2) for Rhino-style knots), got $(length(ks))."))
  end
end
normalize_nurbs_knots(knots, n::Integer, degree::Integer) =
  normalize_bspline_knots(knots, n, degree)

bspline_domain(knots, degree, n) =
  (Float64(knots[degree + 1]), Float64(knots[n + 1]))
nurbs_domain(knots, degree, n) = bspline_domain(knots, degree, n)

function bspline_path(control_points::Locs; degree::Integer=3, periodic::Bool=false,
                      knots=nothing, domain=nothing)
  degree = Int(degree)
  degree >= 1 || throw(ArgumentError("B-spline degree must be at least 1."))
  cps = Loc[control_points...]
  length(cps) > degree || throw(ArgumentError("A degree $degree B-spline path needs at least $(degree + 1) control points."))

  if periodic && knots === nothing
    cps = [cps; cps[1:degree]]
  end

  ks = knots === nothing ?
    default_bspline_knots(length(cps), degree, periodic) :
    normalize_bspline_knots(knots, length(cps), degree)
  all(ks[i] <= ks[i + 1] for i in 1:length(ks)-1) ||
    throw(ArgumentError("B-spline knots must be nondecreasing."))

  dom = domain === nothing ?
    bspline_domain(ks, degree, length(cps)) :
    (Float64(domain[1]), Float64(domain[2]))
  dom[1] < dom[2] || throw(ArgumentError("B-spline domain must be increasing."))

  periodic ?
    ClosedBSplinePath(cps, degree, ks, dom) :
    OpenBSplinePath(cps, degree, ks, dom)
end

open_bspline_path(control_points::Locs=[u0(), x(), xy(), y()]; degree::Integer=3,
                  knots=nothing, domain=nothing) =
  bspline_path(control_points; degree=degree, periodic=false,
               knots=knots, domain=domain)

closed_bspline_path(control_points::Locs=[u0(), x(), xy(), y()]; degree::Integer=3,
                    knots=nothing, domain=nothing) =
  bspline_path(control_points; degree=degree, periodic=true,
               knots=knots, domain=domain)

function nurbs_path(control_points::Locs; degree::Integer=3, periodic::Bool=false,
                    knots=nothing, weights=nothing, domain=nothing)
  degree = Int(degree)
  degree >= 1 || throw(ArgumentError("NURBS degree must be at least 1."))
  cps = Loc[control_points...]
  length(cps) > degree || throw(ArgumentError("A degree $degree NURBS path needs at least $(degree + 1) control points."))
  ws = weights === nothing ? fill(1.0, length(cps)) : Float64.(collect(weights))
  length(ws) == length(cps) || throw(ArgumentError("NURBS weights must match the control point count."))

  if periodic && knots === nothing
    cps = [cps; cps[1:degree]]
    ws = [ws; ws[1:degree]]
  end

  ks = knots === nothing ?
    default_nurbs_knots(length(cps), degree, periodic) :
    normalize_nurbs_knots(knots, length(cps), degree)
  all(ks[i] <= ks[i + 1] for i in 1:length(ks)-1) ||
    throw(ArgumentError("NURBS knots must be nondecreasing."))

  dom = domain === nothing ?
    nurbs_domain(ks, degree, length(cps)) :
    (Float64(domain[1]), Float64(domain[2]))
  dom[1] < dom[2] || throw(ArgumentError("NURBS domain must be increasing."))

  periodic ?
    ClosedNurbsPath(cps, degree, ks, ws, dom) :
    OpenNurbsPath(cps, degree, ks, ws, dom)
end

open_nurbs_path(control_points::Locs=[u0(), x(), xy(), y()]; degree::Integer=3,
                knots=nothing, weights=nothing, domain=nothing) =
  nurbs_path(control_points; degree=degree, periodic=false,
             knots=knots, weights=weights, domain=domain)

closed_nurbs_path(control_points::Locs=[u0(), x(), xy(), y()]; degree::Integer=3,
                  knots=nothing, weights=nothing, domain=nothing) =
  nurbs_path(control_points; degree=degree, periodic=true,
             knots=knots, weights=weights, domain=domain)

control_point_curve_path(control_points::Locs, degree::Integer=3, periodic::Bool=false) =
  bspline_path(control_points; degree=degree, periodic=periodic)

path_domain(path::BSplinePath) = path.domain

function bspline_parameter(path::BSplinePath, t::Real)
  t0, t1 = path_domain(path)
  if is_closed_path(path)
    t == t1 && return t1
    period = t1 - t0
    t0 + mod(t - t0, period)
  else
    clamp(Float64(t), t0, t1)
  end
end
nurbs_parameter(path::NurbsPath, t::Real) = bspline_parameter(path, t)

function nurbs_find_span(knots, degree, t, n)
  t <= knots[degree + 1] && return degree + 1
  t >= knots[n + 1] && return n
  low = degree + 1
  high = n + 1
  mid = div(low + high, 2)
  while t < knots[mid] || t >= knots[mid + 1]
    if t < knots[mid]
      high = mid
    else
      low = mid
    end
    mid = div(low + high, 2)
  end
  mid
end

function nurbs_basis_functions(span, t, degree, knots)
  ns = zeros(Float64, degree + 1)
  left = zeros(Float64, degree)
  right = zeros(Float64, degree)
  ns[1] = 1.0
  for j in 1:degree
    left[j] = t - knots[span + 1 - j]
    right[j] = knots[span + j] - t
    saved = 0.0
    for r in 0:j-1
      denom = right[r + 1] + left[j - r]
      temp = iszero(denom) ? 0.0 : ns[r + 1] / denom
      ns[r + 1] = saved + right[r + 1] * temp
      saved = left[j - r] * temp
    end
    ns[j + 1] = saved
  end
  ns
end

bspline_weights(path::BSplinePath{Closed,false}) where {Closed} = nothing
bspline_weights(path::BSplinePath{Closed,true}) where {Closed} = path.weights
bspline_backend_weights(path::BSplinePath{Closed,false}) where {Closed} =
  fill(1.0, length(path.control_points))
bspline_backend_weights(path::BSplinePath{Closed,true}) where {Closed} = path.weights

function bspline_location_raw(path::BSplinePath, t::Real)
  t = bspline_parameter(path, t)
  cps = path.control_points
  degree = path.degree
  span = nurbs_find_span(path.knots, degree, t, length(cps))
  ns = nurbs_basis_functions(span, t, degree, path.knots)
  weights = bspline_weights(path)
  x = 0.0
  y = 0.0
  z = 0.0
  denom = 0.0
  for j in 0:degree
    i = span - degree + j
    w = isnothing(weights) ? 1.0 : weights[i]
    coeff = ns[j + 1] * w
    p = raw_point(cps[i])
    x += coeff * p[1]
    y += coeff * p[2]
    z += coeff * p[3]
    denom += coeff
  end
  abs(denom) > eps(Float64) ||
    throw(DomainError(t, "B-spline evaluation produced a zero denominator."))
  (x / denom, y / denom, z / denom)
end
nurbs_location_raw(path::NurbsPath, t::Real) = bspline_location_raw(path, t)

bspline_location(path::BSplinePath, t::Real) =
  let p = bspline_location_raw(path, t)
    xyz(p[1], p[2], p[3], world_cs)
  end
nurbs_location(path::NurbsPath, t::Real) = bspline_location(path, t)

function bspline_tangent(path::BSplinePath, t::Real)
  t0, t1 = path_domain(path)
  h = max((t1 - t0) * 1e-6, sqrt(eps(Float64)))
  ta = is_closed_path(path) ? t - h : max(t0, t - h)
  tb = is_closed_path(path) ? t + h : min(t1, t + h)
  tb == ta && return vz(1, world_cs)
  pa = bspline_location_raw(path, ta)
  pb = bspline_location_raw(path, tb)
  vxyz((pb[1] - pa[1]) / (tb - ta),
       (pb[2] - pa[2]) / (tb - ta),
       (pb[3] - pa[3]) / (tb - ta),
       world_cs)
end
nurbs_tangent(path::NurbsPath, t::Real) = bspline_tangent(path, t)

location_at(path::BSplinePath, t::Real) =
  let p = bspline_location(path, t),
      tangent = bspline_tangent(path, t)
    norm(tangent) <= zero_vector_tolerance() ? p : loc_from_o_vz(p, tangent)
  end

path_start(path::BSplinePath) = location_at(path, path_domain(path)[1])
path_end(path::BSplinePath{false}) = location_at(path, path_domain(path)[2])
path_end(path::BSplinePath{true}) = path_start(path)

translate(path::BSplinePath{Closed,false}, v::Vec) where {Closed} =
  bspline_path(translate(path.control_points, v); degree=path.degree,
               periodic=is_closed_path(path), knots=path.knots,
               domain=path.domain)
translate(path::BSplinePath{Closed,true}, v::Vec) where {Closed} =
  nurbs_path(translate(path.control_points, v); degree=path.degree,
             periodic=is_closed_path(path), knots=path.knots,
             weights=path.weights, domain=path.domain)
in_cs(path::BSplinePath{Closed,false}, cs::CS) where {Closed} =
  bspline_path(in_cs(path.control_points, cs); degree=path.degree,
               periodic=is_closed_path(path), knots=path.knots,
               domain=path.domain)
in_cs(path::BSplinePath{Closed,true}, cs::CS) where {Closed} =
  nurbs_path(in_cs(path.control_points, cs); degree=path.degree,
             periodic=is_closed_path(path), knots=path.knots,
             weights=path.weights, domain=path.domain)
scale(path::BSplinePath{Closed,false}, s::Real, p::Loc=u0()) where {Closed} =
  s == 1 ? path :
    bspline_path([p + (q - p)*s for q in path.control_points]; degree=path.degree,
                 periodic=is_closed_path(path), knots=path.knots,
                 domain=path.domain)
scale(path::BSplinePath{Closed,true}, s::Real, p::Loc=u0()) where {Closed} =
  s == 1 ? path :
    nurbs_path([p + (q - p)*s for q in path.control_points]; degree=path.degree,
               periodic=is_closed_path(path), knots=path.knots,
               weights=path.weights, domain=path.domain)


# There is a set of operations over Paths:
# 1. translate a path a given vector
# 2. stroke a path
# 3. fill a (presumably closed) path
# 4. compute a path location given a length from the path beginning
# 5. compute a sub path from a path, a length from the path begining and a length increment
# 6. Produce the meta representation of a path

translate(path::PointPath, v::Vec) = PointPath(path.location + v)
translate(path::LinePath, v::Vec) = LinePath(path.p0 + v, path.p1 + v)
translate(path::CircularPath, v::Vec) = CircularPath(path.center + v, path.radius)
translate(path::EllipticPath, v::Vec) = EllipticPath(path.center + v, path.r1, path.r2)
translate(path::ArcPath, v::Vec) = ArcPath(path.center + v, path.radius, path.start_angle, path.amplitude)
translate(path::EllipticArcPath, v::Vec) =
  EllipticArcPath(path.center + v, path.r1, path.r2, path.start_angle, path.amplitude)
translate(path::RectangularPath, v::Vec) = RectangularPath(path.corner + v, path.dx, path.dy)
translate(path::OpenPolygonalPath, v::Vec) = OpenPolygonalPath(translate(path.vertices, v))
translate(path::ClosedPolygonalPath, v::Vec) = ClosedPolygonalPath(translate(path.vertices, v))
translate(path::OpenSplinePath, v::Vec) = OpenSplinePath(translate(path.vertices, v), path.v0, path.v1)
translate(path::ClosedSplinePath, v::Vec) = ClosedSplinePath(translate(path.vertices, v))
translate(ps::Locs, v::Vec) = map(p->p+v, ps)
translate(seg::BezierSpan, v::Vec) =
  BezierSpan(translate(seg.control_points, v))
translate(path::BezierPath{false}, v::Vec) =
  open_bezier_path([translate(seg, v) for seg in path.spans])
translate(path::BezierPath{true}, v::Vec) =
  closed_bezier_path([translate(seg, v) for seg in path.spans])
translate(path::CompositePath{Closed}, v::Vec) where {Closed} =
  CompositePath{Closed}(OpenPath[translate(piece, v) for piece in path.pieces])

in_cs(path::PointPath, cs::CS) = PointPath(in_cs(path.location, cs))
in_cs(path::LinePath, cs::CS) = LinePath(in_cs(path.p0, cs), in_cs(path.p1, cs))
in_cs(path::CircularPath, cs::CS) = CircularPath(in_cs(path.center, cs), path.radius)
in_cs(path::EllipticPath, cs::CS) = EllipticPath(in_cs(path.center, cs), path.r1, path.r2)
in_cs(path::ArcPath, cs::CS) = ArcPath(in_cs(path.center, cs), path.radius, path.start_angle, path.amplitude)
in_cs(path::EllipticArcPath, cs::CS) =
  EllipticArcPath(in_cs(path.center, cs), path.r1, path.r2, path.start_angle, path.amplitude)
in_cs(path::RectangularPath, cs::CS) = RectangularPath(in_cs(path.corner, cs), path.dx, path.dy)
in_cs(path::OpenPolygonalPath, cs::CS) = OpenPolygonalPath(in_cs(path.vertices, cs))
in_cs(path::ClosedPolygonalPath, cs::CS) = ClosedPolygonalPath(in_cs(path.vertices, cs))
in_cs(path::OpenSplinePath, cs::CS) = OpenSplinePath(in_cs(path.vertices, cs), path.v0, path.v1)
in_cs(path::ClosedSplinePath, cs::CS) = ClosedSplinePath(in_cs(path.vertices, cs))
in_cs(ps::Locs, cs::CS) = map(p->in_cs(p, cs), ps)
in_cs(seg::BezierSpan, cs::CS) =
  BezierSpan(in_cs(seg.control_points, cs))
in_cs(path::BezierPath{false}, cs::CS) =
  open_bezier_path([in_cs(seg, cs) for seg in path.spans])
in_cs(path::BezierPath{true}, cs::CS) =
  closed_bezier_path([in_cs(seg, cs) for seg in path.spans])
in_cs(path::CompositePath{Closed}, cs::CS) where {Closed} =
  CompositePath{Closed}(OpenPath[in_cs(piece, cs) for piece in path.pieces])

scale(path::CircularPath, s::Real, p::Loc=u0()) =
  s == 1 ?
    path :
    let v = path.center - p
      circular_path(p + v*s, path.radius*s)
    end
scale(path::LinePath, s::Real, p::Loc=u0()) =
  s == 1 ? path : LinePath(p + (path.p0 - p)*s, p + (path.p1 - p)*s)
scale(path::ArcPath, s::Real, p::Loc=u0()) =
  s == 1 ? path : ArcPath(p + (path.center - p)*s, path.radius*s, path.start_angle, path.amplitude)
scale(path::EllipticPath, s::Real, p::Loc=u0()) =
  s == 1 ? path : EllipticPath(p + (path.center - p)*s, path.r1*s, path.r2*s)
scale(path::EllipticArcPath, s::Real, p::Loc=u0()) =
  s == 1 ? path : EllipticArcPath(p + (path.center - p)*s, path.r1*s, path.r2*s, path.start_angle, path.amplitude)
scale(path::RectangularPath, s::Real, p::Loc=u0()) =
  s == 1 ?
    path :
    let v = path.corner - p
      rectangular_path(p + v*s, path.dx*s, path.dy*s)
    end
scale(path::OpenPolygonalPath, s::Real, p::Loc=u0()) =
  s == 1 ?
    path :
    open_polygonal_path([p + (q-p)*s for q in path.vertices])
scale(path::ClosedPolygonalPath, s::Real, p::Loc=u0()) =
  s == 1 ?
    path :
    closed_polygonal_path([p + (q-p)*s for q in path.vertices])
scale(path::OpenSplinePath, s::Real, p::Loc=u0()) =
  s == 1 ?
    path :
    open_spline_path([p + (q-p)*s for q in path.vertices], path.v0, path.v1)
scale(path::ClosedSplinePath, s::Real, p::Loc=u0()) =
  s == 1 ?
    path :
    closed_spline_path([p + (q-p)*s for q in path.vertices])
scale(seg::BezierSpan, s::Real, p::Loc=u0()) =
  BezierSpan([p + (q-p)*s for q in seg.control_points])
scale(path::BezierPath{false}, s::Real, p::Loc=u0()) =
  s == 1 ? path : open_bezier_path([scale(seg, s, p) for seg in path.spans])
scale(path::BezierPath{true}, s::Real, p::Loc=u0()) =
  s == 1 ? path : closed_bezier_path([scale(seg, s, p) for seg in path.spans])
scale(path::CompositePath{Closed}, s::Real, p::Loc=u0()) where {Closed} =
  s == 1 ? path : CompositePath{Closed}(OpenPath[scale(piece, s, p) for piece in path.pieces])

rotate(path::Path, Δα, rot_p=u0(), rot_v=vz()) =
  Δα == 0 ?
    path :
    let rotate_vertex(p) = let v = p - rot_p
          rot_p + vcyl(pol_rho(v), pol_phi(v) + Δα, cyl_z(v))
        end
      polygonal_path([rotate_vertex(p) for p in path_vertices(path)])
    end



export planar_polygonal_path
planar_polygonal_path(path) =
  let normal = planar_path_normal(path)
    typeof(path)(in_cs(path.vertices, cs_from_o_vz(path.vertices[1], normal)))
  end

#
path_length(path::LinePath) = distance(path.p0, path.p1)
path_length(path::CircularPath) = 2*pi*path.radius
path_length(path::ArcPath) = path.radius*abs(path.amplitude)
path_length(path::EllipticPath) = length_at_parameter(path, path_domain(path)[2])
path_length(path::EllipticArcPath) = length_at_parameter(path, path_domain(path)[2])
path_length(path::RectangularPath) = 2*(path.dx + path.dy)
path_length(path::OpenPolygonalPath) = path_length(path.vertices)
path_length(path::ClosedPolygonalPath) = path_length(path.vertices) + distance(path.vertices[end], path.vertices[1])

#=
Spline path lengths are computed by polylinizing the spline at the global
`path_smoothness_segments()` density and summing the resulting polyline.
The result is a lower bound on the true (continuously parametrized) arc
length but converges as `path_smoothness_segments()` grows; for the
tessellation densities the rest of the framework already uses, the
discrepancy is well under `coincidence_tolerance()`.

Without these, generic backends (Blender, GL, Shaders, ...) that fall
through to `b_revolved_*`'s path-frame chain raised
`MethodError: no method matching path_length(::OpenSplinePath)` whenever
a `revolve(spline_shape, …)` got promoted through `convert(Path, ::Spline)`.
=#
path_length(path::OpenSplinePath) =
  path_length(convert(OpenPolygonalPath, path).vertices)
path_length(path::ClosedSplinePath) =
  let vs = convert(ClosedPolygonalPath, path).vertices
    path_length(vs) + distance(vs[end], vs[1])
  end
path_length(path::BSplinePath) =
  length_at_parameter(path, path_domain(path)[2])

# Same delegation strategy for arc-length lookup. Without these, the
# revolve fallback path (`b_revolved_curve`'s `path_frames` →
# `path_interpolated_frames` → `location_at(path, t)` for t in [0,1]) is OK,
# but `path_start` / `path_end` and any caller that drives the spline by
# arc length (e.g. `subpath`, `length_at_location`) raise the same kind of
# `MethodError` `path_length` did. Polylinizing first is consistent with
# what `path_length(::SplinePath)` already does.
location_at_length(path::OpenSplinePath, d::Real) =
  location_at_length(convert(OpenPolygonalPath, path), d)
location_at_length(path::ClosedSplinePath, d::Real) =
  location_at_length(convert(ClosedPolygonalPath, path), d)
path_length(ps::Locs) =
  isempty(ps) ? 0.0 : # an empty point list has zero length (avoids ps[1] BoundsError)
  let p = ps[1]
      l = 0.0
    for i in 2:length(ps)
        pp = ps[i]
        l += distance(p, pp)
        p = pp
    end
    l
  end

location_at_length(path::CircularPath, d::Real) =
  location_at(path, d/path.radius)
location_at_length(path::LinePath, d::Real) =
  let len = path_length(path)
    len <= coincidence_tolerance() && return path.p0
    v = unitized(path.p1 - path.p0)
    loc_from_o_vz(path.p0 + v*d, v)
  end
location_at_length(path::ArcPath, d::Real) =
  let Δα = d/path.radius,
      s = sign(path.amplitude)
    Δα <= abs(path.amplitude) + coincidence_tolerance() ?
      location_at(path, Δα*s) :
      error("Exceeded path length by ", Δα - path.amplitude)
  end
location_at_length(path::EllipticPath, d::Real) =
  location_at(path, parameter_at_length(path, d))
location_at_length(path::EllipticArcPath, d::Real) =
  location_at(path, parameter_at_length(path, d))
location_at_length(path::RectangularPath, d::Real) =
    let d = d % (2*(path.dx + path.dy)) # remove multiple periods
        p = path.corner
        for (delta, phi) in zip([path.dx, path.dy, path.dx, path.dy], [0, pi/2, pi, 3pi/2])
            if d - delta < coincidence_tolerance()
                return loc_from_o_phi(add_pol(p, d, phi), phi)
            else
                p = add_pol(p, delta, phi)
                d -= delta
            end
        end
    end
location_at_length(path::OpenPolygonalPath, d::Real) =
  let p = path.vertices[1]
    for i in 2:length(path.vertices)
      pp = path.vertices[i]
      delta = distance(p, pp)
      if d - delta < coincidence_tolerance()
        v = unitized(pp - p)
        return loc_from_o_vz(p+v*d, v)
      else
        p = pp
        d -= delta
      end
    end
    error("Exceeded path length by ", d)
  end
location_at_length(path::ClosedPolygonalPath, d::Real) =
  let p = path.vertices[1]
    for i in countfrom(1)
      pp = path.vertices[i%length(path.vertices)+1]
      delta = distance(p, pp)
      if d - delta < coincidence_tolerance()
        v = unitized(pp - p)
        return loc_from_o_vz(p+v*d, v)
      else
        p = pp
        d -= delta
      end
    end
  end
location_at_length(path::BSplinePath, d::Real) =
  location_at(path, parameter_at_length(path, d))

function path_parameter_speed(path::Path, t::Real)
  t0, t1 = path_domain(path)
  lo, hi = min(t0, t1), max(t0, t1)
  h = max((hi - lo) * 1e-6, sqrt(eps(Float64)))
  ta = clamp(Float64(t) - h, lo, hi)
  tb = clamp(Float64(t) + h, lo, hi)
  tb == ta && return 0.0
  pa = in_world(location_at(path, ta))
  pb = in_world(location_at(path, tb))
  distance(pa, pb) / (tb - ta)
end

simpson_estimate(f, a, b, fa, fm, fb) =
  (b - a) * (fa + 4fm + fb) / 6

function adaptive_simpson(f, a, b, atol, whole, fa, fm, fb, depth)
  c = (a + b) / 2
  lm = (a + c) / 2
  rm = (c + b) / 2
  flm = f(lm)
  frm = f(rm)
  left = simpson_estimate(f, a, c, fa, flm, fm)
  right = simpson_estimate(f, c, b, fm, frm, fb)
  delta = left + right - whole
  depth <= 0 || abs(delta) <= 15atol ?
    left + right + delta / 15 :
    adaptive_simpson(f, a, c, atol / 2, left, fa, flm, fm, depth - 1) +
    adaptive_simpson(f, c, b, atol / 2, right, fm, frm, fb, depth - 1)
end

function integrate_path_parameter_length(path::Path, a::Real, b::Real; atol=curve_length_tolerance(), rtol=curve_length_tolerance())
  a == b && return 0.0
  lo, hi = minmax(Float64(a), Float64(b))
  f = t -> path_parameter_speed(path, t)
  fa = f(lo)
  fb = f(hi)
  fm = f((lo + hi) / 2)
  whole = simpson_estimate(f, lo, hi, fa, fm, fb)
  tol = max(atol, abs(whole) * rtol)
  adaptive_simpson(f, lo, hi, tol, whole, fa, fm, fb, 16)
end

function length_at_parameter(path::Path, t::Real; atol=curve_length_tolerance(), rtol=curve_length_tolerance())
  t0, t1 = path_domain(path)
  t = clamp(Float64(t), min(t0, t1), max(t0, t1))
  integrate_path_parameter_length(path, t0, t; atol=atol, rtol=rtol)
end

function length_at_parameter(path::BSplinePath, t::Real; atol=curve_length_tolerance(), rtol=curve_length_tolerance())
  weights = bspline_weights(path)
  if path.degree == 1 && (isnothing(weights) || all(w -> isapprox(w, weights[1]; atol=eps(Float64)), weights))
    t0, t1 = path_domain(path)
    t = clamp(Float64(t), t0, t1)
    len = 0.0
    for span in path.degree + 1:length(path.control_points)
      a = path.knots[span]
      b = path.knots[span + 1]
      b > a || continue
      p0 = path.control_points[span - path.degree]
      p1 = path.control_points[span - path.degree + 1]
      seg_len = distance(p0, p1)
      if t >= b
        len += seg_len
      elseif t > a
        len += seg_len * (t - a) / (b - a)
        break
      end
    end
    len
  else
    t0, t1 = path_domain(path)
    t = clamp(Float64(t), min(t0, t1), max(t0, t1))
    integrate_path_parameter_length(path, t0, t; atol=atol, rtol=rtol)
  end
end

length_at_parameter(path::Union{RectangularPath,PolygonalPath}, t::Real; atol=curve_length_tolerance(), rtol=curve_length_tolerance()) =
  clamp(Float64(t), 0.0, path_length(path))
length_at_parameter(path::CircularPath, t::Real; atol=curve_length_tolerance(), rtol=curve_length_tolerance()) =
  path.radius * clamp(Float64(t), 0.0, 2π)
length_at_parameter(path::ArcPath, t::Real; atol=curve_length_tolerance(), rtol=curve_length_tolerance()) =
  path.radius * min(abs(Float64(t)), abs(path.amplitude))
length_at_parameter(path::LinePath, t::Real; atol=curve_length_tolerance(), rtol=curve_length_tolerance()) =
  clamp(Float64(t), 0.0, path_length(path))

parameter_at_length(path::Union{RectangularPath,PolygonalPath}, d::Real; atol=curve_length_tolerance(), rtol=curve_length_tolerance()) =
  clamp(Float64(d), 0.0, path_length(path))
parameter_at_length(path::CircularPath, d::Real; atol=curve_length_tolerance(), rtol=curve_length_tolerance()) =
  clamp(Float64(d), 0.0, path_length(path)) / path.radius
parameter_at_length(path::ArcPath, d::Real; atol=curve_length_tolerance(), rtol=curve_length_tolerance()) =
  sign(path.amplitude) * clamp(Float64(d), 0.0, path_length(path)) / path.radius
parameter_at_length(path::LinePath, d::Real; atol=curve_length_tolerance(), rtol=curve_length_tolerance()) =
  clamp(Float64(d), 0.0, path_length(path))

function parameter_at_length(path::Path, d::Real; atol=curve_length_tolerance(), rtol=curve_length_tolerance())
  total = path_length(path)
  target = clamp(Float64(d), 0.0, total)
  t0, t1 = path_domain(path)
  target <= atol && return t0
  total - target <= atol && return t1
  lo, hi = Float64(t0), Float64(t1)
  for _ in 1:64
    mid = (lo + hi) / 2
    if length_at_parameter(path, mid; atol=atol, rtol=rtol) < target
      lo = mid
    else
      hi = mid
    end
  end
  (lo + hi) / 2
end

function divide_path_parameters_by_count(path::Path, count::Integer; include_ends::Bool=true)
  count = Int(count)
  count <= 0 && return Float64[]
  total = path_length(path)
  lengths = collect(range(0.0, total; length=count + 1))
  !include_ends && (lengths = lengths[2:end-1])
  [parameter_at_length(path, d) for d in lengths]
end

divide_path_by_count(path::Path, count::Integer; include_ends::Bool=true) =
  [location_at(path, t) for t in divide_path_parameters_by_count(path, count; include_ends=include_ends)]

curve_path(curve::Path) = curve
curve_path(curve) = shape_path(curve)
divide_curve_parameters(curve, count::Integer, include_ends::Bool=true) =
  divide_path_parameters_by_count(curve_path(curve), count; include_ends=include_ends)
divide_curve(curve, count::Integer, include_ends::Bool=true) =
  divide_path_by_count(curve_path(curve), count; include_ends=include_ends)

subpath_starting_at(path::Path, d::Real) = subpath(path, d, path_length(path))
subpath_ending_at(path::Path, d::Real) = subpath(path, 0, d)

subpath(path::CircularPath, a::Real, b::Real) =
  arc_path(path.center, path.radius, a/path.radius, (b-a)/path.radius)
subpath(path::ArcPath, a::Real, b::Real) =
  b <= path_length(path) + coincidence_tolerance() ?
      arc_path(path.center,
               path.radius,
               path.start_angle + a/path.radius*sign(path.amplitude),
               (b - a)/path.radius*sign(path.amplitude)) :
    error("Exceeded path length by ", path.amplitude - b/path.radius)
subpath(path::RectangularPath, a::Real, b::Real) =
  subpath(convert(ClosedPolygonalPath, path), a, b)
subpath(path::ClosedPolygonalPath, a::Real, b::Real) =
  subpath(convert(OpenPolygonalPath, path), a, b)
subpath(path::OpenPolygonalPath, a::Real, b::Real) =
  subpath_starting_at(subpath_ending_at(path, b), a)

subpath_starting_at(path::OpenPolygonalPath, d::Real) =
    let pts = path.vertices
        p1 = pts[1]
        for i in 2:length(pts)
            p2 = pts[i]
            delta = distance(p1, p2)
            diff = d - delta
            if diff < 0
                mp = p1 + (p2 - p1)*d/delta
                return open_polygonal_path([mp, pts[i:end]...])
            elseif abs(diff) < coincidence_tolerance()
                return open_polygonal_path(i == length(pts) ? [pts[i], pts[i]] : pts[i:end])
            else
                p1 = p2
                d -= delta
            end
        end
        abs(d) < coincidence_tolerance() ?
          path :
          error("Exceeded path length by ", d)
    end

subpath_ending_at(path::OpenPolygonalPath, d::Real) =
    let pts = path.vertices
        p1 = pts[1]
        for i in 2:length(pts)
            p2 = pts[i]
            delta = distance(p1, p2)
            diff = d - delta
            if diff < 0
                mp = p1 + (p2 - p1)*d/delta
                return open_polygonal_path([pts[1:i-1]..., mp])
            elseif abs(diff) < coincidence_tolerance()
                return open_polygonal_path(pts[1:i])
            else
              p1 = p2
              d -= delta
            end
        end
        abs(d) < coincidence_tolerance() ?
          path :
          error("Exceeded path length by ", d)
    end


# A composite path is a connected sequence of open paths. Closed composites
# close by connecting the final piece back to the first piece's start.

function ensure_connected_paths(paths)
  ps = OpenPath[p for p in paths if !is_empty_path(p)]
  for i in 1:max(length(ps) - 1, 0)
    p, q = ps[i], ps[i + 1]
    coincident_path_location(path_end(p), path_start(q)) ||
      throw(ArgumentError("Composite path is disconnected: $(typeof(p)) ends at $(path_end(p)), but $(typeof(q)) starts at $(path_start(q))."))
  end
  ps
end

function composite_path(paths::Vector{<:OpenPath}; closed::Bool=false)
  pieces = ensure_connected_paths(paths)
  isempty(pieces) && return empty_path()
  closed && !coincident_path_location(path_end(pieces[end]), path_start(pieces[1])) &&
    throw(ArgumentError("A closed composite path must end where it starts."))
  CompositePath{closed}(pieces)
end
composite_path(paths::OpenPath...; closed::Bool=false) =
  composite_path(OpenPath[paths...]; closed=closed)

open_path_sequence(paths...) =
  composite_path(ensure_connected_paths([paths...]); closed=false)
closed_path_sequence(paths...) =
  let pieces = ensure_connected_paths([paths...])
    isempty(pieces) && throw(ArgumentError("A closed composite path needs at least one piece."))
    start = path_start(pieces[1])
    finish = path_end(pieces[end])
    !coincident_path_location(start, finish) && push!(pieces, line_path(finish, start))
    CompositePath{true}(pieces)
  end

join_paths(p1::OpenPolygonalPath, p2::CompositePath{false}) =
  path_sequence(p1, p2.pieces...)

join_paths(p1::ArcPath, p2::CompositePath{false}) =
  path_sequence(p1, p2.pieces...)

# HACK : Include treatment of empty paths.

path_sequence(paths...) =
  let pieces = ensure_connected_paths([paths...])
    isempty(pieces) ? empty_path() :
    coincident_path_location(path_start(pieces[1]), path_end(pieces[end])) ?
      CompositePath{true}(pieces) :
      CompositePath{false}(pieces)
  end

meta_program(p::OpenPolygonalPath) =
    Expr(:call, :open_polygonal_path, meta_program(p.vertices))

meta_program(p::ClosedPolygonalPath) =
  Expr(:call, :closed_polygonal_path, meta_program(p.vertices))

meta_program(p::ArcPath) =
  Expr(:call, :arc_path, meta_program(p.center), meta_program(p.radius),
       meta_program(p.start_angle), meta_program(p.amplitude))

meta_program(p::CircularPath) =
  Expr(:call, :circular_path, meta_program(p.center), meta_program(p.radius))

meta_program(p::EllipticPath) =
  Expr(:call, :elliptic_path, meta_program(p.center), meta_program(p.r1), meta_program(p.r2))

meta_program(p::RectangularPath) =
  Expr(:call, :rectangular_path, meta_program(p.corner), meta_program(p.dx), meta_program(p.dy))

meta_program(p::OpenSplinePath) =
  Expr(:call, :open_spline_path, Expr(:vect, map(meta_program, p.vertices)...))

meta_program(p::ClosedSplinePath) =
  Expr(:call, :closed_spline_path, Expr(:vect, map(meta_program, p.vertices)...))

meta_program(p::BezierPath) =
  Expr(:call, :bezier_path,
       Expr(:vect, map(meta_program, p.spans)...),
       Expr(:kw, :closed, is_closed_path(p)))

meta_program(p::NonRationalBSplinePath) =
  Expr(:call, :bspline_path,
       Expr(:vect, map(meta_program, p.control_points)...),
       Expr(:kw, :degree, p.degree),
       Expr(:kw, :periodic, is_closed_path(p)),
       Expr(:kw, :knots, p.knots),
       Expr(:kw, :domain, p.domain))

meta_program(p::NurbsPath) =
  Expr(:call, :nurbs_path,
       Expr(:vect, map(meta_program, p.control_points)...),
       Expr(:kw, :degree, p.degree),
       Expr(:kw, :periodic, is_closed_path(p)),
       Expr(:kw, :knots, p.knots),
       Expr(:kw, :weights, p.weights),
       Expr(:kw, :domain, p.domain))

meta_program(p::LinePath) =
  Expr(:call, :line_path, meta_program(p.p0), meta_program(p.p1))

meta_program(p::EllipticArcPath) =
  Expr(:call, :elliptic_arc_path, meta_program(p.center), meta_program(p.r1),
       meta_program(p.r2), meta_program(p.start_angle), meta_program(p.amplitude))

meta_program(s::BezierSpan) =
  Expr(:call, :BezierSpan, Expr(:vect, map(meta_program, s.control_points)...))

meta_program(p::CompositePath) =
  Expr(:call, :composite_path,
       Expr(:vect, map(meta_program, p.pieces)...),
       Expr(:kw, :closed, is_closed_path(p)))

# Path can be made of subparts
abstract type PathOp end
struct LineOp <: PathOp
  vec::Vec
end
struct CloseOp <: PathOp end
struct ArcOp <: PathOp
    radius::Real
    start_angle::Real
    amplitude::Real
end
struct LineXThenYOp <: PathOp
  vec::Vec
end
struct LineYThenXOp <: PathOp
  vec::Vec
end

struct PathOps <: Path
  start::Loc
  ops::Vector{<:PathOp}
  closed::Bool
end
open_path_ops(start, ops...) = _path_ops_composite_path(start, [ops...], false)
closed_path_ops(start, ops...) = _path_ops_composite_path(start, [ops...], true)

path_length(path::PathOps) =
  let len = mapreduce(path_length, +, path.ops, init=0)
    path.closed ?
      len + distance(path.start, location_at_length(path, len)) :
      len
  end
path_length(op::LineOp) = norm(op.vec)
path_length(op::ArcOp) = op.radius*abs(op.amplitude)

translate(path::PathOps, v::Vec) =
  PathOps(path.start + v, path.ops, path.closed)


location_at_length(path::PathOps, d::Real) =
  let ops = path.ops,
      start = path.start
    for op in ops
      delta = path_length(op)
      if d < delta + coincidence_tolerance()
        return location_at_length(op, start, d)
      else
        start = location_at_length(op, start, delta)
        d -= delta
      end
    end
    error("Exceeded path length by ", d)
  end

location_at_length(op::LineOp, start::Loc, d::Real) =
  start + op.vec*d/path_length(op)
location_at_length(op::ArcOp, start::Loc, d::Real) =
  let center = start - vpol(op.radius, op.start_angle),
      a = d/op.radius*sign(op.amplitude)
    center + vpol(op.radius, op.start_angle + a)
  end

subpath_starting_at(path::PathOps, d::Real) =
  let ops = path.ops,
      start = location_at_length(path, d)
    for i in 1:length(ops)
      op = ops[i]
      delta = path_length(op)
      if d == delta
        return PathOps(start, ops[i+1:end], false)
      elseif d < delta + coincidence_tolerance()
        op = subpath_starting_at(op, d)
        return PathOps(start, [op, ops[i+1:end]...], false)
      else
        d -= delta
      end
    end
    error("Exceeded path length by ", d)
  end
subpath_ending_at(path::PathOps, d::Real) =
  let ops = path.ops
    for i in 1:length(ops)
      op = ops[i]
      delta = path_length(op)
      if d == delta
        return PathOps(path.start, ops[1:i], false)
      elseif d < delta + coincidence_tolerance()
        return PathOps(path.start, [ops[1:i-1]..., subpath_ending_at(op, d)], false)
      else
        d -= delta
      end
    end
    error("Exceeded path length by ", d)
  end

subpath_starting_at(pathOp::LineOp, d::Real) =
  let len = path_length(pathOp)
    LineOp(pathOp.vec*(len-d)/len)
  end
subpath_starting_at(pathOp::ArcOp, d::Real) =
  let a = d/pathOp.radius
    ArcOp(pathOp.radius, pathOp.start_angle + a, pathOp.amplitude - a)
  end

subpath_ending_at(pathOp::LineOp, d::Real) =
  let len = path_length(pathOp)
    LineOp(pathOp.vec*d/len)
  end
subpath_ending_at(pathOp::ArcOp, d::Real) =
  let a = d/pathOp.radius
    ArcOp(pathOp.radius, pathOp.start_angle, a)
  end

subpath(path::Path, a::Real, b::Real) =
  subpath_starting_at(subpath_ending_at(path, b), a)

path_domain(path::CompositePath) = (0, path_length(path))
location_at(path::CompositePath, d::Real) = location_at_length(path, d)
path_start(path::CompositePath) = path_start(path.pieces[1])
path_end(path::CompositePath{false}) = path_end(path.pieces[end])
path_end(path::CompositePath{true}) = path_start(path)

path_length(path::CompositePath) =
  sum(map(path_length, path.pieces))
location_at_length(path::CompositePath, d::Real) =
  let paths = path.pieces
    if is_closed_path(path)
      total = path_length(path)
      total > coincidence_tolerance() && (d = d % total)
    end
    for path in paths
      delta = path_length(path)
      if d <= delta + coincidence_tolerance()
          return location_at_length(path, d)
      else
          d -= delta
      end
    end
    error("Exceeded path length by ", d)
  end
subpath_starting_at(path::CompositePath, d::Real) =
  let subpaths = path.pieces
    for i in 1:length(subpaths)
      subpath = subpaths[i]
      delta = path_length(subpath)
      if d == delta
        return CompositePath{false}(subpaths[i+1:end])
      elseif d < delta + coincidence_tolerance()
        subpath = subpath_starting_at(subpath, d)
        return CompositePath{false}(OpenPath[subpath, subpaths[i+1:end]...])
      else
        d -= delta
      end
    end
    error("Exceeded path length by ", d)
  end
subpath_ending_at(path::CompositePath, d::Real) =
  let subpaths = path.pieces
    for i in 1:length(subpaths)
      subpath = subpaths[i]
      delta = path_length(subpath)
      if d == delta
        return CompositePath{false}(subpaths[1:i])
      elseif d < delta + coincidence_tolerance()
        return CompositePath{false}(OpenPath[subpaths[1:i-1]..., subpath_ending_at(subpath, d)])
      else
        d -= delta
      end
    end
    error("Exceeded path length by ", d)
  end

convert(::Type{PathOps}, path::CompositePath) =
  let start = in_world(location_at_length(path, 0)),
      ops = mapfoldl(convert_to_path_ops, vcat, path.pieces, init=PathOp[])
    PathOps(start, ops, coincident_path_location(start, path_end(path)))
  end
convert_to_path_ops(path::ArcPath) =
  [ArcOp(path.radius, path.start_angle, path.amplitude)]
convert_to_path_ops(path::PolygonalPath) =
  [LineOp(in_world(p)-in_world(q)) for (p,q) in zip(path.vertices[[2:end;1]], path.vertices)]
convert_to_path_ops(path::RectangularPath) =
  let p0 = in_world(path.corner),
      p1 = in_world(add_x(path.corner, path.dx)),
      p2 = in_world(add_xy(path.corner, path.dx, path.dy)),
      p3 = in_world(add_y(path.corner, path.dy))
    [LineOp(p1-p0), LineOp(p2-p1), LineOp(p3-p2), LineOp(p0-p3)]
  end

# A path set is a set of independent paths.
struct PathSet <: GeometryElement
  paths::Vector{<:Path}
end

# Should we just use tuples instead of arrays?
path_set(paths...) =
  PathSet([paths...])

meta_program(r::Region) =
  Expr(:call, :region, map(meta_program, r.paths)...)

export region
region(outer, inners...) =
  Region([convert(ClosedPath, outer), convert.(ClosedPath, inners)...])
outer_path(region::Region) = region.paths[1]
inner_paths(region::Region) = region.paths[2:end]

region(r::Region, inners...) =
  region(outer_path(r), inner_paths(r)..., inners...)

# Conversions from/to paths

#=
Planarity tolerance.

A panel (planar region) is defined by a sequence of vertices that should
all lie on a common plane. Even when the authoring geometry is exactly
planar, numerical operations (coordinate transforms, transports between
frames) introduce small deviations; we accept a panel as planar if every
vertex's signed distance from the reference plane is below this threshold.

Compared against `|dot(p - p0, n̂)|`, the signed distance from each vertex
p to the plane through p0 with unit normal n̂. Units are metres. The
default 1e-6 m (one micrometre) is tight enough to reject human-authored
mistakes (where deviations are typically mm or larger) while tolerating
the accumulated rounding of several transform compositions. Loosen it
when ingesting measured or imported geometry whose provenance is less
precise.

See also: zero_vector_tolerance (used to reject a panel whose first three
vertices are collinear, leaving the plane normal undefined).
=#

"Maximum signed distance a vertex may lie off the plane before a region is rejected as non-planar. `|dot(p - p0, n̂)| < planarity_tolerance()`. [metres]"
const planarity_tolerance = Parameter(1e-6)
export planarity_tolerance

function planar_region(vs::Locs)
  length(vs) < 3 && error("A panel requires at least 3 vertices.")
  let p0 = in_world(vs[1]),
      p1 = in_world(vs[2]),
      p2 = in_world(vs[3]),
      v1 = p1 - p0,
      v2 = p2 - p0,
      n = cross(v1, v2)
    # A stricter guard than zero_vector_tolerance(): we need the normal to
    # be well above float noise so that `unitized(n)` below is numerically
    # safe. 1e-10 m² is ~10 orders of magnitude above Float64 resolution
    # at metre scale.
    norm(n) < 1e-10 && error("First 3 panel vertices are collinear.")
    let n = unitized(n),
        frame = loc_from_o_vx_vy(p0, v1, v2),
        wpts = map(in_world, vs),
        max_dev = maximum(abs(dot(wp - p0, n)) for wp in wpts)
      max_dev > planarity_tolerance() &&
        error("Panel vertices are not coplanar (max deviation: $(max_dev) > $(planarity_tolerance())).")
      let ux = unitized(v1),
          uz = n,
          uy = cross(uz, ux),
          pts_2d = [xy(dot(wp - p0, ux), dot(wp - p0, uy), frame.cs) for wp in wpts]
        region(closed_polygonal_path(pts_2d))
      end
    end
  end
end

convert(::Type{Region}, vs::Locs) =
  planar_region(vs)
convert(::Type{Region}, path::ClosedPath) =
  region(path)

# Operations on path containers
translate(path::T, v::Vec) where T<:Union{PathSet,Region} =
  T(translate.(path.paths, v))
scale(path::T, s::Real, p::Loc=u0()) where T<:Union{PathSet,Region} =
  T(scale.(path.paths, s, p))

in_cs(path::T, cs::CS) where T<:Union{PathSet,Region} =
  T(map(p->in_cs(p, cs), path.paths))
planar_path_normal(path::CompositePath) =
  planar_path_normal(path.pieces[1])
planar_path_normal(path::T) where T<:Union{PathSet,Region} =
  planar_path_normal(path.paths[1])

# Convertions from/to paths
convert(::Type{OpenPath}, vs::Locs) =
  open_polygonal_path(vs)
convert(::Type{ClosedPath}, vs::Locs) =
  closed_polygonal_path(vs)
convert(::Type{Vector{<:ClosedPath}}, ps::Vector{<:Any}) =
  ClosedPath[convert(ClosedPath, p) for p in ps]

convert(::Type{Path}, vs::Locs) =
  coincident_path_location(vs[1], vs[end]) ?
    closed_polygonal_path(vs[1:end-1]) :
    open_polygonal_path(vs)
convert(::Type{ClosedPath}, p::OpenPolygonalPath) =
  closed_polygonal_path(coincident_path_location(path_start(p), path_end(p)) ?
    path_vertices(p)[1:end-1] :
    path_vertices(p))
convert(::Type{ClosedPolygonalPath}, path::RectangularPath) =
  let p = path.corner,
      dx = path.dx,
      dy = path.dy
    closed_polygonal_path([p, add_x(p, dx), add_xy(p, dx, dy), add_y(p, dy)])
  end

path_interpolated_lengths(path, t0=0, t1=path_length(path), tol=collinearity_tolerance(), min_recursion=1) =
  let p0 = location_at_length(path, t0),
      p1 = location_at_length(path, t1),
      tm = (t0 + t1)/2.0,
      pm = location_at_length(path, tm)
    min_recursion < 0 && collinear_points(p0, pm, p1, tol) ?
      [t0, tm, t1] :
      [path_interpolated_lengths(path, t0, tm, tol, min_recursion - 1)...,
       path_interpolated_lengths(path, tm, t1, tol, min_recursion - 1)[2:end]...]
  end

#### Smooth-path → polyline conversions
#
# Deterministic uniform sampling at density `path_smoothness_segments` per 2π
# of parameter span. See that parameter's prose for the rationale (paired
# smooth boundaries align by construction; no false-flat termination).

# Segment count for an arc-like path of given amplitude, scaled from the
# global smoothness budget and clamped to ≥ 2 so degenerate amplitudes
# (zero, near-zero) still produce a well-formed polyline.
arc_segment_count(amplitude) =
  max(2, ceil(Int, path_smoothness_segments() * abs(amplitude) / (2π)))

# n+1 frames sampled at uniform parameter spacing across [t0, t1].
uniform_path_frames(path, t0, t1, n) =
  [location_at(path, t) for t in range(t0, t1; length=n+1)]

convert(::Type{ClosedPolygonalPath}, path::CircularPath) =
  let n = path_smoothness_segments()
    closed_polygonal_path(uniform_path_frames(path, 0, 2π, n)[1:end-1])
  end
convert(::Type{OpenPolygonalPath}, path::CircularPath) =
  let n = path_smoothness_segments()
    open_polygonal_path(uniform_path_frames(path, 0, 2π, n))
  end
convert(::Type{ClosedPolygonalPath}, path::EllipticPath) =
  let n = path_smoothness_segments()
    closed_polygonal_path(uniform_path_frames(path, 0, 2π, n)[1:end-1])
  end
convert(::Type{OpenPolygonalPath}, path::EllipticPath) =
  let n = path_smoothness_segments()
    open_polygonal_path(uniform_path_frames(path, 0, 2π, n))
  end
convert(::Type{OpenPolygonalPath}, path::LinePath) =
  open_polygonal_path([path.p0, path.p1])
convert(::Type{OpenPolygonalPath}, path::ArcPath) =
  open_polygonal_path(uniform_path_frames(path, 0, path.amplitude,
                                          arc_segment_count(path.amplitude)))
convert(::Type{OpenPolygonalPath}, path::EllipticArcPath) =
  open_polygonal_path(uniform_path_frames(path, 0, path.amplitude,
                                          arc_segment_count(path.amplitude)))
convert(::Type{OpenPolygonalPath}, path::ClosedPolygonalPath) =
  open_polygonal_path(vcat(path.vertices, [path.vertices[1]]))
convert(::Type{OpenPolygonalPath}, path::RectangularPath) =
  convert(OpenPolygonalPath, convert(ClosedPolygonalPath, path))
# Splines: use the user's control points when they are dense enough to
# render smoothly on their own; otherwise resample uniformly. Dense user
# input is preserved exactly so paired splines (parallel boundaries of a
# thin region) flatten to mathematically aligned polylines.
convert(::Type{OpenPolygonalPath}, path::OpenSplinePath) =
  let vs = path.vertices,
      n = path_smoothness_segments()
    length(vs) >= n + 1 ?
      open_polygonal_path(vs) :
      open_polygonal_path(uniform_path_frames(path, 0, 1, n))
  end
convert(::Type{ClosedPolygonalPath}, path::ClosedSplinePath) =
  let vs = path.vertices,
      n = path_smoothness_segments()
    length(vs) >= n ?
      closed_polygonal_path(vs) :
      closed_polygonal_path(uniform_path_frames(path, 0, 1, n)[1:end-1])
  end
convert(::Type{OpenPolygonalPath}, path::ClosedSplinePath) =
  let p = convert(ClosedPolygonalPath, path)
    open_polygonal_path(vcat(p.vertices, [p.vertices[1]]))
  end
convert(::Type{OpenPolygonalPath}, path::BezierPath{false}) =
  let (t0, t1) = path_domain(path),
      n = max(path_smoothness_segments(), length(path.spans) * 8)
    open_polygonal_path(uniform_path_frames(path, t0, t1, n))
  end
convert(::Type{ClosedPolygonalPath}, path::BezierPath{true}) =
  let (t0, t1) = path_domain(path),
      n = max(path_smoothness_segments(), length(path.spans) * 8)
    closed_polygonal_path(uniform_path_frames(path, t0, t1, n)[1:end-1])
  end
convert(::Type{OpenPolygonalPath}, path::BezierPath{true}) =
  let p = convert(ClosedPolygonalPath, path)
    open_polygonal_path(vcat(p.vertices, [p.vertices[1]]))
  end
convert(::Type{OpenPolygonalPath}, path::BSplinePath{false}) =
  let (t0, t1) = path_domain(path),
      n = max(path_smoothness_segments(), length(path.control_points) * 8)
    open_polygonal_path(uniform_path_frames(path, t0, t1, n))
  end
convert(::Type{ClosedPolygonalPath}, path::BSplinePath{true}) =
  let (t0, t1) = path_domain(path),
      n = max(path_smoothness_segments(), length(path.control_points) * 8)
    closed_polygonal_path(uniform_path_frames(path, t0, t1, n)[1:end-1])
  end
convert(::Type{OpenPolygonalPath}, path::BSplinePath{true}) =
  let p = convert(ClosedPolygonalPath, path)
    open_polygonal_path(vcat(p.vertices, [p.vertices[1]]))
  end

#=
PathOps stores a start location plus a sequence of LineOp/ArcOp/etc.
Without a `convert(::Type{OpenPolygonalPath}, ::PathOps)` method, callers
that go through the generic `path_vertices(::Path)` fallback hit
MethodError — most visibly `b_extruded_curve(::Backend, ::Path, ...)`.
Sample 8 vertices per op (LineOps get a segment, ArcOps get a curved
sample) and join into one polyline. Avoids the full-length
`location_at_length` walk because that errors at the exact endpoint of
closed paths.
=#
convert(::Type{OpenPolygonalPath}, path::PathOps) =
  let vs = Loc[path.start]
      curr = path.start
      for op in path.ops
        seg_len = path_length(op)
        n_steps = op isa LineOp ? 1 : 8
        for i in 1:n_steps
          push!(vs, location_at_length(op, curr, seg_len * i / n_steps))
        end
        curr = vs[end]
      end
      path.closed && push!(vs, path.start)
      open_polygonal_path(vs)
  end


path_pieces(path::EmptyPath) = OpenPath[]
path_pieces(path::PointPath) = OpenPath[]
path_pieces(path::LinePath) = OpenPath[path]
path_pieces(path::ArcPath) = OpenPath[path]
path_pieces(path::EllipticArcPath) = OpenPath[path]
path_pieces(path::CircularPath) =
  OpenPath[arc_path(path.center, path.radius, 0, 2π)]
path_pieces(path::EllipticPath) =
  OpenPath[elliptic_arc_path(path.center, path.r1, path.r2, 0, 2π)]
path_pieces(path::RectangularPath) =
  let p0 = path.corner,
      p1 = add_x(path.corner, path.dx),
      p2 = add_xy(path.corner, path.dx, path.dy),
      p3 = add_y(path.corner, path.dy)
    OpenPath[line_path(p0, p1), line_path(p1, p2),
             line_path(p2, p3), line_path(p3, p0)]
  end
path_pieces(path::OpenPolygonalPath) =
  OpenPath[line_path(p0, p1) for (p0, p1) in zip(path.vertices[1:end-1], path.vertices[2:end])]
path_pieces(path::ClosedPolygonalPath) =
  let vs = path.vertices
    OpenPath[line_path(vs[i], vs[mod1(i + 1, length(vs))]) for i in eachindex(vs)]
  end
path_pieces(path::OpenSplinePath) =
  OpenPath[path]
path_pieces(path::ClosedSplinePath) =
  OpenPath[open_spline_path([path.vertices..., path.vertices[1]])]
path_pieces(path::BezierPath{false}) =
  OpenPath[open_bezier_path(span) for span in path.spans]
path_pieces(path::BezierPath{true}) =
  OpenPath[open_bezier_path(span) for span in path.spans]
path_pieces(path::BSplinePath{false}) =
  OpenPath[path]
path_pieces(path::BSplinePath{true}) =
  OpenPath[convert(OpenPolygonalPath, path)]
path_pieces(path::CompositePath) =
  mapreduce(path_pieces, vcat, path.pieces; init=OpenPath[])
path_pieces(path::PathOps) =
  let curr = path.start,
      pieces = OpenPath[]
    for op in path.ops
      piece = pathop_piece(op, curr)
      if !isnothing(piece)
        push!(pieces, piece)
        curr = path_end(piece)
      end
    end
    if path.closed && !coincident_path_location(curr, path.start)
      push!(pieces, line_path(curr, path.start))
    end
    pieces
  end
path_pieces(path::Path) =
  throw(ArgumentError("$(typeof(path)) is not a connected contour path."))

single_arc_path(path::Path) =
  let pieces = path_pieces(path)
    length(pieces) == 1 && pieces[1] isa ArcPath ? pieces[1] : nothing
  end

single_arc_segment(path::Path) = single_arc_path(path)

pathop_piece(op::LineOp, start::Loc) =
  line_path(start, start + op.vec)
pathop_piece(op::ArcOp, start::Loc) =
  let center = start - vpol(op.radius, op.start_angle)
    arc_path(center, op.radius, op.start_angle, op.amplitude)
  end
pathop_piece(::CloseOp, start::Loc) = nothing

function _path_ops_composite_path(start::Loc, ops::Vector, closed::Bool)
  pieces = OpenPath[]
  curr = start
  for op in ops
    if op isa CloseOp
      if !coincident_path_location(curr, start)
        push!(pieces, line_path(curr, start))
        curr = start
      end
      continue
    end
    piece = pathop_piece(op, curr)
    if !isnothing(piece)
      push!(pieces, piece)
      curr = path_end(piece)
    end
  end
  if closed && !coincident_path_location(curr, start)
    push!(pieces, line_path(curr, start))
  end
  composite_path(pieces; closed=closed)
end

path_control_points(path::EmptyPath) = Loc[]
path_control_points(path::PointPath) = Loc[path.location]
path_control_points(path::LinePath) = Loc[path.p0, path.p1]
path_control_points(path::CircularPath) = Loc[path.center]
path_control_points(path::ArcPath) = Loc[path.center, path_start(path), path_end(path)]
path_control_points(path::EllipticPath) = Loc[path.center]
path_control_points(path::EllipticArcPath) = Loc[path.center, path_start(path), path_end(path)]
path_control_points(path::RectangularPath) = path_vertices(path)
path_control_points(path::Union{PolygonalPath,InterpolatingSplinePath}) = path.vertices
path_control_points(path::BezierPath) =
  mapreduce(seg -> seg.control_points, vcat, path.spans; init=Loc[])
path_control_points(path::BSplinePath) = path.control_points
path_control_points(path::CompositePath) =
  mapreduce(path_control_points, vcat, path.pieces; init=Loc[])
path_control_points(path::PathOps) =
  path_vertices(convert(OpenPolygonalPath, path))
path_control_points(path::Path) =
  throw(ArgumentError("$(typeof(path)) has no authored control-point representation."))

path_breakpoints(path::Path) =
  let pieces = path_pieces(path)
    isempty(pieces) && return path_control_points(path)
    pts = Loc[path_start(pieces[1])]
    append!(pts, path_end.(pieces))
    is_closed_path(path) && !isempty(pts) && pop!(pts)
    pts
  end

function path_points(path::Path; mode::Symbol=:sample, resolution=nothing)
  resolution === nothing ||
    throw(ArgumentError("Per-call path point resolution is not implemented yet; use path_smoothness_segments."))
  mode === :sample && return path_vertices(path)
  mode === :control && return path_control_points(path)
  mode === :breakpoints && return path_breakpoints(path)
  throw(ArgumentError("Unknown path point mode $(mode). Use :sample, :control, or :breakpoints."))
end


curve_interpolator(pts::Locs, closed::Bool) =
  let pts = length(pts) > 2 ? pts : [pts[1], intermediate_loc(pts...), pts[2]]
    ParametricSpline(
      [pt.raw[i] for i in 1:3, pt in in_world.(closed ? [pts..., pts[1]] : pts)],
      k=min(length(pts)-1, 3),
      periodic=closed)
  end
curve_control_points(interpolator) =
  let cps = get_coeffs(interpolator)
    [xyz(cps[1,j], cps[2,j], cps[3,j], world_cs)
     for j in 1:size(cps, 2)]
  end
curve_knots(interpolator) =
  get_knots(interpolator)

convert(::Type{ClosedPolygonalPath}, path::CompositePath{true}) =
  let paths = path.pieces,
      vertices = Loc[]
    for path in paths
      append!(vertices, convert(OpenPolygonalPath, path).vertices[1:end-1])
    end
    #= A composite whose every piece is degenerate (zero-length) contributes no
       vertices, so `vertices` ends up empty. Constructing closed_polygonal_path([])
       would BoundsError in ensure_no_repeated_locations (it reads locs[1]/locs[end]).
       Match the constructor style of closed_path_sequence and refuse explicitly. =#
    isempty(vertices) && throw(ArgumentError("Cannot convert a degenerate composite path (no vertices) to a ClosedPolygonalPath."))
    closed_polygonal_path(vertices)
  end

convert(::Type{OpenPolygonalPath}, path::CompositePath{false}) =
  let paths = path.pieces,
      vertices = Loc[]
    for path in paths[1:end-1]
      append!(vertices, convert(OpenPolygonalPath, path).vertices[1:end-1])
    end
    append!(vertices, convert(OpenPolygonalPath, paths[end]).vertices)
    #= If every piece is degenerate the accumulator stays empty; open_polygonal_path([])
       yields an OpenPolygonalPath with no vertices that BoundsErrors downstream
       (path_start/.vertices[1]). Refuse explicitly, mirroring the closed variant. =#
    isempty(vertices) && throw(ArgumentError("Cannot convert a degenerate composite path (no vertices) to an OpenPolygonalPath."))
    open_polygonal_path(vertices)
  end
convert(::Type{OpenPolygonalPath}, path::CompositePath{true}) =
  convert(OpenPolygonalPath, convert(ClosedPolygonalPath, path))


convert(::Type{ClosedPath}, paths::Vector{<:Path}) =
  closed_path_sequence(paths...)

# It is possible to convert a PathSet to a singleton path
# by considering the first path as the outer path and all the
# others as inner paths
#=
The foldl below needs at least one path and only yields a ClosedPath when
`subtract_paths` actually runs (2+ paths). Two degenerate inputs reach
here — both produced by `_analytic_trim`/`path_set()` (GeometryOperations)
and by `flatten_paths` over a single-element CSG result (Shapes):

- An empty PathSet would make `foldl` reduce over an empty collection
  ("reduce over empty collection"), since there is no `init`. There is no
  outer boundary to build a closed path from, so we error — matching
  `closed_path_sequence`, the analogous closed-path-from-list constructor.
- A singleton PathSet would make `foldl` return the held path verbatim,
  bypassing `subtract_paths`; that path can be open (e.g. a LinePath from
  a trim), violating the declared ClosedPath return type. We convert it
  explicitly so the result is always a ClosedPath.
=#
convert(::Type{ClosedPath}, pset::PathSet) =
  isempty(pset.paths) ?
    throw(ArgumentError("Cannot convert an empty PathSet to a ClosedPath: there is no outer boundary.")) :
    length(pset.paths) == 1 ?
      convert(ClosedPath, pset.paths[1]) :
      foldl(subtract_paths, pset.paths)

#### Utilities
path_vertices(path::Union{PolygonalPath,InterpolatingSplinePath}) =
  path.vertices
path_vertices(path::Path) =
  path_vertices(convert(is_closed_path(path) ? ClosedPolygonalPath : OpenPolygonalPath,
                        path))

iterate(path::PolygonalPath, i=1) =
  i > length(path.vertices) ?
    nothing :
    (path.vertices[i], i+1)

loc_from_o_vx_vz(o, vx, vz) =
  norm(vx) < zero_vector_tolerance() ?
    loc_from_o_vz(o, vz) :
    let vy = -cross(vx, vz)
      loc_from_o_vx_vy(o, cross(vy, vz), vy)
    end

loc_from_o_px_pz(o, px, pz) =
  loc_from_o_vx_vz(o, px - o, pz - o)

path_frames(path::Path) =
  path_interpolated_frames(path)

path_frames(path::RectangularPath) =
  path_frames(convert(ClosedPolygonalPath, path))

average_frame(f1, f2) =
  let vx1 = in_world(vx(1, f1.cs)),
      vy1 = in_world(vy(1, f1.cs)),
      vx2 = in_world(vx(1, f2.cs)),
      vy2 = in_world(vy(1, f2.cs))
    loc_from_o_vx_vy(intermediate_loc(f1, f2), (vx1 + vx2)/2, (vy1 + vy2)/2)
  end

path_frames(path::OpenPolygonalPath) =
  let pts = path.vertices,
      pt1 = pts[1],
      pt2 = pts[2],
      ptn1 = pts[end-1],
      ptn = pts[end]
    length(pts) < 3 ?
      [loc_from_o_vz(pt1, pt1-pt2), loc_from_o_vz(ptn, pt1-pt2)] :
      let normal(pprev, p, pnext) = (unitized(pnext-p) - unitized(p-pprev))
        [loc_from_o_vx_vz(pt1, normal(pt1, pt2, pts[3]), pt2-pt1),
         [loc_from_o_vx_vz(p, normal(pprev, p, pnext), ((p-pprev) + (pnext-p))/2)
          for (pprev, p, pnext) in zip(pts, drop(pts, 1), drop(pts, 2))]...,
         loc_from_o_vx_vz(ptn, normal(pts[end-2], ptn1, ptn), ptn - ptn1)]
      end
  end

path_frames(path::ClosedPolygonalPath) =
  let pts = path.vertices
    path_frames(OpenPolygonalPath([pts[end], pts..., pts[1]]))[begin+1:end-1]
  end

subpaths(path::OpenPolygonalPath) =
  let ps = path.vertices
    map((p0, p1)->open_polygonal_path([p0, p1]), ps[1:end-1], ps[2:end])
  end
subpaths(path::ClosedPolygonalPath) =
  let ps = path.vertices
      map((p0, p1)->open_polygonal_path([p0, p1]), ps, [ps[2:end]..., ps[1]])
  end
subpaths(path::Path) = path_pieces(path)

subtract_paths(path1::Path, path2::Path) =
  subtract_paths(convert(ClosedPolygonalPath, path1), convert(ClosedPolygonalPath, path2))
subtract_paths(path1::ClosedPolygonalPath, path2::ClosedPolygonalPath) =
  closed_polygonal_path(
    subtract_polygon_vertices(
      path_vertices(path1),
      path_vertices(path2)))

closed_offsetted_path(path, v) =
  let ps = path_vertices(path)
    closed_polygonal_path([ps..., reverse(map(p -> p+v, ps))...])
  end

closed_path_for_height(path, h) =
  closed_offsetted_path(path, vz(h))

## Smoothnesss
# A smooth curve means that the curve is differentiable up to an intended order
# In practice, it means that it does not have "corners"
is_smooth_path(path::Path) = false
is_smooth_path(pts::Locs) = false
is_smooth_path(path::Union{ArcPath,CircularPath,SplinePath}) = true
is_smooth_path(path::Union{EllipticPath,EllipticArcPath,BezierPath,BSplinePath}) = true
is_smooth_path(path::CompositePath) =
  length(path.pieces) == 1 && is_smooth_path(path.pieces[1])

##Profiles are just predefined paths used for sections, usually centered at the origin
export rectangular_profile,
       circular_profile,
       top_aligned_rectangular_profile,
       bottom_aligned_rectangular_profile,
       i_profile, plus_profile

rectangular_profile(Width::Real=0.1, Height::Real=0.1; width::Real=Width, height::Real=Height) =
  centered_rectangular_path(u0(), width, height)

circular_profile(Radius::Real=0.1; radius::Real=Radius) =
  circular_path(u0(), radius)

top_aligned_rectangular_profile(Width::Real=0.1, Height::Real=0.1; width::Real=Width, height::Real=Height) =
  rectangular_path(xy(-width/2,-height), width, height)

bottom_aligned_rectangular_profile(Width::Real=0.1, Height::Real=0.1; width::Real=Width, height::Real=Height) =
  rectangular_path(xy(-width/2, 0), width, height)

i_profile(Width::Real=0.1, Height::Real=Width; width::Real=Width, height::Real=Height, web_thickness::Real=width/5, flange_thickness::Real=width/10) =
    let w2 = width/2,
        h2 = height/2,
        wt2 = web_thickness/2,
        ft = flange_thickness
      mirrored_on_x(
        mirrored_on_y(
          open_polygonal_path([
            x(wt2), xy(wt2, h2-ft), xy(w2, h2-ft), xy(w2, h2), y(h2)])))
    end

plus_profile(Width::Real=0.1, Height::Real=Width; width::Real=Width, height::Real=Height, thickness::Real=width/5) =
    let w2 = width/2,
        h2 = height/2,
        t2 = thickness/2
      mirrored_on_x(
        mirrored_on_y(
          open_polygonal_path([
            x(w2), xy(w2, t2), xy(t2, t2), xy(t2, h2), y(h2)])))
    end

export path_vertices_on
path_vertices_on(path::Path, p) =
  on_cs(path_vertices(path), p)

export path_on
path_on(path::PointPath, p) =
  point_path(on_cs(path.location, p))
path_on(path::LinePath, p) =
  line_path(on_cs(path.p0, p), on_cs(path.p1, p))
path_on(path::ArcPath, p) =
  arc_path(on_cs(path.center, p), path.radius, path.start_angle, path.amplitude)
path_on(path::CircularPath, p) =
  circular_path(on_cs(path.center, p), path.radius)
path_on(path::EllipticPath, p) =
  let c = path.center
    elliptic_path(
      loc_from_o_vx_vy(on_cs(c, p), uvx(c.cs), uvy(c.cs)),
      path.r1, path.r2)
  end
path_on(path::EllipticArcPath, p) =
  let c = path.center
    elliptic_arc_path(
      loc_from_o_vx_vy(on_cs(c, p), uvx(c.cs), uvy(c.cs)),
      path.r1, path.r2, path.start_angle, path.amplitude)
  end
path_on(path::RectangularPath, p) =
  rectangular_path(on_cs(path.corner, p), path.dx, path.dy)
path_on(path::OpenPolygonalPath, p) =
  open_polygonal_path(on_cs(path_vertices(path), p))
path_on(path::ClosedPolygonalPath, p) =
  closed_polygonal_path(on_cs(path_vertices(path), p))
path_on(path::OpenSplinePath, p) =
  open_spline_path(on_cs(path_vertices(path), p))
path_on(path::ClosedSplinePath, p) =
  closed_spline_path(on_cs(path_vertices(path), p))
path_on(path::BezierPath{false}, p) =
  open_bezier_path([_span_on(seg, p) for seg in path.spans])
path_on(path::BezierPath{true}, p) =
  closed_bezier_path([_span_on(seg, p) for seg in path.spans])
path_on(path::BSplinePath{Closed,false}, p) where {Closed} =
  bspline_path(on_cs(path.control_points, p); degree=path.degree,
               periodic=is_closed_path(path), knots=path.knots,
               domain=path.domain)
path_on(path::BSplinePath{Closed,true}, p) where {Closed} =
  nurbs_path(on_cs(path.control_points, p); degree=path.degree,
             periodic=is_closed_path(path), knots=path.knots,
             weights=path.weights, domain=path.domain)
_span_on(seg::BezierSpan, p) =
  BezierSpan(on_cs(seg.control_points, p))
path_on(path::CompositePath{Closed}, p) where {Closed} =
  CompositePath{Closed}(OpenPath[path_on(piece, p) for piece in path.pieces])
path_on(path::Region, p) =
  region([path_on(path, p) for path in path.paths]...)

## Utility operations
Base.reverse(path::CircularPath) =
  circular_path(loc_from_o_vz(path.center, vz(-1, path.center.cs)), path.radius)
Base.reverse(path::LinePath) =
  line_path(path.p1, path.p0)
#=
Reversing an arc must trace the same points in the opposite order. The
analogous trick used for `CircularPath` above — flip the center CS's Z
so that positive sweep maps to the original negative sweep — relies on
`cs_from_o_vz` happening to pick an X axis compatible with the fixed
angle shift. That compatibility holds for a full circle (no start
matters) but not for an arc: `cs_from_o_vz((0,0,-1))` introduces a 90°
rotation of X' that misaligns `start_angle + amplitude` with the true
endpoint direction. So we build the reversed center explicitly: X' at
the endpoint direction, Y' 90° clockwise of X' (which makes Z'
automatically −Z, right-handed). Then the reversed arc sweeps from 0
to |amplitude| in that CS and traces `angle = θ − ϕ` in the original.

Note: `abs(amplitude)` is used because `location_at` applies
`sign(amplitude)` twice (in `location_at_length` and again inside
`location_at`), so the sign cancels and the magnitude is what governs
the traced sweep.
=#
"""Reverse of an arc, tracing the same points in opposite order."""
Base.reverse(path::ArcPath) =
  let θ = path.start_angle + path.amplitude
    arc_path(loc_from_o_vx_vy(path.center,
                              vpol(1, θ,       path.center.cs),
                              vpol(1, θ - π/2, path.center.cs)),
             path.radius, 0, abs(path.amplitude))
  end
Base.reverse(path::EllipticArcPath) =
  elliptic_arc_path(path.center, path.r1, path.r2,
                    path.start_angle + path.amplitude, -path.amplitude)
Base.reverse(path::RectangularPath) =
  rectangular_path(loc_from_o_vz(path.corner, vz(-1, path.corner.cs)), -path.dx, path.dy)
Base.reverse(path::OpenPolygonalPath) =
  open_polygonal_path(reverse(path_vertices(path)))
Base.reverse(path::ClosedPolygonalPath) =
  closed_polygonal_path(reverse(path_vertices(path)))
Base.reverse(path::BezierPath{false}) =
  open_bezier_path([BezierSpan(reverse(span.control_points)) for span in reverse(path.spans)])
Base.reverse(path::BezierPath{true}) =
  closed_bezier_path([BezierSpan(reverse(span.control_points)) for span in reverse(path.spans)])
Base.reverse(path::CompositePath{Closed}) where {Closed} =
  CompositePath{Closed}(OpenPath[reverse(piece) for piece in reverse(path.pieces)])
Base.reverse(path::Region) =
  region(reverse.(path.paths)...)


mirrored_path(path::Path, p::Loc, v::Vec) =
  error("mirrored_path is not yet implemented for $(typeof(path))")

mirrored_on_x(path::OpenPolygonalPath) =
  join_paths(path, open_polygonal_path(reverse(map(p->xyz(p.x, -p.y, -p.z, p.cs), path_vertices(path)))))
mirrored_on_y(path::OpenPolygonalPath) =
  join_paths(path, open_polygonal_path(reverse(map(p->xyz(-p.x, p.y, -p.z, p.cs), path_vertices(path)))))
mirrored_on_z(path::OpenPolygonalPath) =
  join_paths(path, open_polygonal_path(reverse(map(p->xyz(-p.x, -p.y, p.z, p.cs), path_vertices(path)))))

export mesh
struct Mesh <: SurfaceGeometry
  vertices::Locs
  faces::Vector{Vector{Int}}
end
mesh(vertices::Locs=[u0(), ux(), uy()], faces::Vector{Vector{Int}}=[[0,1,2]]) =
  Mesh(vertices, faces)

union(m::Mesh, ms...) =
  length(ms) == 0 ?
    m :
    let n = length(m.vertices)
      union(
        mesh(vcat(m.vertices, ms[1].vertices),
            vcat(m.faces, [face.+n for face in ms[1].faces])),
        ms[2:end]...)
    end

# The inverse of location_at_length
export length_at_location
length_at_location(path::Path, p::Loc, t0=0, t1=path_length(path)) =
  t1 - t0 < coincidence_tolerance() ?
    (t0+t1)/2 :
    let ts = division(t0, t1, 10),
        pts = map(t->location_at_length(path, t), ts),
        idx = argmin(map(p1 -> distance(p, p1), pts))
      length_at_location(path, p, ts[max(1, idx-1)], ts[min(length(ts), idx+1)])
    end

export Grid, grid, grid_interpolator, location_at

grid_interpolator(ptss) =
  let (nu, nv) = size(ptss).-1,
      us = division(0, 1, nu),
      vs = division(0, 1, nv),
      rawptss = map(p->in_world(p).raw, ptss),
      order(n) = min(n, 5),
      ou = order(nu),
      ov = order(nv),
      xinterpolator = Spline2D(us, vs, map(p->p[1], rawptss), kx=ou, ky=ov),
      yinterpolator = Spline2D(us, vs, map(p->p[2], rawptss), kx=ou, ky=ov),
      zinterpolator = Spline2D(us, vs, map(p->p[3], rawptss), kx=ou, ky=ov)
    (xinterpolator, yinterpolator, zinterpolator)
  end
smooth_grid(ptss, level_u=2, level_v=2) =
  let (nu, nv) = size(ptss)
    if level_u > 1 && level_v > 1
	    let interpolator = grid_interpolator(ptss)
        [location_at(interpolator, u, v)
	       for u in division(0, 1, 4*nu-7), v in division(0, 1, 4*nv-7)] # 2->1, 3->5, 4->9, 5->13
      end
    elseif level_u > 1
      let interpolators = map(col->curve_interpolator(collect(col), false), eachcol(ptss))
        [location_at(interpolator, u) for u in division(0, 1, 4*nu-7), interpolator in interpolators]
      end
    elseif level_v > 1
      let interpolators = map(row->curve_interpolator(collect(row), false), eachrow(ptss))
        [location_at(interpolator, v) for interpolator in interpolators, v in division(0, 1, 4*nv-7)]
      end
    else
      ptss
    end
  end


struct Grid
  vertices::Array{Loc,2}
  interpolator::LazyParameter{Any}
end
grid(ptss) = Grid(ptss, LazyParameter(()->grid_interpolator(ptss)))

location_at(interpolator::Tuple{Spline2D,Spline2D,Spline2D}, u, v) =
  let o = xyz(evaluate.(interpolator, u, v)..., world_cs),
      vx = unitized(vxyz(evaluate_derivative.(interpolator, u, v, 1, 0)..., world_cs)),
      vy = unitized(vxyz(evaluate_derivative.(interpolator, u, v, 0, 1)..., world_cs))
    loc_from_o_vx_vy(o, vx, vy)
  end
location_at(grid::Grid, u, v) =
  location_at(grid.interpolator(), u, v)

# We need to support convertion to OBJ format
export write_obj
write_obj(io::IO, ptss::AbstractMatrix{<:Loc}, closed_u, closed_v, smooth_u, smooth_v, interpolator) =
  let (nu, nv) = size(ptss)
    if smooth_u || smooth_v
      ptss = [location_at(interpolator, u, v)
              for u in division(0, 1, nu*4), v in division(0, 1, nv*4)]
      (nu, nv) = size(ptss)
    end
    vertex(p) =
      println(io,"v  ",p.x," ",p.y," ",p.z)
    normal(n) =
      println(io,"vn ",n.x," ",n.y," ",n.z)
    face(p0, p1, p2, p3) = # OBJ format uses 1-based indexes
      let p0 = p0+1, p1=p1+1, p2=p2+1, p3=p3+1
        smooth_u || smooth_v ?
          println(io,"f ",p0,"//",p0," ",p1,"//",p1," ",p2,"//",p2," ",p3,"//",p3) :
          println(io,"f ",p0," ",p1," ",p2," ",p3)
      end
    quad_strip_faces(s, n) =
      foreach(face, zip(s:s+n-1, s+1:s+n, s+n+1:s+2*n-1, s+n:s+2*n-2))
    quad_strip_closed_faces(s, n) =
      begin
        quad_strip_faces(s, n)
        face(s+n-1, s, s+n, s+2*n-1)
      end
    # Vertices
    for j in 1:nv
      for i in 1:nu
        vertex(in_world(ptss[i,j]))
      end
    end
    # Normals
    if smooth_u || smooth_v
      for j in 1:nv
        for i in 1:nu
          normal(in_world(uz(ptss[i,j].cs)))
        end
      end
    end
    # Faces
    if closed_u
      for i in 0:nv-2
        quad_strip_closed_faces(i*nu, nu)
      end
      if closed_v
        foreach((p, q)->face(p, p+1, q+1, q), zip((nv-1)*nu:nv*nu-2, 0:nu-2))
        face(nv*nu-1, (nv-1)*nu, 0, nu-1)
      end
    else
      for i in 0:nv-2
        quad_strip_faces(i*nu, nu)
        if closed_v
          foreach((p, q)->face(p, p+1, q+1, q), zip((nv-1)*nu:nv*nu-1, 0:nu-1))
        end
      end
    end
  end
