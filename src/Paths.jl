export empty_path,
       point_path,
       #open_path,
       #closed_path,
       open_path_ops,
       closed_path_ops,
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
       OpenSplinePath,
       open_spline_path,
       ClosedSplinePath,
       closed_spline_path,
       SplinePath,
       spline_path,
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
       path_tolerance,
       is_closed_path,
       coincident_path_location

path_tolerance = Parameter(1e-10)
coincident_path_location(p1::Loc, p2::Loc) = distance(p1, p2) < path_tolerance()



struct EmptyPath <: Path
end

empty_path() = EmptyPath()
is_empty_path(path::Path) = false
is_empty_path(path::EmptyPath) = true

struct PointPath <: Path
    location::Loc
end

point_path(loc::Loc) = PointPath(loc)

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
is_closed_path(pts::Locs) = coincident_path_location(pts[1], pts[end])

# Default path domain, probably only makes sense for Splines.
path_domain(path::Path) = (0, 1)

# Generic map_division using path default domain divided into equal parts
map_division(f::Function, path::Path, n::Integer) =
  map_division(ϕ->f(location_at(path, ϕ)), path_domain(path)..., n, is_open_path(path))
# Even more generic map_division, with default domain and divided "naturaly"
map_division(f::Function, path::Path) =
  f.(path_interpolated_frames(path))

path_interpolated_frames(path::Path, (t0, t1)=path_domain(path), epsilon=collinearity_tolerance(), min_recursion=1) =
  let p0 = location_at(path, t0),
      p1 = location_at(path, t1),
      tm = (t0 + t1)/2.0,
      pm = location_at(path, tm)
    min_recursion < 0 &&
    collinear_points(p0, pm, p1, epsilon) &&
    let tr = t0 + (t1 - t0)/(1.99+rand()*0.02),
        pr = location_at(path, tr) # To avoid coincidences
      collinear_points(p0, pr, p1, epsilon)
    end ?
      [p0, pm, p1] :
      [path_interpolated_frames(path, (t0, tm), epsilon, min_recursion - 1)...,
       path_interpolated_frames(path, (tm, t1), epsilon, min_recursion - 1)[2:end]...]
  end

## Arc path
struct ArcPath <: OpenPath
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
struct CircularPath <: ClosedPath
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
struct EllipticPath <: ClosedPath
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

## Rectangular path
struct RectangularPath <: ClosedPath
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

struct OpenPolygonalPath <: OpenPath
    vertices::Locs
end
open_polygonal_path(vertices=[u0(), x(), xy(), y()]) =
  OpenPolygonalPath(vertices)

struct ClosedPolygonalPath <: ClosedPath
    vertices::Locs
end
closed_polygonal_path(vertices=[u0(), x(), xy(), y()]) =
  ClosedPolygonalPath(ensure_no_repeated_locations(vertices))

PolygonalPath = Union{OpenPolygonalPath, ClosedPolygonalPath}

polygonal_path(vertices::Locs=[u0(), x(), xy(), y(), u0()]) =
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

join_paths(p1::OpenPolygonalPath, p2::OpenPolygonalPath) =
    coincident_path_location(p1.vertices[end], p2.vertices[1]) ?
      collinear_points(p1.vertices[end-1], p1.vertices[end], p2.vertices[2]) ?
        polygonal_path([p1.vertices[1:end-1]..., p2.vertices[2:end]...]) :
        polygonal_path([p1.vertices[1:end-1]..., p2.vertices...]) :
      polygonal_path([p1.vertices..., p2.vertices...])

# Splines

struct OpenSplinePath <: OpenPath
    vertices::Locs
    v0::Union{Bool,Vec}
    v1::Union{Bool,Vec}
    interpolator
    OpenSplinePath(vs, v0, v1) = new(vs, v0, v1, curve_interpolator(vs, false))
end
open_spline_path(vertices=[u0(), x(), xy(), y()], v0=false, v1=false) =
  OpenSplinePath(vertices, v0, v1)

struct ClosedSplinePath <: ClosedPath
    vertices::Locs
    interpolator
    ClosedSplinePath(vs) = new(vs, curve_interpolator(vs, true))
end
closed_spline_path(vertices=[u0(), x(), xy(), y()]) =
  let vertices = ensure_no_repeated_locations(vertices)
    ClosedSplinePath(vertices)
  end

spline_path(vertices::Locs) =
  coincident_path_location(vertices[1], vertices[end]) ?
    closed_spline_path(vertices[1:end-1]) :
    open_spline_path(vertices)
spline_path(v::Loc, vs...) = spline_path([v, vs...])


location_at(path::Union{OpenSplinePath,ClosedSplinePath}, ϕ::Real) =
  location_at(path.interpolator, ϕ)
location_at(interpolator::ParametricSpline, ϕ) =
  let f = interpolator(ϕ),
      d1 = derivative(interpolator, ϕ), #Interpolations.gradient(interpolator, ϕ)[1],
      d2 = derivative(interpolator, ϕ, nu=2),#u0 = path.vertices[1],
      p = xyz(f[1], f[2], f[3], world_cs),
      t = vxyz(d1[1], d1[2], d1[3], world_cs),
      n = vxyz(d2[1], d2[2], d2[3], world_cs)#,
    norm(n) <= 1e-9 ? loc_from_o_vz(p, t) : loc_from_o_vx_vy(p, n, cross(t, n))
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
            norm(n) <= 1e-9 ? loc_from_o_vz(p, t) : loc_from_o_vx_vy(p, n, cross(t, n)),
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

SplinePath = Union{OpenSplinePath, ClosedSplinePath}

path_start(path::SplinePath) = path.vertices[1]
path_end(path::OpenSplinePath) = path.vertices[end]
path_end(path::ClosedSplinePath) = path.vertices[1]

convert(::Type{ClosedSplinePath}, path::Path) =
  closed_spline_path(path_vertices(path))

#=
      fixed_normal(vn, vt) = norm(vn) < path_tolerance() ? SVector{3}(vpol(1, sph_phi(xyz(vt[1],vt[2],vt[3], world_cs))+pi/2).raw[1:3]) : vn
    map_division(
      t-> let p = interpolator(t),
              vt = Interpolations.gradient(interpolator, t)[1],
              vn = Interpolations.hessian(interpolator, t)[1],
              vy = cross(vt, vn)
            f(loc_from_o_vx_vy(
                xyz(p[1], p[2], p[3], world_cs),
                vxyz(vn[1], vn[2], vn[3], world_cs),
                vxyz(vy[1], vy[2], vy[3], world_cs)))
          end,
     0.0, 1.0, n)
 end
=#

# There is a set of operations over Paths:
# 1. translate a path a given vector
# 2. stroke a path
# 3. fill a (presumably closed) path
# 4. compute a path location given a length from the path beginning
# 5. compute a sub path from a path, a length from the path begining and a length increment
# 6. Produce the meta representation of a path

translate(path::PointPath, v::Vec) = PointPath(path.location + v)
translate(path::CircularPath, v::Vec) = CircularPath(path.center + v, path.radius)
translate(path::EllipticPath, v::Vec) = EllipticPath(path.center + v, path.r1, path.r2)
translate(path::ArcPath, v::Vec) = ArcPath(path.center + v, path.radius, path.start_angle, path.amplitude)
translate(path::RectangularPath, v::Vec) = RectangularPath(path.corner + v, path.dx, path.dy)
translate(path::OpenPolygonalPath, v::Vec) = OpenPolygonalPath(translate(path.vertices, v))
translate(path::ClosedPolygonalPath, v::Vec) = ClosedPolygonalPath(translate(path.vertices, v))
translate(path::OpenSplinePath, v::Vec) = OpenSplinePath(translate(path.vertices, v), path.v0, path.v1)
translate(path::ClosedSplinePath, v::Vec) = ClosedSplinePath(translate(path.vertices, v))
translate(ps::Locs, v::Vec) = map(p->p+v, ps)

in_cs(path::PointPath, cs::CS) = PointPath(in_cs(path.location, cs))
in_cs(path::CircularPath, cs::CS) = CircularPath(in_cs(path.center, cs), path.radius)
in_cs(path::ArcPath, cs::CS) = ArcPath(in_cs(path.center, cs), path.radius, path.start_angle, path.amplitude)
in_cs(path::RectangularPath, cs::CS) = RectangularPath(in_cs(path.corner, cs), path.dx, path.dy)
in_cs(path::OpenPolygonalPath, cs::CS) = OpenPolygonalPath(in_cs(path.vertices, cs))
in_cs(path::ClosedPolygonalPath, cs::CS) = ClosedPolygonalPath(in_cs(path.vertices, cs))
in_cs(path::OpenSplinePath, cs::CS) = OpenSplinePath(in_cs(path.vertices, cs), path.v0, path.v1)
in_cs(path::ClosedSplinePath, cs::CS) = ClosedSplinePath(in_cs(path.vertices, cs))
in_cs(ps::Locs, cs::CS) = map(p->in_cs(p, cs), ps)

scale(path::CircularPath, s::Real, p::Loc=u0()) =
  s == 1 ?
    path :
    let v = path.center - p
      circular_path(p + v*s, path.radius*s)
    end
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

rotate(path::Path, Δα, rot_p=u0(), rot_v=vz()) =
  Δα == 0 ?
    path :
    let rotate_vertex(p) = let v = p - rot_p
          rot_p + vcyl(pol_rho(v), pol_phi(v) + Δα, cyl_z(v))
        end
      polygonal_path([rotate_vertex(p) for p in path_vertices(path)])
    end



#
transform(s, p::Loc) = in_cs(s, p)

export planar_polygonal_path
planar_polygonal_path(path) =
  let normal = planar_path_normal(path)
    typeof(path)(in_cs(path.vertices, cs_from_o_vz(path.vertices[1], normal)))
  end

#
path_length(path::CircularPath) = 2*pi*path.radius
path_length(path::ArcPath) = path.radius*abs(path.amplitude)
path_length(path::RectangularPath) = 2*(path.dx + path.dy)
path_length(path::OpenPolygonalPath) = path_length(path.vertices)
path_length(path::ClosedPolygonalPath) = path_length(path.vertices) + distance(path.vertices[end], path.vertices[1])
path_length(ps::Locs) =
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
location_at_length(path::ArcPath, d::Real) =
  let Δα = d/path.radius,
      s = sign(path.amplitude)
    Δα <= abs(path.amplitude) + path_tolerance() ?
      location_at(path, Δα*s) :
      error("Exceeded path length by ", Δα - path.amplitude)
  end
location_at_length(path::RectangularPath, d::Real) =
    let d = d % (2*(path.dx + path.dy)) # remove multiple periods
        p = path.corner
        for (delta, phi) in zip([path.dx, path.dy, path.dx, path.dy], [0, pi/2, pi, 3pi/2])
            if d - delta < path_tolerance()
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
      if d - delta < path_tolerance()
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
      if d - delta < path_tolerance()
        v = unitized(pp - p)
        return loc_from_o_vz(p+v*d, v)
      else
        p = pp
        d -= delta
      end
    end
  end

subpath_starting_at(path::Path, d::Real) = subpath(path, d, path_length(path))
subpath_ending_at(path::Path, d::Real) = subpath(path, 0, d)

subpath(path::CircularPath, a::Real, b::Real) =
  arc_path(path.center, path.radius, a/path.radius, (b-a)/path.radius)
subpath(path::ArcPath, a::Real, b::Real) =
  b <= path_length(path) + path_tolerance() ?
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
            elseif abs(diff) < 1e-9
                return open_polygonal_path(i == length(pts) ? [pts[i], pts[i]] : pts[i:end])
            else
                p1 = p2
                d -= delta
            end
        end
        abs(d) < path_tolerance() ?
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
            elseif abs(diff) < path_tolerance()
                return open_polygonal_path(pts[1:i])
            else
              p1 = p2
              d -= delta
            end
        end
        abs(d) < path_tolerance() ?
          path :
          error("Exceeded path length by ", d)
    end

#=
collect_vertices_length(p::Loc, vs::Locs, d::Real) =
    let pp = vs[1]
        dist = distance(p, pp)
        d <= dist ?
        push!(Vector{Loc}(), p + (pp - p)/dist*d) :
        unshift!(collect_vertices_length(pp, vs[2:end], d - dist), pp)
    end

subpath(path::OpenPolygonalPath, a::Real, b::Real) =
    let p = path.vertices[1]
        pts = []
        for i in 2:length(path.vertices)
            pp = path.vertices[i]
            delta = distance(p, pp)
            if a <= delta
                phi = pol_phi(pp - p)
                p0 = add_pol(p, d, phi)
                return open_polygonal_path(
                        unshift!(collect_vertices_length(p0,
                                                         vcat(path.vertices[i:end], path.vertices),
                                                         delta_d),
                                 p0))
            else
                p = pp
                d -= delta
            end
        end
        error("Exceeded path length")
    end

=#

meta_program(p::OpenPolygonalPath) =
    Expr(:call, :open_polygonal_path, meta_program(p.vertices))



# Path can be made of subparts
abstract type PathOp end
#struct MoveToOp <: PathOp loc::Loc end
#struct MoveOp <: PathOp vec::Vec end
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
open_path_ops(start, ops...) = PathOps(start, [ops...], false)
closed_path_ops(start, ops...) = PathOps(start, [ops...], true)

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

#translate_op(op::MoveToOp, v) = MoveToOp(op.loc + v)
#translate_op(op::LineToOp, v) = LineToOp(op.loc + v)
#translate_op(op::LineToXThenToYOp, v) = LineToXThenToYOpp(op.loc + v)
#translate_op(op::LineToYThenToXOp, v) = LineToYThenToXOp(op.loc + v)
#translate_op(op::PathOp, v) = op

location_at_length(path::PathOps, d::Real) =
  let ops = path.ops,
      start = path.start
    for op in ops
      delta = path_length(op)
      if d < delta + path_tolerance()
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
      elseif d < delta + path_tolerance()
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
      elseif d < delta + path_tolerance()
        return PathOps(path.start, [ops[1:i-1]..., subpath_ending_at(op, d)], false)
      else
        d -= delta
      end
    end
    error("Exceeded path length by ", d)
  end

subpath_starting_at(pathOp::LineOp, d::Real) =
  let len = path_length(pathOp.vec)
    LineOp(pathOp.vec*(len-d)/len)
  end
subpath_starting_at(pathOp::ArcOp, d::Real) =
  let a = d/pathOp.radius
    ArcOp(pathOp.radius, pathOp.start_angle + a, pathOp.amplitude - a)
  end

subpath_ending_at(pathOp::LineOp, d::Real) =
  let len = path_length(pathOp.vec)
    LineOp(pathOp.vec*d/len)
  end
subpath_ending_at(pathOp::ArcOp, d::Real) =
  let a = d/pathOp.radius
    ArcOp(pathOp.radius, pathOp.start_angle, a)
  end

subpath(path::Path, a::Real, b::Real) =
  subpath_starting_at(subpath_ending_at(path, b), a)

# A path sequence is a sequence of paths where the next element of the sequence
# starts at the same place where the previous element ends.

struct OpenPathSequence <: OpenPath
  paths::Vector{<:Path}
end
open_path_sequence(paths...) =
  OpenPathSequence(ensure_connected_paths([paths...]))
struct ClosedPathSequence <: ClosedPath
  paths::Vector{<:Path}
end
closed_path_sequence(paths...) =
  let start = path_start(paths[1]),
      finish = path_end(paths[end])
    start ≈ finish ?
      ClosedPathSequence(ensure_connected_paths(paths)) :
      ClosedPathSequence(ensure_connected_paths([paths..., open_polygonal_path([finish, start])]))
  end

PathSequence = Union{OpenPathSequence, ClosedPathSequence}

# HACK : Include treatment of empty paths.



path_sequence(paths...) =
  let paths = ensure_connected_paths([paths...])
    coincident_path_location(path_start(paths[1]), path_end(paths[end])) ?
      ClosedPathSequence(paths) :
      OpenPathSequence(paths)
  end

ensure_connected_paths(paths) = filter(!is_empty_path, paths)

path_length(path::PathSequence) =
  sum(map(path_length, path.paths))
location_at_length(path::PathSequence, d::Real) =
  let paths = path.paths
    for path in paths
      delta = path_length(path)
      if d <= delta + path_tolerance()
          return location_at_length(path, d)
      else
          d -= delta
      end
    end
    error("Exceeded path length by ", d)
  end
subpath_starting_at(path::PathSequence, d::Real) =
  let subpaths = path.paths
    for i in 1:length(subpaths)
      subpath = subpaths[i]
      delta = path_length(subpath)
      if d == delta
        return OpenPathSequence(ops[i+1:end])
      elseif d < delta + path_tolerance()
        subpath = subpath_starting_at(subpath, d)
        return OpenPathSequence([subpath, subpaths[i+1:end]...])
      else
        d -= delta
      end
    end
    error("Exceeded path length by ", d)
  end
subpath_ending_at(path::PathSequence, d::Real) =
  let subpaths = path.paths
    for i in 1:length(subpaths)
      subpath = subpaths[i]
      delta = path_length(subpath)
      if d == delta
        return OpenPathSequence(subpaths[1:i])
      elseif d < delta + path_tolerance()
        return OpenPathSequence([subpaths[1:i-1]..., subpath_ending_at(subpath, d)])
      else
        d -= delta
      end
    end
    error("Exceeded path length by ", d)
  end

convert(::Type{PathOps}, path::PathSequence) =
  let start = in_world(location_at_length(path, 0)),
      ops = mapfoldl(convert_to_path_ops, vcat, path.paths, init=PathOp[])
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
struct PathSet <: Path
  paths::Vector{<:Path}
end

# Should we just use tuples instead of arrays?
path_set(paths...) =
  PathSet([paths...])

region(outer, inners...) =
  Region([convert(ClosedPath, outer), convert.(ClosedPath, inners)...])
outer_path(region::Region) = region.paths[1]
inner_paths(region::Region) = region.paths[2:end]

region(r::Region, inners...) =
  region(outer_path(r), inner_paths(r)..., inners...)

# Convertions from/to paths
convert(::Type{Region}, vs::Locs) =
  region(closed_polygonal_path(vs))
convert(::Type{Region}, path::ClosedPath) =
  region(path)

# Operations on path containers
translate(path::T, v::Vec) where T<:Union{PathSequence,PathSet,Region} =
  T(translate.(path.paths, v))
scale(path::T, s::Real, p::Loc=u0()) where T<:Union{PathSequence,PathSet,Region} =
  T(scale.(path.paths, s, p))

in_cs(path::T, cs::CS) where T<:Union{PathSequence,PathSet,Region} =
  T(map(p->in_cs(p, cs), path.paths))
planar_path_normal(path::T) where T<:Union{PathSequence,PathSet,Region} =
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
#=
convert(::Type{ClosedPath}, p::OpenPath) =
  if isa(p.ops[end], CloseOp) || coincident_path_location(path_start(p), path_end(p))
    closed_path(p.ops)
  else
    error("Can't convert to a Closed Path: $p")
  end
convert(::Type{ClosedPath}, ops::Vector{<:PathOp}) =
  if isa(ops[end], CloseOp)
    closed_path(ops)
  else
    error("Can't convert to a Closed Path: $ops")
  end
convert(::Type{OpenPath}, ops::Vector{<:PathOp}) = open_path(ops)
convert(::Type{Path}, ops::Vector{<:PathOp}) =
  if isa(ops[end], CloseOp)
    closed_path(ops)
  else
    open_path(ops)
  end
=#
convert(::Type{ClosedPolygonalPath}, path::RectangularPath) =
  let p = path.corner,
      dx = path.dx,
      dy = path.dy
    closed_polygonal_path([p, add_x(p, dx), add_xy(p, dx, dy), add_y(p, dy)])
  end

path_interpolated_lengths(path, t0=0, t1=path_length(path), epsilon=collinearity_tolerance(), min_recursion=1) =
  let p0 = location_at_length(path, t0),
      p1 = location_at_length(path, t1),
      tm = (t0 + t1)/2.0,
      pm = location_at_length(path, tm)
    min_recursion < 0 && collinear_points(p0, pm, p1, epsilon) ?
      [t0, tm, t1] :
      [path_interpolated_lengths(path, t0, tm, epsilon, min_recursion - 1)...,
       path_interpolated_lengths(path, tm, t1, epsilon, min_recursion - 1)[2:end]...]
  end

convert(::Type{ClosedPolygonalPath}, path::CircularPath) =
  closed_polygonal_path(path_interpolated_frames(path)[1:end-1])
convert(::Type{OpenPolygonalPath}, path::CircularPath) =
  open_polygonal_path(path_interpolated_frames(path))
convert(::Type{ClosedPolygonalPath}, path::EllipticPath) =
  closed_polygonal_path(path_interpolated_frames(path)[1:end-1])
convert(::Type{OpenPolygonalPath}, path::EllipticPath) =
  open_polygonal_path(path_interpolated_frames(path))
convert(::Type{OpenPolygonalPath}, path::ArcPath) =
  open_polygonal_path(path_interpolated_frames(path))
convert(::Type{OpenPolygonalPath}, path::ClosedPolygonalPath) =
  open_polygonal_path(vcat(path.vertices, [path.vertices[1]]))
convert(::Type{OpenPolygonalPath}, path::RectangularPath) =
  convert(OpenPolygonalPath, convert(ClosedPolygonalPath, path))
convert(::Type{OpenPolygonalPath}, path::OpenSplinePath) =
  open_polygonal_path(map_division(identity, path))


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

convert(::Type{ClosedPolygonalPath}, path::ClosedPathSequence) =
  let paths = path.paths,
      vertices = []
    for path in paths
      append!(vertices, convert(OpenPolygonalPath, path).vertices[1:end-1])
    end
    closed_polygonal_path(vertices)
  end

convert(::Type{OpenPolygonalPath}, path::OpenPathSequence) =
  let paths = path.paths,
      vertices = []
    for path in paths[1:end-1]
      append!(vertices, convert(OpenPolygonalPath, path).vertices[1:end-1])
    end
    append!(vertices, convert(OpenPolygonalPath, paths[end]).vertices)
    open_polygonal_path(vertices)
  end


convert(::Type{ClosedPath}, paths::Vector{<:Path}) =
  ClosedPathSequence(ensure_connected_paths(paths))

# It is possible to convert a PathSet to a singleton path
# by considering the first path as the outer path and all the
# others as inner paths
convert(::Type{ClosedPath}, pset::PathSet) =
  foldl(subtract_paths, pset.paths)

#### Utilities
path_vertices(path::Union{PolygonalPath,SplinePath}) =
  path.vertices
path_vertices(path::Path) =
  path_vertices(convert(is_closed_path(path) ? ClosedPolygonalPath : OpenPolygonalPath,
                        path))

iterate(path::PolygonalPath, i=1) =
  i > length(path.vertices) ?
    nothing :
    (path.vertices[i], i+1)

loc_from_o_vx_vz(o, vx, vz) =
  norm(vx) < min_norm ?
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
subpaths(path::Path) = subpaths(convert(OpenPolygonalPath, path))

subtract_paths(path1::Path, path2::Path) =
  subtract_paths(convert(ClosedPolygonalPath, path1), convert(ClosedPolygonalPath, path2))
subtract_paths(path1::ClosedPolygonalPath, path2::ClosedPolygonalPath) =
  closed_polygonal_path(
    subtract_polygon_vertices(
      path_vertices(path1),
      path_vertices(path2)))

## Smoothnesss
# A smooth curve means that the curve is differentiable up to an intended order
# In practice, it means that it does not have "corners"
is_smooth_path(path::Path) = false
is_smooth_path(pts::Locs) = false
is_smooth_path(path::Union{ArcPath,CircularPath,SplinePath}) = true

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
path_on(path::CircularPath, p) =
  circular_path(on_cs(path.center, p), path.radius)
path_on(path::EllipticPath, p) =
  elliptic_path(on_cs(path.center, p), path.r1, path.r2)
path_on(path::RectangularPath, p) =
  rectangular_path(on_cs(path.corner, p), path.dx, path.dy)
path_on(path::OpenPolygonalPath, p) =
  open_polygonal_path(on_cs(path_vertices(path), p))
path_on(path::ClosedPolygonalPath, p) =
  closed_polygonal_path(on_cs(path_vertices(path), p))
path_on(path::OpenSplinePath, p) =
  open_spline_path(on_cs(path_vertices(path), p))
path_on(path::Region, p) =
  region([path_on(path, p) for path in path.paths])
path_on(path::ClosedPathSequence, p) =
  closed_path_sequence([path_on(path, p) for path in path.paths]...)

## Utility operations
Base.reverse(path::CircularPath) =
  circular_path(loc_from_o_vz(path.center, vz(-1, path.center.cs)), path.radius)
Base.reverse(path::RectangularPath) =
  rectangular_path(loc_from_o_vz(path.corner, vz(-1, path.corner.cs)), -path.dx, path.dy)
Base.reverse(path::OpenPolygonalPath) =
  open_polygonal_path(reverse(path_vertices(path)))
Base.reverse(path::ClosedPolygonalPath) =
  closed_polygonal_path(reverse(path_vertices(path)))
Base.reverse(path::ClosedPathSequence) =
  ClosedPathSequence(reverse(reverse.(path.paths)))
Base.reverse(path::Region) =
  region(reverse.(path.paths)...)


mirrored_path(path::Path, p::Loc, v::Vec) =
  error("Mush be finished")

mirrored_on_x(path::OpenPolygonalPath) =
  join_paths(path, open_polygonal_path(reverse(map(p->xyz(p.x, -p.y, -p.z, p.cs), path_vertices(path)))))
mirrored_on_y(path::OpenPolygonalPath) =
  join_paths(path, open_polygonal_path(reverse(map(p->xyz(-p.x, p.y, -p.z, p.cs), path_vertices(path)))))
mirrored_on_z(path::OpenPolygonalPath) =
  join_paths(path, open_polygonal_path(reverse(map(p->xyz(-p.x, -p.y, p.z, p.cs), path_vertices(path)))))

export mesh
struct Mesh <: Path
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
  t1 - t0 < epsilon() ?
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
  