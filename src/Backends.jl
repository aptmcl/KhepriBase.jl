#=
Khepri backends need to implement a layered set of virtual operations that
support the higher-level user-accessible operations. Each backend needs to
directly implement a subset of these operations in order to be usable by Khepri.

These operations are layered because different backends might want to support
them at different levels of abstraction. For example, a CAD backend might want
to support the creation of cylinders but not columns, while a BIM backend might
want to directly support the creation of columns. To handle both these cases,
Khepri provides a higher-level default column operation that relies on a
lower-level cylinder operation.

This is where these operations are specified.

Backends are types parameterized by a key identifying the backend (e.g.,
AutoCAD) and by the type of reference they use
=#

abstract type Backend{K,T} end
backend_name(::Backend) = "AbstractBackend"
show(io::IO, b::Backend) = print(io, backend_name(b))

# Backends need to implement operations or an exception is triggered
struct UnimplementedBackendOperationException <: Exception
	backend
	operation
  args
end
showerror(io::IO, e::UnimplementedBackendOperationException) =
	print(io, "Operation $(e.operation) is not available in backend $(e.backend).$(length(e.args) > 0 ? "Args: $(e.args)" : "")")

missing_specialization(b::Backend, oper=:unknown_operation, args...) =
  error(UnimplementedBackendOperationException(b, oper, args))

macro bdef(call)
  name, escname, params = call.args[1], esc(call.args[1]), esc.(call.args[2:end])
  quote
    export $(escname)
    $(escname)(b::Backend, $(params...)) =
      missing_specialization(b, $(QuoteNode(name)), $(params...))
  end
end

@bdef(void_ref())
############################################################
# Zeroth tier: curves. Not all backends support these.

export b_point, b_line, b_closed_line, b_polygon, b_regular_polygon,
       b_nurbs_curve,
       b_spline, b_closed_spline, b_circle, b_arc, b_rectangle

@bdef(b_point(p, mat))

@bdef(b_line(ps, mat))

b_polygon(b::Backend, ps, mat) =
  b_line(b, [ps..., p[1]], mat)

b_regular_polygon(b::Backend, edges, c, r, angle, inscribed, mat) =
  b_polygon(b, regular_polygon_vertices(edges, c, r, angle, inscribed), mat)

b_nurbs_curve(b::Backend, order, ps, knots, weights, closed, mat) =
  b_line(b, ps, closed, mat)

b_spline(b::Backend, ps, v1, v2, interpolator, mat) =
  let ci = curve_interpolator(ps, false),
      cpts = curve_control_points(ci),
      n = length(cpts),
      knots = curve_knots(ci)
    b_nurbs_curve(b, 5, cpts, knots, fill(1.0, n), false, mat)
  end

b_closed_spline(b::Backend, ps, mat) =
  let ci = curve_interpolator(ps, true),
      cpts = curve_control_points(ci),
      n = length(cpts),
      knots = curve_knots(ci)
    b_nurbs_curve(b, 5, cpts, knots, fill(1.0, n), true, mat)
  end

b_circle(b::Backend, c, r, mat) =
  b_closed_spline(b, regular_polygon_vertices(32, c, r, 0, true), mat)

b_arc(b::Backend, c, r, α, Δα, mat) =
  b_spline(b,
    [center + vpol(r, a, center.cs)
     for a in division(α, α + Δα, Δα*32/2/π, false)],
    nothing, nothing, # THIS NEEDS TO BE FIXED
    mat)

b_rectangle(b::Backend, c, dx, dy, mat) =
  b_polygon(b, [c, add_x(c, dx), add_xy(c, dx, dy), add_y(c, dy)], mat)

# First tier: everything is a triangle or a set of triangles
export b_trig, b_quad, b_ngon, b_quad_strip, b_quad_strip_closed

@bdef(b_trig(p1, p2, p3))

# By default, we silently drop the mat
b_trig(b::Backend, p1, p2, p3, mat) =
  b_trig(b, p1, p2, p3)

b_quad(b::Backend, p1, p2, p3, p4, mat) =
  [b_trig(b, p1, p2, p3, mat),
   b_trig(b, p1, p3, p4, mat)]

b_ngon(b::Backend, ps, pivot, smooth, mat) =
  [(b_trig(b, pivot, ps[i], ps[i+1], mat)
    for i in 1:size(ps,1)-1)...,
	 b_trig(b, pivot, ps[end], ps[1], mat)]

b_quad_strip(b::Backend, ps, qs, smooth, mat) =
  [b_quad(b, ps[i], ps[i+1], qs[i+1], qs[i], mat)
   for i in 1:size(ps,1)-1]

b_quad_strip_closed(b::Backend, ps, qs, smooth, mat) =
  b_quad_strip(b, [ps..., ps[1]], [qs..., qs[1]], smooth, mat)

############################################################
# Second tier: surfaces
export b_surface_polygon, b_surface_regular_polygon,
	   b_surface_circle, b_surface_arc, b_surface_closed_spline,
	   b_surface, b_surface_grid, b_smooth_surface_grid

b_surface_polygon(b::Backend, ps, mat) =
  # This only works for convex polygons
  b_ngon(b, ps, trig_center(ps[1], ps[2], ps[3]), false, mat)

b_surface_polygon_with_holes(b::Backend, ps, qss, mat) =
  # By default, we use half-edges
  b_surface_polygon(b, foldl(subtract_polygon_vertices, qss, init=ps), mat)

b_surface_rectangle(b::Backend, c, dx, dy, mat) =
  b_quad(b, c, add_x(c, dx), add_xy(c, dx, dy), add_y(c, dy), mat)

b_surface_regular_polygon(b::Backend, edges, c, r, angle, inscribed, mat) =
  b_ngon(b, regular_polygon_vertices(edges, c, r, angle, inscribed), c, false, mat)

b_surface_circle(b::Backend, c, r, mat) =
  b_surface_regular_polygon(b, 32, c, r, 0, true, mat)

b_surface_arc(b::Backend, c, r, α, Δα, mat) =
  b_ngon(b,
         [center + vpol(r, a, center.cs)
          for a in division(α, α + Δα, Δα*32/2/π, false)],
         c, false, mat)

@bdef(b_surface_closed_spline(ps, mat))

b_surface_grid(b::Backend, ptss, closed_u, closed_v, smooth_u, smooth_v, interpolator, mat) =
  let (nu, nv) = size(ptss)
	if smooth_u && smooth_v
      ptss = [location_at_grid_interpolator(interpolator, u, v)
	          for u in division(0, 1, 4*nu-7), v in division(0, 1, 4*nv-7)] # 2->1, 3->5, 4->9, 5->13
	  (nu, nv) = size(ptss)
	end
	closed_u ?
      vcat([b_quad_strip_closed(b, ptss[i,:], ptss[i+1,:], smooth_u, mat) for i in 1:nu-1],
	       closed_v ? [b_quad_strip_closed(b, ptss[end,:], ptss[1,:], smooth_u, mat)] : []) :
	  vcat([b_quad_strip(b, ptss[i,:], ptss[i+1,:], smooth_u, mat) for i in 1:nu-1],
	       closed_v ? [b_quad_strip(b, ptss[end,:], ptss[1,:], smooth_u, mat)] : [])
  end

b_smooth_surface_grid(b::Backend, ptss, closed_u, closed_v, mat) =
  b_surface_grid(b, smooth_grid(ptss), closed_u, closed_v, true, true, mat)

# Parametric surface
#=
parametric {
    function { sin(u)*cos(v) }
    function { sin(u)*sin(v) }
    function { cos(u) }

    <0,0>, <2*pi,pi>
    contained_by { sphere{0, 1.1} }
    max_gradient ??
    accuracy 0.0001
    precompute 10 x,y,z
    pigment {rgb 1}
  }
=#

############################################################
# Third tier: solids
export b_generic_pyramid_frustum, b_generic_pyramid, b_generic_prism,
       b_generic_pyramid_frustum_with_holes, b_generic_prism_with_holes,
  	   b_pyramid_frustum, b_pyramid, b_prism,
  	   b_regular_pyramid_frustum, b_regular_pyramid, b_regular_prism,
  	   b_cylinder,
  	   b_cuboid,
  	   b_box,
  	   b_sphere,
  	   b_cone,
  	   b_torus

# Each solid can have just one material or multiple materials
b_generic_pyramid_frustum(b::Backend, bs, ts, smooth, bmat, tmat, smat) =
  [b_surface_polygon(b, reverse(bs), bmat),
   b_quad_strip_closed(b, bs, ts, smooth, smat),
   b_surface_polygon(b, ts, tmat)]

b_generic_pyramid_frustum_with_holes(b::Backend, bs, ts, smooth, bbs, tts, smooths, bmat, tmat, smat) =
  [b_surface_polygon_with_holes(b, reverse(bs), bbs, bmat),
   b_quad_strip_closed(b, bs, ts, smooth, smat),
   [b_quad_strip_closed(b, bs, ts, smooth, smat)
    for (bs, ts, smooth) in zip(bbs, tts, smooths)]...,
   b_surface_polygon_with_holes(b, ts, reverse.(tts), tmat)]

b_generic_pyramid(b::Backend, bs, t, smooth, bmat, smat) =
	[b_surface_polygon(b, reverse(bs), bmat),
	 b_ngon(b, bs, t, smooth, smat)]

b_generic_prism(b::Backend, bs, smooth, v, bmat, tmat, smat) =
  b_generic_pyramid_frustum(b, bs, translate(bs, v), smooth, bmat, tmat, smat)

b_generic_prism_with_holes(b::Backend, bs, smooth, bss, smooths, v, bmat, tmat, smat) =
  b_generic_pyramid_frustum_with_holes(b, bs, translate(bs, v), smooth, bss, translate.(bss, v), smooths, bmat, tmat, smat)

b_pyramid_frustum(b::Backend, bs, ts, mat) =
  b_pyramid_frustum(b, bs, ts, mat, mat, mat)

b_pyramid_frustum(b::Backend, bs, ts, bmat, tmat, smat) =
  b_generic_pyramid_frustum(b, bs, ts, false, bmat, tmat, smat)

b_pyramid(b::Backend, bs, t, mat) =
	b_pyramid(b, bs, t, mat, mat)
b_pyramid(b::Backend, bs, t, bmat, smat) =
  b_generic_pyramid(b, bs, t, false, bmat, smat)

b_prism(b::Backend, bs, v, mat) =
	b_prism(b, bs, v, mat, mat, mat)
b_prism(b::Backend, bs, v, bmat, tmat, smat) =
  b_pyramid_frustum(b, bs, translate(bs, v), bmat, tmat, smat)

b_regular_pyramid_frustum(b::Backend, edges, cb, rb, angle, h, rt, inscribed, mat) =
	b_regular_pyramid_frustum(b, edges, cb, rb, angle, h, rt, inscribed, mat, mat, mat)
b_regular_pyramid_frustum(b::Backend, edges, cb, rb, angle, h, rt, inscribed, bmat, tmat, smat) =
  b_pyramid_frustum(
    b,
    regular_polygon_vertices(edges, cb, rb, angle, inscribed),
    regular_polygon_vertices(edges, add_z(cb, h), rt, angle, inscribed),
    bmat, tmat, smat)

b_regular_pyramid(b::Backend, edges, cb, rb, angle, h, inscribed, mat) =
	b_regular_pyramid(b, edges, cb, rb, angle, h, inscribed, mat, mat)
b_regular_pyramid(b::Backend, edges, cb, rb, angle, h, inscribed, bmat, smat) =
  b_pyramid(
  	b,
  	regular_polygon_vertices(edges, cb, rb, angle, inscribed),
  	add_z(cb, h),
  	bmat, smat)

b_regular_prism(b::Backend, edges, cb, rb, angle, h, inscribed, mat) =
	b_regular_prism(b, edges, cb, rb, angle, h, inscribed, mat, mat, mat)
b_regular_prism(b::Backend, edges, cb, rb, angle, h, inscribed, bmat, tmat, smat) =
	b_regular_pyramid_frustum(b, edges, cb, rb, angle, h, rt, inscribed, bmat, tmat, smat)

b_cylinder(b::Backend, cb, r, h, mat) =
	b_cylinder(b, cb, r, h, mat, mat, mat)
b_cylinder(b::Backend, cb, r, h, bmat, tmat, smat) =
  b_generic_prism(
  	b,
  	regular_polygon_vertices(32, cb, r, 0, true),
  	true,
    vz(h, cb.cs),
  	bmat, tmat, smat)

b_cuboid(b::Backend, pb0, pb1, pb2, pb3, pt0, pt1, pt2, pt3, mat) =
  [b_quad(b, pb3, pb2, pb1, pb0, mat),
   b_quad_strip_closed(b, [pb0, pb1, pb2, pb3], [pt0, pt1, pt2, pt3], false, mat),
   b_quad(b, pt0, pt1, pt2, pt3, mat)]

b_box(b::Backend, c, dx, dy, dz, mat) =
  let pb0 = c,
      pb1 = add_x(c, dx),
      pb2 = add_xy(c, dx, dy),
      pb3 = add_y(c, dy),
      pt0 = add_z(pb0, dz),
      pt1 = add_z(pb1, dz),
      pt2 = add_z(pb2, dz),
      pt3 = add_z(pb3, dz)
    b_cuboid(b, pb0, pb1, pb2, pb3, pt0, pt1, pt2, pt3, mat)
  end

b_sphere(b::Backend, c, r, mat) =
  let ϕs = division(0, 2π, 32, false)
    [b_ngon(b, [add_sph(c, r, ϕ, π/16) for ϕ in ϕs], add_sph(c, r, 0, 0), true, mat),
  	 [b_quad_strip_closed(b,
  			[add_sph(c, r, ϕ, ψ+π/16) for ϕ in ϕs],
  			[add_sph(c, r, ϕ, ψ) for ϕ in ϕs],
  			true, mat) for ψ in π/16:π/16:π-π/16]...,
  	 b_ngon(b, [add_sph(c, r, ϕ, π-π/16) for ϕ in ϕs], add_sph(c, r, 0, π), true, mat)]
	end

b_cone(b::Backend, cb, r, h, mat) =
	b_cone(b, cb, r, h, mat, mat)

b_cone(b::Backend, cb, r, h, bmat, smat) =
	b_generic_pyramid(
		b,
		regular_polygon_vertices(32, cb, r, 0, true),
		add_z(cb, h),
		true,
		bmat, smat)

b_cone_frustum(b::Backend, cb, rb, h, rt, mat) =
	b_cone_frustum(b, cb, rb, h, rt, mat, mat, mat)

b_cone_frustum(b::Backend, cb, rb, h, rt, bmat, tmat, smat) =
  b_generic_pyramid_frustum(
  	b,
  	regular_polygon_vertices(32, cb, rb, 0, true),
  	regular_polygon_vertices(32, add_z(cb, h), rt, 0, true),
	true,
  	bmat, tmat, smat)

b_torus(b::Backend, c, ra, rb, mat) =
  b_surface_grid(
  	b,
	[add_sph(add_pol(c, ra, ϕ), rb, ϕ, ψ)
	 for ψ in division(0, 2π, 16, false), ϕ in division(0, 2π, 32, false)],
    true, true,
	true, true,
	mat)

##################################################################
# Paths and Regions
b_surface(b::Backend, path::ClosedPath, mat) =
  b_surface_polygon(b, path_vertices(path), mat)

b_surface(b::Backend, region::Region, mat) =
  b_surface_polygon_with_holes(
    b,
    path_vertices(outer_path(region)),
    [reverse(path_vertices(path)) for path in inner_paths(region)],
    mat)

# In theory, this should be implemented using a loft
b_path_frustum(b::Backend, bpath, tpath, bmat, tmat, smat) =
  let blength = path_length(bpath),
	  tlength = path_length(tpath),
	  n = max(length(path_vertices(bpath)), length(path_vertices(bpath))),
	  bs = division(bpath, n),
	  ts = division(tpath, n)
	  # We should rotate one of the vertices array to minimize the distance
	  # between corresponding so that they align better.
	b_generic_pyramid_frustum(
	  b, bs, ts,
	  is_smooth_path(bpath) || is_smooth_path(tpath),
	  bmat, tmat, smat)
	end

# Extruding a profile
b_extrusion(b::Backend, profile, v, cb, mat) =
  b_extrusion(b, profile, v, cb, mat, mat, mat)

b_extrusion(b::Backend, profile, v, cb, bmat, tmat, smat) =
  let path = profile
  	b_generic_prism(
  	  b,
  	  path_vertices_on(path, cb),
  	  is_smooth_path(path),
      v,
  	  bmat, tmat, smat)
  end

b_extrusion(b::Backend, profile::CircularPath, v, cb, bmat, tmat, smat) =
  v.cs === cb.cs && iszero(v.x) && iszero(v.y) ?
  	b_cylinder(b, add_xy(cb, profile.center.x, profile.center.y), profile.radius, v.z, bmat, tmat, smat) :
	b_generic_prism(b,
	  path_vertices_on(profile, cb),
	  is_smooth_path(profile),
	  v,
	  bmat, tmat, smat)

b_extrusion(b::Backend, profile::Region, v, cb, bmat, tmat, smat) =
  let outer = outer_path(profile),
      inners = inner_paths(profile)
    isempty(inners) ?
      b_generic_prism(b,
        path_vertices_on(outer, cb),
        is_smooth_path(outer),
        v,
        bmat, tmat, smat) :
      b_generic_prism_with_holes(b,
        path_vertices_on(outer, cb),
        is_smooth_path(outer),
        path_vertices_on.(inners, cb),
        is_smooth_path.(inners),
        v,
        bmat, tmat, smat)
  end

# Lofts, sweeps, etc
export b_sweep, b_loft
b_loft(b::Backend, profiles, closed, smooth, mat) =
  let ptss = path_vertices.(profiles),
	  n = mapreduce(length, max, ptss),
	  vss = map(profile->map_division(identity, profile, n), profiles)
	b_surface_grid(b, hcat(vss...), is_closed_path(profiles[1]), closed, is_smooth_path(profiles[1]), smooth, mat)
  end

b_sweep(b::Backend, path, profile, rotation, scaling, mat) =
#  let vertices = in_world.(path_vertices(profile)),
#      frames = map_division(identity, path, 100),
#	  points = [xyz(cx(p), cy(p), cz(p), frame.cs) for p in vertices, frame in frames]
  let frames = rotation_minimizing_frames(path_frames(path)),
	  profiles = map_division(s->scale(profile, 1) #=2.1+sin(s)), 0, 32pi,=#, 0, 1, 100),# #map_division(identity, path, 100), #
	  verticess = map(path_vertices, profiles),
	  verticess = is_smooth_path(profile) ?
	  	let n = mapreduce(length, max, verticess)-1
		  map(path->map_division(identity, path, n), profiles)
	    end :
	    verticess
	  points = hcat(map(path_vertices_on, verticess, frames)...)
    b_surface_grid(
      b,
      points,
      is_closed_path(path),
	  is_closed_path(profile),
      is_smooth_path(path),
	  is_smooth_path(profile),
	  grid_interpolator(points),
	  mat)
  end

# Stroke and fill operations over paths
# This is specially useful for debuging

#=
backend_stroke_op(b::Backend, op::MoveToOp, start::Loc, curr::Loc, refs) =
    (op.loc, op.loc, refs)
backend_stroke_op(b::Backend, op::MoveOp, start::Loc, curr::Loc, refs) =
    (start, curr + op.vec, refs)
backend_stroke_op(b::Backend, op::LineToOp, start::Loc, curr::Loc, refs) =
    (start, op.loc, push!(refs, backend_stroke_line(b, [curr, op.loc])))
backend_stroke_op(b::Backend, op::LineXThenYOp, start::Loc, curr::Loc, refs) =
    (start,
     start + op.vec,
     push!(refs, backend_stroke_line(b, [curr, curr + vec_in(op.vec, curr.cs).x, curr + op.vec])))
backend_stroke_op(b::Backend, op::LineYThenXOp, start::Loc, curr::Loc, refs) =
    (start,
     start + op.vec,
     push!(refs, backend_stroke_line(b, [curr, curr + vec_in(op.vec, curr.cs).y, curr + op.vec])))
backend_stroke_op(b::Backend, op::LineToXThenToYOp, start::Loc, curr::Loc, refs) =
    (start, op.loc, push!(refs, backend_stroke_line(b, [curr, xy(curr.x, loc_in(op.loc, curr.cs).x, curr.cs), op.loc])))
backend_stroke_op(b::Backend, op::LineToYThenToXOp, start::Loc, curr::Loc, refs) =
    (start, op.loc, push!(refs, backend_stroke_line(b, [curr, xy(curr.x, loc_in(op.loc, curr.cs).y, curr.cs), op.loc])))
=#

b_stroke(b::Backend, path::CircularPath, mat) =
  b_circle(b, path.center, path.radius, mat)
b_stroke(b::Backend, path::RectangularPath, mat) =
  b_rectangle(b, path.corner, path.dx, path.dy, mat)
b_stroke(b::Backend, path::ArcPath, mat) =
  b_arc(b, path.center, path.radius, path.start_angle, path.amplitude, mat)
b_stroke(b::Backend, path::OpenPolygonalPath, mat) =
  b_line(b, path.vertices, mat)
b_stroke(b::Backend, path::ClosedPolygonalPath, mat) =
  b_polygon(b, path.vertices, mat)
b_stroke(b::Backend, path::OpenSplinePath, mat) =
  b_spline(b, path.vertices, path.v0, path.v1, path.interpolator, mat)
b_stroke(b::Backend, path::ClosedSplinePath, mat) =
  b_closed_spline(b, path.vertices, mat)
b_stroke(b::Backend, path::Region, mat) =
  [b_stroke(b, path, mat) for path in path.paths]
b_stroke(b::Backend, path::Mesh, mat) =
  let vs = m.vertices
    for face in m.faces
      b_line(b, vs[face.+1], mat) #1-indexed
    end
  end

b_fill(b::Backend, path::CircularPath, mat) =
  b_surface_circle(b, path.center, path.radius, mat)
b_fill(b::Backend, path::RectangularPath, mat) =
  b_surface_rectangle(b, path.corner, path.dx, path.dy, mat)
b_fill(b::Backend, path::ClosedPolygonalPath, mat) =
  b_surface_polygon(b, path.vertices, mat)
b_fill(b::Backend, path::ClosedSplinePath, mat) =
  b_surface_closed_spline(b, path.vertices, mat)
b_fill(b::Backend, path::Region, mat) =
  b_surface(b, path, mat)

#=
backend_fill(b::Backend, path::ClosedSplinePath) =
  backend_fill_curves(b, @remote(b, ClosedSpline(path.vertices)))

b_stroke_unite(b::Backend, refs) = @remote(b, JoinCurves(refs))

backend_fill_curves(b::Backend, gs::Guids) = @remote(b, SurfaceFrom(gs))
backend_fill_curves(b::Backend, g::Guid) = @remote(b, SurfaceFrom([g]))

b_stroke_arc(b::Backend, center::Loc, radius::Real, start_angle::Real, amplitude::Real) =
    let end_angle = start_angle + amplitude
        if end_angle > start_angle
            @remote(b, Arc(center, vx(1, center.cs), vy(1, center.cs), radius, start_angle, end_angle))
        else
            @remote(b, Arc(center, vx(1, center.cs), vy(1, center.cs), radius, end_angle, start_angle))
        end
    end
=#

##################################################################
# Dimensions
#=
Technical drawings need dimensions.
We need extension lines and dimension lines (with or without arrows)
and with a text.
Guidelines say that an extension line should be start 1mm away from
the object and extend 2mm beyond the dimension line. This all depends
on the scale we are using so I'll include a size parameter and I'll
consider the inicial spacing as 5% of the size and the extension to
be extra 10% in excess of the size.
=#
export b_dimension, b_ext_line, b_dim_line, b_text, b_text_size

b_dimension(b::Backend, p, q, str, size, mat) =
  let qp = in_world(q - p),
	  phi = pol_phi(qp),
	  outside = pi/2 <= phi <= 3pi/2,
	  v = vpol(outside ? size : 2*size, phi-pi/2),
	  uv = unitized(v),
	  (si, se) = (0.1*size, 0.2*size),
	  (vi, ve) = (uv*si, uv*se),
	  (tp, tq, tv) = outside ? (q, p, vpol(1, phi + pi)) : (p, q, vpol(1, phi))
	[b_ext_line(b, p + vi, p + v + ve, mat),
     b_ext_line(b, q + vi, q + v + ve, mat),
     b_dim_line(b, tp + v, tq + v, tv, str, size, outside, mat)]
  end
b_ext_line(b::Backend, p, q, mat) =
  b_line(b, [p, q], mat)
b_dim_line(b::Backend, p, q, tv, str, size, outside, mat) =
  let (minx, maxx, miny, maxy) = b_text_size(b, str, size, mat),
	  tp = p + tv*((distance(p, q)-(maxx-minx))/2)
    [b_line(b, [p, q], mat),
     b_text(b, str, add_y(loc_from_o_vx(tp, tv), size*0.1-miny), size, mat)]
  end

# To ensure a portable font, we will 'draw' the letters
const letter_glyph = Dict(
  ' '=>(bb=[(0, 0), (2/3, 1)], vss=[]),
  '!'=>(bb=[(0, 0), (0, 1)], vss=[[(0, 0), (0, 1/6)], [(0, 1/3), (0, 1)]]),
  '"'=>(bb=[(0, 2/3), (1/3, 1)], vss=[[(0, 2/3), (1/6, 1)], [(1/3, 1), (1/6, 2/3)]]),
  '#'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1/3), (2/3, 1/3)], [(2/3, 2/3), (0, 2/3)], [(1/6, 1), (1/6, 0)], [(1/2, 0), (1/2, 1)]]),
  '$'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1/6), (1/2, 1/6), (2/3, 1/3), (1/2, 1/2), (1/6, 1/2), (0, 2/3), (1/6, 5/6), (2/3, 5/6)], [(1/3, 1), (1/3, 0)]]),
  '%'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1), (0, 5/6), (1/6, 5/6), (1/6, 1), (0, 1)], [(2/3, 1), (0, 0)], [(2/3, 0), (1/2, 0), (1/2, 1/6), (2/3, 1/6), (2/3, 0)]]),
  '&'=>(bb=[(0, 0), (2/3, 1)], vss=[[(2/3, 1/3), (1/3, 0), (1/6, 0), (0, 1/6), (0, 1/3), (1/3, 2/3), (1/3, 5/6), (1/6, 1), (0, 5/6), (0, 2/3), (2/3, 0)]]),
  '\''=>(bb=[(0, 2/3), (1/6, 1)], vss=[[(0, 2/3), (1/6, 1)]]),
  '('=>(bb=[(0, 0), (1/3, 1)], vss=[[(1/3, 1), (0, 2/3), (0, 1/3), (1/3, 0)]]),
  ')'=>(bb=[(0, 0), (1/3, 1)], vss=[[(0, 1), (1/3, 2/3), (1/3, 1/3), (0, 0)]]),
  '*'=>(bb=[(0, 1/6), (2/3, 5/6)], vss=[[(1/3, 1/6), (1/3, 5/6)], [(2/3, 1/2), (0, 1/2)], [(2/3, 5/6), (0, 1/6)], [(0, 5/6), (2/3, 1/6)]]),
  '+'=>(bb=[(0, 1/6), (2/3, 5/6)], vss=[[(1/3, 1/6), (1/3, 5/6)], [(2/3, 1/2), (0, 1/2)]]),
  ','=>(bb=[(0, -1/6), (1/6, 1/6)], vss=[[(1/6, 1/6), (1/6, 0), (0, -1/6)]]),
  '-'=>(bb=[(0, 1/2), (2/3, 1/2)], vss=[[(0, 1/2), (2/3, 1/2)]]),
  '.'=>(bb=[(0, 0), (0, 1/6)], vss=[[(0, 0), (0, 1/6)]]),
  '/'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (2/3, 1)]]),
  '0'=>(bb=[(0, 0), (1/2, 1)], vss=[[(1/6, 0), (0, 1/6), (0, 5/6), (1/6, 1), (1/3, 1), (1/2, 5/6), (1/2, 1/6), (1/3, 0), (1/6, 0)]]),
  '1'=>(bb=[(0, 0), (1/3, 1)], vss=[[(0, 5/6), (1/6, 1), (1/6, 0)], [(0, 0), (1/3, 0)]]),
  '2'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 5/6), (1/6, 1), (1/2, 1), (2/3, 5/6), (2/3, 2/3), (1/2, 1/2), (1/6, 1/2), (0, 1/3), (0, 0), (2/3, 0)]]),
  '3'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 5/6), (1/6, 1), (1/2, 1), (2/3, 5/6), (2/3, 2/3), (1/2, 1/2), (1/3, 1/2)], [(1/2, 1/2), (2/3, 1/3), (2/3, 1/6), (1/2, 0), (1/6, 0), (0, 1/6)]]),
  '4'=>(bb=[(0, 0), (2/3, 1)], vss=[[(2/3, 1/3), (0, 1/3), (1/2, 1), (1/2, 0)]]),
  '5'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1/6), (1/6, 0), (1/2, 0), (2/3, 1/6), (2/3, 1/2), (1/2, 2/3), (0, 2/3), (0, 1), (2/3, 1)]]),
  '6'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1/2), (1/2, 1/2), (2/3, 1/3), (2/3, 1/6), (1/2, 0), (1/6, 0), (0, 1/6), (0, 2/3), (1/3, 1), (1/2, 1)]]),
  '7'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1), (2/3, 1), (1/6, 0)]]),
  '8'=>(bb=[(0, 0), (2/3, 1)], vss=[[(1/6, 0), (0, 1/6), (0, 1/3), (1/6, 1/2), (1/2, 1/2), (2/3, 2/3), (2/3, 5/6), (1/2, 1), (1/6, 1), (0, 5/6), (0, 2/3), (1/6, 1/2)], [(1/2, 1/2), (2/3, 1/3), (2/3, 1/6), (1/2, 0), (1/6, 0)]]),
  '9'=>(bb=[(0, 0), (2/3, 1)], vss=[[(1/6, 0), (1/3, 0), (2/3, 1/3), (2/3, 5/6), (1/2, 1), (1/6, 1), (0, 5/6), (0, 2/3), (1/6, 1/2), (2/3, 1/2)]]),
  ':'=>(bb=[(0, 1/6), (0, 2/3)], vss=[[(0, 2/3), (0, 1/2)], [(0, 1/3), (0, 1/6)]]),
  ';'=>(bb=[(0, -1/6), (1/6, 2/3)], vss=[[(1/6, 2/3), (1/6, 1/2)], [(1/6, 1/3), (1/6, 0), (0, -1/6)]]),
  '<'=>(bb=[(0, 0), (1/2, 1)], vss=[[(1/2, 1), (0, 1/2), (1/2, 0)]]),
  '='=>(bb=[(0, 1/3), (2/3, 2/3)], vss=[[(0, 2/3), (2/3, 2/3)], [(2/3, 1/3), (0, 1/3)]]),
  '>'=>(bb=[(0, 0), (1/2, 1)], vss=[[(0, 1), (1/2, 1/2), (0, 0)]]),
  '?'=>(bb=[(0, 0), (1/2, 1)], vss=[[(0, 5/6), (1/6, 1), (1/3, 1), (1/2, 5/6), (1/2, 2/3), (1/3, 1/2), (1/3, 1/3)], [(1/3, 1/6), (1/3, 0)]]),
  '@'=>(bb=[(0, 0), (2/3, 1)], vss=[[(1/2, 1/2), (1/3, 1/3), (1/6, 1/3), (1/6, 1/2), (1/3, 2/3), (1/2, 2/3), (1/2, 1/3), (2/3, 1/2), (2/3, 5/6), (1/2, 1), (1/6, 1), (0, 5/6), (0, 1/6), (1/6, 0), (2/3, 0)]]),
  'A'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1/3), (1/3, 1), (2/3, 1/3), (2/3, 0)], [(0, 1/3), (2/3, 1/3)]]),
  'B'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (1/2, 0), (2/3, 1/6), (2/3, 1/3), (1/2, 1/2), (1/6, 1/2)], [(1/2, 1/2), (2/3, 2/3), (2/3, 5/6), (1/2, 1), (0, 1)], [(1/6, 1), (1/6, 0)]]),
  'C'=>(bb=[(0, 0), (2/3, 1)], vss=[[(2/3, 1/6), (1/2, 0), (1/6, 0), (0, 1/6), (0, 5/6), (1/6, 1), (1/2, 1), (2/3, 5/6)]]),
  'D'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (1/2, 0), (2/3, 1/6), (2/3, 5/6), (1/2, 1), (0, 1)], [(1/6, 1), (1/6, 0)]]),
  'E'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1), (2/3, 1)], [(0, 1/2), (1/3, 1/2)], [(0, 0), (2/3, 0)]]),
  'F'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1), (2/3, 1)], [(0, 1/2), (1/3, 1/2)]]),
  'G'=>(bb=[(0, 0), (2/3, 1)], vss=[[(1/2, 1/2), (2/3, 1/2), (2/3, 0), (1/6, 0), (0, 1/6), (0, 5/6), (1/6, 1), (2/3, 1)]]),
  'H'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1)], [(0, 1/2), (2/3, 1/2)], [(2/3, 1), (2/3, 0)]]),
  'I'=>(bb=[(0, 0), (1/3, 1)], vss=[[(0, 1), (1/3, 1)], [(1/6, 1), (1/6, 0)], [(0, 0), (1/3, 0)]]),
  'J'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1/6), (1/6, 0), (1/2, 0), (2/3, 1/6), (2/3, 1)]]),
  'K'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1)], [(2/3, 1), (1/6, 1/2), (0, 1/2)], [(1/6, 1/2), (2/3, 0)]]),
  'L'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1), (0, 0), (2/3, 0)]]),
  'M'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1), (1/3, 1/3), (2/3, 1), (2/3, 0)]]),
  'N'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1), (2/3, 0), (2/3, 1)]]),
  'O'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1), (2/3, 1), (2/3, 0), (0, 0)]]),
  'P'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1), (1/2, 1), (2/3, 5/6), (2/3, 2/3), (1/2, 1/2), (0, 1/2)]]),
  'Q'=>(bb=[(0, 0), (2/3, 1)], vss=[[(1/3, 1/3), (1/2, 1/6), (1/3, 0), (1/6, 0), (0, 1/6), (0, 5/6), (1/6, 1), (1/2, 1), (2/3, 5/6), (2/3, 1/3), (1/2, 1/6), (2/3, 0)]]),
  'R'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1), (1/2, 1), (2/3, 5/6), (2/3, 2/3), (1/2, 1/2), (0, 1/2)], [(1/6, 1/2), (2/3, 0)]]),
  'S'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1/6), (1/6, 0), (1/2, 0), (2/3, 1/6), (0, 5/6), (1/6, 1), (1/2, 1), (2/3, 5/6)]]),
  'T'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1), (2/3, 1)], [(1/3, 1), (1/3, 0)]]),
  'U'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1), (0, 1/6), (1/6, 0), (1/2, 0), (2/3, 1/6), (2/3, 1)]]),
  'V'=>(bb=[(0, 0), (1, 1)], vss=[[(0, 1), (1/2, 0), (1, 1)]]),
  'W'=>(bb=[(0, 0), (1, 1)], vss=[[(0, 1), (1/3, 0), (1/2, 1/2), (2/3, 0), (1, 1)]]),
  'X'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (2/3, 1)], [(0, 1), (2/3, 0)]]),
  'Y'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1), (1/3, 1/2), (1/3, 0)], [(1/3, 1/2), (2/3, 1)]]),
  'Z'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1), (2/3, 1), (0, 0), (2/3, 0)]]),
  '['=>(bb=[(0, 0), (1/3, 1)], vss=[[(0, 0), (0, 1), (1/3, 1)], [(1/3, 0), (0, 0)]]),
  '\\'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1), (2/3, 0)]]),
  ']'=>(bb=[(0, 0), (1/3, 1)], vss=[[(0, 1), (1/3, 1), (1/3, 0), (0, 0)]]),
  '^'=>(bb=[(0, 2/3), (2/3, 1)], vss=[[(0, 2/3), (1/3, 1), (2/3, 2/3)]]),
  '_'=>(bb=[(0, -1/6), (2/3, -1/6)], vss=[[(0, -1/6), (2/3, -1/6)]]),
  '`'=>(bb=[(0, 2/3), (1/6, 1)], vss=[[(0, 1), (1/6, 2/3)]]),
  'a'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(1/3, 0), (1/6, 0), (0, 1/6), (0, 1/2), (1/6, 2/3), (1/3, 2/3), (1/2, 1/2), (1/2, 1/6), (1/3, 0)], [(1/2, 1/6), (2/3, 0)]]),
  'b'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1)], [(0, 1/3), (1/3, 2/3), (1/2, 2/3), (2/3, 1/2), (2/3, 1/6), (1/2, 0), (1/3, 0), (0, 1/3)]]),
  'c'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(2/3, 2/3), (1/6, 2/3), (0, 1/2), (0, 1/6), (1/6, 0), (2/3, 0)]]),
  'd'=>(bb=[(0, 0), (2/3, 1)], vss=[[(2/3, 1/3), (1/3, 0), (1/6, 0), (0, 1/6), (0, 1/2), (1/6, 2/3), (1/3, 2/3), (2/3, 1/3)], [(2/3, 1), (2/3, 0)]]),
  'e'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(0, 1/3), (1/2, 1/3), (2/3, 1/2), (1/2, 2/3), (1/6, 2/3), (0, 1/2), (0, 1/6), (1/6, 0), (1/2, 0)]]),
  'f'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 1/2), (1/2, 1/2)], [(2/3, 5/6), (1/2, 1), (1/3, 1), (1/6, 5/6), (1/6, 0)]]),
  'g'=>(bb=[(0, -1/3), (2/3, 2/3)], vss=[[(0, -1/6), (1/6, -1/3), (1/2, -1/3), (2/3, -1/6), (2/3, 1/2), (1/2, 2/3), (1/6, 2/3), (0, 1/2), (0, 1/6), (1/6, 0), (2/3, 0)]]),
  'h'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1)], [(0, 1/3), (1/3, 2/3), (1/2, 2/3), (2/3, 1/2), (2/3, 0)]]),
  'i'=>(bb=[(0, 0), (0, 1)], vss=[[(0, 0), (0, 2/3)], [(0, 5/6), (0, 1)]]),
  'j'=>(bb=[(0, -1/3), (1/2, 1)], vss=[[(0, -1/6), (1/6, -1/3), (1/3, -1/3), (1/2, -1/6), (1/2, 2/3)], [(1/2, 5/6), (1/2, 1)]]),
  'k'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 0), (0, 1)], [(0, 1/3), (1/3, 1/3), (2/3, 2/3)], [(1/3, 1/3), (2/3, 0)]]),
  'l'=>(bb=[(0, 0), (1/6, 1)], vss=[[(0, 1), (0, 1/6), (1/6, 0)]]),
  'm'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(0, 0), (0, 2/3)], [(0, 1/2), (1/6, 2/3), (1/3, 1/2), (1/3, 1/3)], [(1/3, 1/2), (1/2, 2/3), (2/3, 1/2), (2/3, 0)]]),
  'n'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(0, 0), (0, 2/3)], [(0, 1/3), (1/3, 2/3), (1/2, 2/3), (2/3, 1/2), (2/3, 0)]]),
  'o'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(1/2, 0), (1/6, 0), (0, 1/6), (0, 1/2), (1/6, 2/3), (1/2, 2/3), (2/3, 1/2), (2/3, 1/6), (1/2, 0)]]),
  'p'=>(bb=[(0, -1/3), (2/3, 2/3)], vss=[[(0, -1/3), (0, 2/3)], [(0, 1/2), (1/6, 2/3), (1/2, 2/3), (2/3, 1/2), (2/3, 1/6), (1/2, 0), (0, 0)]]),
  'q'=>(bb=[(0, -1/3), (2/3, 2/3)], vss=[[(2/3, -1/3), (2/3, 2/3)], [(2/3, 1/2), (1/2, 2/3), (1/6, 2/3), (0, 1/2), (0, 1/6), (1/6, 0), (2/3, 0)]]),
  'r'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(0, 0), (0, 2/3)], [(0, 1/3), (1/3, 2/3), (1/2, 2/3), (2/3, 1/2)]]),
  's'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(0, 0), (1/2, 0), (2/3, 1/6), (1/2, 1/3), (1/6, 1/3), (0, 1/2), (1/6, 2/3), (2/3, 2/3)]]),
  't'=>(bb=[(0, 0), (2/3, 1)], vss=[[(0, 2/3), (2/3, 2/3)], [(1/3, 1), (1/3, 1/6), (1/2, 0), (2/3, 1/6)]]),
  'u'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(0, 2/3), (0, 1/6), (1/6, 0), (1/3, 0), (2/3, 1/3)], [(2/3, 2/3), (2/3, 0)]]),
  'v'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(0, 2/3), (1/3, 0), (2/3, 2/3)]]),
  'w'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(0, 2/3), (1/6, 0), (1/3, 2/3), (1/2, 0), (2/3, 2/3)]]),
  'x'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(0, 0), (2/3, 2/3)], [(0, 2/3), (2/3, 0)]]),
  'y'=>(bb=[(0, -1/3), (2/3, 2/3)], vss=[[(0, 2/3), (1/3, 0)], [(2/3, 2/3), (1/6, -1/3), (0, -1/3)]]),
  'z'=>(bb=[(0, 0), (2/3, 2/3)], vss=[[(0, 2/3), (2/3, 2/3), (0, 0), (2/3, 0)]]),
  '{'=>(bb=[(0, 0), (1/3, 1)], vss=[[(1/3, 1), (1/6, 5/6), (1/6, 2/3), (0, 1/2), (1/6, 1/3), (1/6, 1/6), (1/3, 0)]]),
  '|'=>(bb=[(0, 0), (0, 1)], vss=[[(0, 0), (0, 1)]]),
  '}'=>(bb=[(0, 0), (1/3, 1)], vss=[[(0, 0), (1/6, 1/6), (1/6, 1/3), (1/3, 1/2), (1/6, 2/3), (1/6, 5/6), (0, 1)]]),
  '~'=>(bb=[(0, 1/2), (2/3, 2/3)], vss=[[(0, 1/2), (1/6, 2/3), (1/2, 1/2), (2/3, 2/3)]])
)

b_text(b::Backend, str, p, size, mat) =
  let dx = 0,
	  inter_letter_spacing_factor = 1/3,
	  refs = []
    for c in str
  	  let glyph = letter_glyph[c]
  	    for vs in glyph.vss
  	  	  push!(refs, b_line(b, [add_xy(p, dx + v[1]*size, v[2]*size) for v in vs], mat))
  	    end
  	    dx += (glyph.bb[2][1]-glyph.bb[1][1] + inter_letter_spacing_factor)*size
  	  end
    end
	refs
  end

b_text_size(b::Backend, str, size, mat) =
  let dx = 0, minx = 0, maxx = 0, miny = 0, maxy = 0,
	  inter_letter_spacing_factor = 1/3
    for c in str
  	  let glyph = letter_glyph[c]
		minx = min(minx, dx + glyph.bb[1][1])
		maxx = max(maxx, dx + glyph.bb[2][1]-glyph.bb[1][1])
  	    dx += glyph.bb[2][1]-glyph.bb[1][1] + inter_letter_spacing_factor
		miny = min(miny, glyph.bb[1][2])
		maxy = max(maxy, glyph.bb[2][2])
  	  end
    end
	(minx, maxx, miny, maxy).*size
  end

##################################################################
# Materials
#=
In most cases, the material already exists in the backend and is
accessed by name.
=#
export b_get_material, b_new_material

b_get_material(b::Backend, mat) = mat

#=
It is also possible to algorithmically create materials by
specifying the material properties, such as color, roughness, etc.
Given that different kinds of materials need specialized treatment
(e.g., glass or metal), there are specific operations for these kinds.
=#

@bdef(b_new_material(path, color, specularity, roughness, transmissivity, transmitted_specular))

export b_all_shapes
b_all_shapes(b::Backend) = b.shapes

#=
Utilities for interactive development
=#

@bdef(b_all_refs())

b_delete_all_refs(b::Backend) =
  b_delete_refs(b, b_all_shapes(b))

b_delete_refs(b::Backend{K,T}, rs::Vector{T}) where {K,T} =
  for r in rs
	b_delete_ref(b, r)
  end

b_delete_ref(b::Backend{K,T}, r::T) where {K,T} =
  missing_specialization(b, :b_delete_ref, r)

# BIM
export b_slab, b_roof, b_beam, b_column, b_free_column, b_wall

#=
BIM operations require some extra support from the backends.
Given that humans prefer to live in horizontal surfaces, one interesting idea is
to separate 3D coordinates in two parts, one for the horizontal 2D coordinates
x and y, and another for the 1D vertical coordinate z, known as level.
=#

level_height(b::Backend, level) = level_height(level)

#=
Another relevant concept is the family. It contains generic information about
the construction element.
Given the wide variety of families (slabs, beams, windows, etc), it is not
possible to define a generic constructor or getter. However, for certain kinds
of families, it is possible to retrieve a set of materials that should be used
to represent the family. For a slab family, it should also be possible to
retrieve its thickness.
=#

#=
Horizontal BIM elements rely on the level
=#

material_refs(b::Backend, materials) =
  [material_ref(b, mat) for mat in materials]

b_slab(b::Backend, profile, level, family) =
  b_extrusion(
    b,
	profile,
    vz(slab_family_thickness(b, family)),
	z(level_height(b, level) + slab_family_elevation(b, family)),
    material_ref(b, family.bottom_material),
	material_ref(b, family.top_material),
	material_ref(b, family.side_material))

b_roof(b::Backend, region, level, family) =
  b_slab(b, region, level, family)

b_beam(b::Backend, c, h, angle, family) =
  let c = loc_from_o_phi(c, angle),
	  mat = material_ref(b, family.material)
    b_extrusion(
      b,
      family_profile(b, family),
      vz(h, c.cs),
	  c,
  	  mat, mat, mat)
  end

b_column(b::Backend, cb, angle, bottom_level, top_level, family) =
  let base_height = level_height(bottom_level),
      top_height = level_height(top_level)
    b_beam(b, add_z(loc_from_o_phi(cb, angle), base_height), top_height-base_height, 0, family)
  end

b_free_column(b::Backend, cb, h, angle, family) =
  b_beam(b, cb, h, angle, family)

# A poor's man approach to deal with Z-fighting
const support_z_fighting_factor = 0.999
const wall_z_fighting_factor = 0.998

b_wall(b::Backend, w_path, w_height, l_thickness, r_thickness, lmat, rmat, smat) =
  let w_paths = subpaths(w_path),
      r_w_paths = subpaths(offset(w_path, -r_thickness)),
      l_w_paths = subpaths(offset(w_path, l_thickness)),
      w_height = w_height*wall_z_fighting_factor,
      prevlength = 0,
      refs = []
    for (w_seg_path, r_w_path, l_w_path) in zip(w_paths, r_w_paths, l_w_paths)
      let currlength = prevlength + path_length(w_seg_path),
          c_r_w_path = closed_path_for_height(r_w_path, w_height),
          c_l_w_path = closed_path_for_height(l_w_path, w_height)
        append!(refs, b_pyramid_frustum(b, path_vertices(c_r_w_path), path_vertices(c_l_w_path), rmat, lmat, smat))
        prevlength = currlength
      end
    end
    [refs...]
  end

b_wall(b::Backend, w_path, w_height, l_thickness, r_thickness, family) =
  path_length(w_path) < path_tolerance() ?
    void_ref(b) :
    b_wall(b, w_path, w_height, l_thickness, r_thickness, family_materials(b, family)[1:3])

# Lights

@bdef b_ieslight(file::String, loc::Loc, dir::Vec, alpha::Real, beta::Real, gamma::Real)
@bdef b_pointlight(loc::Loc, color::RGB, range::Real, intensity::Real)
@bdef b_spotlight(loc::Loc, dir::Vec, hotspot::Real, falloff::Real)

# Trusses
export b_truss_node, b_truss_node_support, b_truss_bar

b_truss_node(b::Backend, p, family) =
  b_sphere(b, p, family.radius, material_ref(b, family.material))

b_truss_node_support(b::Backend, cb, family) =
  b_regular_pyramid(
    b, 4, add_z(cb, -3*family.radius),
    family.radius, 0, 3*family.radius, false,
    material_ref(b, family.material))

b_truss_bar(b::Backend, p, q, family) =
  let (c, h) = position_and_height(p, q)
    b_cylinder(
      b, c, family.radius, h,
      material_ref(b, family.material))
  end

# Analysis

@bdef b_truss_analysis(load::Vec, self_weight::Bool)
@bdef b_node_displacement_function(res::Any)

export b_truss_bars_volume
b_truss_bars_volume(b::Backend) =
  sum(truss_bar_volume, b.truss_bars)

#=
Operations that rely on a backend need to have a backend selected and will
generate an exception if there is none.
=#

struct UndefinedBackendException <: Exception end
showerror(io::IO, e::UndefinedBackendException) = print(io, "No current backend.")

# We can have several backends active at the same time
const Backends = Tuple{Vararg{Backend}}
const current_backends = Parameter{Backends}(())
has_current_backend() = !isempty(current_backends())

export add_current_backend
add_current_backend(b::Backend) =
  current_backends(tuple(b, current_backends()...))
# but for backward compatibility reasons, we might also select just one.
top_backend() =
  let bs = current_backends()
	isempty(bs) ?
  	  throw(UndefinedBackendException()) :
	  bs[1]
  end
# The current_backend function is just an alias for current_backends
current_backend() = current_backends()
current_backend(bs::Backends) = current_backends(bs)
# but it knows how to treat a single backend.
current_backend(b::Backend) = current_backends((b,))

backend(backend::Backend) =
  has_current_backend() ?
    switch_to_backend(top_backend(), backend) :
    current_backend(backend)
switch_to_backend(from::Backend, to::Backend) =
  current_backend(to)

# Variables with backend-specific values can be useful.
# Basically, they are dictionaries
# but they also support a default value for the case
# where there is no backend-specific value available
export BackendParameter
mutable struct BackendParameter
	default::Any
	value::IdDict{Backend, Any}
	BackendParameter(v::Any, ps::Pair{<:Backend}...) = new(v, IdDict{Backend, Any}(ps...))
	BackendParameter(p::BackendParameter) = new(p.default, copy(p.value))
	BackendParameter() = BackendParameter(nothing)
  	BackendParameter(ps::Pair{<:Backend}...) = BackendParameter(nothing, ps...)
end

(p::BackendParameter)(b::Backend=top_backend()) = get(p.value, b, p.default)
(p::BackendParameter)(b::Backend, newvalue) = p.value[b] = newvalue
(p::BackendParameter)(newvalue) = p.default = newvalue

Base.copy(p::BackendParameter) = BackendParameter(p)

export @backend
macro backend(b, expr)
  quote
	with(current_backends, ($(esc(b)),)) do
	  $(esc(expr))
    end
  end
end

export @backends
macro backends(b, expr)
  quote
	with(current_backends, $(esc(b))) do
	  $(esc(expr))
    end
  end
end

## THIS NEEDS TO BE SIMPLIFIED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Many functions default the backend to the top_backend and throw an error if there is none.
# We will simplify their definition with a macro:
# @defop delete_all_shapes()
# that expands into
# delete_all_shapes(backend::Backend=top_backend()) = throw(UndefinedBackendException())
# Note that according to Julia semantics the previous definition actually generates two different ones:
# delete_all_shapes() = delete_all_shapes(top_backend())
# delete_all_shapes(backend::Backend) = throw(UndefinedBackendException())
# Hopefully, backends will specialize the function for each specific backend

#macro defop(name_params)
#    name, params = name_params.args[1], name_params.args[2:end]
#    quote
#        export $(esc(name))
#        $(esc(name))($(map(esc,params)...), backend::Backend=top_backend()) =
#          throw(UndefinedBackendException())
#    end
#end
#=
@macroexpand @bdef(stroke(path::Path))
@macroexpand @bcall(Base.backend, stroke(Main.my_path))
@macroexpand @bscall(Base.backends, stroke(Main.my_path))
@macroexpand @cbscall(stroke(path))
=#

#=

Each shape can have a material and a classification.
A material might be just a color or something more complex
a classification might be a layer in some CAD tools or something else
Some backends will ignore this information, some use it in a dynamic
fashion. For performance reasons, operations that create shapes can
also be used without materials or classification.

=#

#@bdef add_door(w::Wall, loc::Loc, family::DoorFamily)
#@bdef add_window(w::Wall, loc::Loc, family::WindowFamily)
#@bdef bounding_box(shapes::Shapes)

#@bdef chair(c, angle, family)
export b_table
b_table(b::Backend, p, length, width, height, top_thickness, leg_thickness, mat) =
  let dx = length/2,
	  dy = width/2,
	  leg_x = dx - leg_thickness/2,
	  leg_y = dy - leg_thickness/2,
	  c = add_xy(p, -dx, -dy),
	  table_top = b_box(b, add_z(c, height - top_thickness), length, width, top_thickness, mat),
	  pts = add_xy.(add_xy.(p, [+leg_x, +leg_x, -leg_x, -leg_x], [-leg_y, +leg_y, +leg_y, -leg_y]), -leg_thickness/2, -leg_thickness/2),
	  legs = [b_box(b, pt, leg_thickness, leg_thickness, height - top_thickness, mat) for pt in pts]
	[table_top, legs]
  end

export b_chair
b_chair(b::Backend, p, length, width, height, seat_height, thickness, mat) =
  [b_table(b, p, length, width, seat_height, thickness, thickness, mat),
   b_box(b, add_xyz(p, -length/2, -width/2, seat_height), thickness, width, height - seat_height, mat)]

#=
To avoid an explosion of parameters, we expect functions as arguments, e.g.
b_table_and_chairs(b,
  loc_from_o_phi(p, α),
  p->b_table(b, p, t_length, t_width, t_height, t_top_thickness, t_leg_thickness, mat),
  p->b_chair(b, p, c_length, c_width, c_height, c_seat_height, c_thickness, mat),
  l_length,
  t_width,
  chairs_top,
  chairs_bottom,
  chairs_right,
  chairs_left,
  spacing)

As an example:
b_table_and_chairs(blender,
  loc_from_o_phi(xy(1,2), pi/3),
  p->b_table(blender, p, 3, 1, 1, 0.1, 0.1, -1),
  p->b_chair(blender, p, 0.7, 0.7, 1.5, 0.7, 0.05, -1),
  3, 1, 1, 1, 2, 2, 1)
=#
b_table_and_chairs(b::Backend, p, table::Function, chair::Function,
                   table_length, table_width,
                   chairs_on_top, chairs_on_bottom,
                   chairs_on_right, chairs_on_left,
                   spacing) =
  let dx = table_length/2,
      dy = table_width/2,
      row(p, angle, n) = [loc_from_o_phi(add_pol(p, i*spacing, angle), angle+pi/2) for i in 0:n-1],
      centered_row(p, angle, n) = row(add_pol(p, -spacing*(n-1)/2, angle), angle, n)
    [table(p),
     chair.(centered_row(add_x(p, -dx), -pi/2, chairs_on_bottom))...,
     chair.(centered_row(add_x(p, +dx), +pi/2, chairs_on_top))...,
     chair.(centered_row(add_y(p, +dy), -pi, chairs_on_right))...,
     chair.(centered_row(add_y(p, -dy), 0, chairs_on_left))...]
  end

#@bdef curtain_wall(s, path::Path, bottom::Real, height::Real, l_thickness::Real, r_thickness::Real, kind::Symbol)
backend_curtain_wall(b::Backend, s, path::Path, bottom::Real, height::Real, l_thickness::Real, r_thickness::Real, kind::Symbol) =
  let family = getproperty(s.family, kind),
      mat = material_ref(b, family.material)
    with_family_in_layer(b, family) do
      b_wall(b, translate(path, vz(bottom)), height, l_thickness, r_thickness, mat, mat, mat)
    end
  end

@bdef curtain_wall(s, path::Path, bottom::Real, height::Real, thickness::Real, kind::Symbol)

backend_fill(b, path) =
  backend_fill_curves(b, backend_stroke(b, path))

backend_frame_at(b, c, t) = throw(UndefinedBackendException())
backend_fill_curves(b, ids) = throw(UndefinedBackendException())

#@bdef fill(m::Mesh)
backend_fill(b::Backend, m::Mesh) =
  backend_surface_mesh(b, m.vertices, m.faces)

#@bdef fill(path::ClosedPathSequence)
backend_fill(b::Backend, path::ClosedPathSequence) =
  backend_fill_curves(b, map(path->backend_stroke(b, path), path.paths))

@bdef ground(level::Loc, color::RGB)
#@bdef loft_curve_point(profile::Shape, point::Shape)
#@bdef loft_points(profiles::Shapes, rails::Shapes, ruled::Bool, closed::Bool)
#@bdef loft_surface_point(profile::Shape, point::Shape)
#@bdef loft_surfaces(profiles::Shapes, rails::Shapes, ruled::Bool, closed::Bool)
#@bdef map_division(f::Function, s::Shape1D, n::Int)
#@bdef map_division(f::Function, s::Shape2D, nu::Int, nv::Int)
#@bdef map_division(f::Function, s::SurfaceGrid, nu::Int, nv::Int)
@bdef name()
#@bdef panel(bot::Locs, top::Locs, family::PanelFamily)

# We assume there are properties to store the view details
export b_get_view, b_set_view, b_set_view_top
b_set_view(b::Backend, camera, target, lens, aperture) =
  begin
    b.camera = camera
    b.target = target
    b.lens = lens
  end
b_set_view_top(b::Backend) =
  begin
    b.camera = u0()
    b.target = u0()
    b.lens = 0
  end
b_get_view(b::Backend) =
  b.camera, b.target, b.lens

@bdef b_zoom_extents()

export b_realistic_sky
b_realistic_sky(b::Backend, date, latitude, longitude, elevation, meridian, turbidity, withsun) =
  let (altitude, azimuth) = sun_pos(date, meridian, latitude, longitude)
    b_realistic_sky(b, altitude, azimuth, turbidity, withsun)
  end
@bdef b_realistic_sky(altitude, azimuth, turbidity, withsun)
@bdef b_set_sun(altitude, azimuth)
export b_set_ground
b_set_ground(b::Backend, level, mat) =
  b_surface_regular_polygon(b, 16, z(level), 10000, 0, true, mat)

@bdef b_render_view(path)

#@bdef revolve_curve(profile::Shape, p::Loc, n::Vec, start_angle::Real, amplitude::Real)
#@bdef revolve_point(profile::Shape, p::Loc, n::Vec, start_angle::Real, amplitude::Real)
#@bdef revolve_surface(profile::Shape, p::Loc, n::Vec, start_angle::Real, amplitude::Real)
@bdef right_cuboid(cb, width, height, h, angle, material)

backend_stroke(b::Backend, path::Union{OpenPathSequence,ClosedPathSequence}) =
  backend_stroke_unite(b, map(path->backend_stroke(b, path), path.paths))
backend_stroke(b::Backend, path::PathOps) =
  begin
      start, curr, refs = path.start, path.start, []
      for op in path.ops
          start, curr, refs = backend_stroke_op(b, op, start, curr, refs)
      end
      if path.closed
          push!(refs, backend_stroke_line(b, [curr, start]))
      end
      backend_stroke_unite(b, refs)
  end

backend_stroke(b::Backend, path::PathSet) =
    for p in path.paths
        backend_stroke(b, p)
    end

@bdef stroke_op(op::LineXThenYOp, start::Loc, curr::Loc, refs)
@bdef stroke_op(op::LineYThenXOp, start::Loc, curr::Loc, refs)
#@bdef stroke_op(op::MoveOp, start::Loc, curr::Loc, refs)
#@bdef stroke_op(op::MoveToOp, start::Loc, curr::Loc, refs)

#@bdef stroke_op(op::LineOp, start::Loc, curr::Loc, refs)
backend_stroke_op(b::Backend, op::LineOp, start::Loc, curr::Loc, refs) =
  (start, curr + op.vec, push!(refs, backend_stroke_line(b, [curr, curr + op.vec])))

#backend_stroke_op(b::Backend, op::CloseOp, start::Loc, curr::Loc, refs) =
#    (start, start, push!(refs, backend_stroke_line(b, [curr, start])))

#@bdef stroke_op(op::ArcOp, start::Loc, curr::Loc, refs)
backend_stroke_op(b::Backend, op::ArcOp, start::Loc, curr::Loc, refs) =
  let center = curr - vpol(op.radius, op.start_angle)
    (start, center + vpol(op.radius, op.start_angle + op.amplitude),
     push!(refs, backend_stroke_arc(b, center, op.radius, op.start_angle, op.amplitude)))
  end


#

@bdef stroke_unite(refs)
#@bdef surface_boundary(s::Shape2D)
#@bdef surface_domain(s::Shape2D)
@bdef surface_mesh(vertices, faces)

#@bdef wall(path, height, l_thickness, r_thickness, family)

# A poor's man approach to deal with Z-fighting
const support_z_fighting_factor = 0.999
const wall_z_fighting_factor = 0.998

backend_wall(b::Backend, w_path, w_height, l_thickness, r_thickness, family) =
  path_length(w_path) < path_tolerance() ?
    realize(b, empty_shape()) : # not beautiful
    let (matright, matleft) = wall_materials(b, family_ref(b, family))
      backend_wall_with_materials(b, w_path, w_height, l_thickness, r_thickness, matright, matleft)
    end

@bdef wall_path(path::OpenPolygonalPath, height, l_thickness, r_thickness)
@bdef wall_path(path::Path, height, l_thickness, r_thickness)

#@bdef wall_with_materials(w_path, w_height, l_thickness, r_thickness, matright, matleft)
backend_wall_with_materials(b::Backend, w_path, w_height, l_thickness, r_thickness, matright, matleft) =
  let w_paths = subpaths(w_path),
      r_w_paths = subpaths(offset(w_path, -r_thickness)),
      l_w_paths = subpaths(offset(w_path, l_thickness)),
      w_height = w_height*wall_z_fighting_factor,
      prevlength = 0,
      refs = []
    for (w_seg_path, r_w_path, l_w_path) in zip(w_paths, r_w_paths, l_w_paths)
      let currlength = prevlength + path_length(w_seg_path),
          c_r_w_path = closed_path_for_height(r_w_path, w_height),
          c_l_w_path = closed_path_for_height(l_w_path, w_height)
        push!(refs, realize_pyramid_frustum(b, matright, matleft, matleft, c_l_w_path, c_r_w_path))
        prevlength = currlength
      end
    end
    [refs...]
  end

# Select things from a backend
@bdef(b_select_position(prompt))
@bdef(b_select_positions(prompt))
@bdef(b_select_point(prompt))
@bdef(b_select_points(prompt))
@bdef(b_select_curve(prompt))
@bdef(b_select_curves(prompt))
@bdef(b_select_surface(prompt))
@bdef(b_select_surfaces(prompt))
@bdef(b_select_solid(prompt))
@bdef(b_select_solids(prompt))
@bdef(b_select_shape(prompt))
@bdef(b_select_shapes(prompt))

#=
Backends might use different communication mechanisms, e.g., sockets, COM,
RMI, etc. We start by defining socket-based communication.
=#

struct SocketBackend{K,T} <: Backend{K,T}
  connection::LazyParameter{TCPSocket}
  remote::NamedTuple
end

export connect_to, start_and_connect_to

connect_to(backend::AbstractString, port::Integer; attempts=10, wait=8) =
  for i in 1:attempts
    try
      return connect(port)
    catch e
      if i == attempts
        @info("Couldn't connect with $(backend).")
        throw(e)
      else
        @info("Please, start/restart $(backend).")
        sleep(wait)
      end
    end
  end
#
start_and_connect_to(backend, start, port; attempts=20, wait=5) =
  for i in 1:attempts
    try
      return connect(port)
    catch e
      if i == 1
        @info("Starting $(backend).")
   	    start()
        sleep(wait)
      elseif i == attempts
        @info("Couldn't connect with $(backend).")
        throw(e)
      else
        sleep(wait)
      end
    end
  end



# To simplify remote calls
macro remote(b, call)
  let op = call.args[1],
      args = map(esc, call.args[2:end]),
      b = esc(b)
    :(call_remote(getfield(getfield($(b), :remote), $(QuoteNode(op))), $(b).connection(), $(args...)))
  end
end

macro get_remote(b, op)
  let b = esc(b)
    :(getfield(getfield($(b), :remote), $(QuoteNode(op))))
  end
end

# A common solution is to use socket-based communication
SocketBackend{K,T}(c::LazyParameter{TCPSocket}) where {K,T} =
  SocketBackend{K,T}(c, NamedTuple{}())

connection(b::SocketBackend) = b.connection()

reset_backend(b::SocketBackend) =
  begin
    for f in b.remote
      reset_opcode(f)
    end
    close(b.connection())
    reset(b.connection)
  end

#One less dynamic option is to use a file-based backend. To that end, we implement
#the IOBuffer_Backend

Base.@kwdef mutable struct IOBufferBackend{K,T} <: Backend{K,T}
  out::IOBuffer=IOBuffer()
  camera::Loc=xyz(10,10,10)
  target::Loc=xyz(0,0,0)
  lens::Real=35
end
connection(backend::IOBufferBackend) = backend.out

################################################################
# Not all backends support all stuff. Some of it might need to be supported
# by ourselves. Layer are one example.

# Layers
export AbstractLayer, BasicLayer
abstract type AbstractLayer end
struct BasicLayer <: AbstractLayer
  name::String
  active::Bool
  color::RGB
end

export b_layer, b_current_layer,
       b_all_shapes_in_layer, b_delete_all_shapes_in_layer

# Default implementation assumes that backends have properties for current_layer and layers (a dict)
b_layer(b::Backend, name, active, color) =
  BasicLayer(name, active, color)
b_current_layer(b::Backend) = b.current_layer
b_current_layer(b::Backend, layer) = b.current_layer = layer
b_all_shapes_in_layer(b::Backend, layer) = b.layers[layer]
b_delete_all_shapes_in_layer(b::Backend, layer) = b_delete_shapes(b_all_shapes_in_layer(b, layer))
