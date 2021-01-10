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
show(io::IO, b::Backend{K,T}) where {K,T} = print(io, backend_name(b))

# Backends need to implement operations or an exception is triggered
struct UnimplementedBackendOperationException <: Exception
	backend
	operation
  args
end
showerror(io::IO, e::UnimplementedBackendOperationException) =
	print(io, "Operation $(e.operation) is not available in backend $(e.backend).$(length(e.args) > 0 ? "Args: $(e.args)" : "")")

missing_specialization(b::Backend{K,T}, oper=:unknown_operation, args...) where {K,T} =
  error(UnimplementedBackendOperationException(b, oper, args))

macro bdef(call)
  name, escname, params = call.args[1], esc(call.args[1]), esc.(call.args[2:end])
  quote
    export $(escname)
    $(escname)(b::Backend{K,T}, $(params...)) where {K,T} =
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

b_polygon(b::Backend{K,T}, ps, mat) where {K,T} =
  b_line(b, [ps..., p[1]], mat)

b_regular_polygon(b::Backend{K,T}, edges, c, r, angle, inscribed, mat) where {K,T} =
  b_polygon(b, regular_polygon_vertices(edges, c, r, angle, inscribed), mat)

b_nurbs_curve(b::Backend{K,T}, order, ps, knots, weights, closed, mat) where {K,T} =
  b_line(b, ps, closed, mat)

b_spline(b::Backend{K,T}, ps, v1, v2, interpolator, mat) where {K,T} =
  let ci = curve_interpolator(ps, false),
      cpts = curve_control_points(ci),
      n = length(cpts),
      knots = curve_knots(ci)
    b_nurbs_curve(b, 5, cpts, knots, fill(1.0, n), false, mat)
  end

b_closed_spline(b::Backend{K,T}, ps, mat) where {K,T} =
  let ci = curve_interpolator(ps, true),
      cpts = curve_control_points(ci),
      n = length(cpts),
      knots = curve_knots(ci)
    b_nurbs_curve(b, 5, cpts, knots, fill(1.0, n), true, mat)
  end

b_circle(b::Backend{K,T}, c, r, mat) where {K,T} =
  b_closed_spline(b, regular_polygon_vertices(32, c, r, 0, true), mat)

b_arc(b::Backend{K,T}, c, r, α, Δα, mat) where {K,T} =
  b_spline(b,
    [center + vpol(r, a, center.cs)
     for a in division(α, α + Δα, Δα*32/2/π, false)],
    nothing, nothing, # THIS NEEDS TO BE FIXED
    mat)

b_rectangle(b::Backend{K,T}, c, dx, dy, mat) where {K,T} =
  b_polygon(b, [c, add_x(c, dx), add_xy(c, dx, dy), add_y(c, dy)], mat)

# First tier: everything is a triangle or a set of triangles
export b_trig, b_quad, b_ngon, b_quad_strip, b_quad_strip_closed

@bdef(b_trig(p1, p2, p3))

# By default, we silently drop the mat
b_trig(b::Backend{K,T}, p1, p2, p3, mat) where {K,T} =
  b_trig(b, p1, p2, p3)

b_quad(b::Backend{K,T}, p1, p2, p3, p4, mat) where {K,T} =
  [b_trig(b, p1, p2, p3, mat),
   b_trig(b, p1, p3, p4, mat)]

b_ngon(b::Backend{K,T}, ps, pivot, smooth, mat) where {K,T} =
  [(b_trig(b, pivot, ps[i], ps[i+1], mat)
    for i in 1:size(ps,1)-1)...,
	 b_trig(b, pivot, ps[end], ps[1], mat)]

b_quad_strip(b::Backend{K,T}, ps, qs, smooth, mat) where {K,T} =
  [b_quad(b, ps[i], ps[i+1], qs[i+1], qs[i], mat)
   for i in 1:size(ps,1)-1]

b_quad_strip_closed(b::Backend{K,T}, ps, qs, smooth, mat) where {K,T} =
  b_quad_strip(b, [ps..., ps[1]], [qs..., qs[1]], smooth, mat)

############################################################
# Second tier: surfaces
export b_surface_polygon, b_surface_regular_polygon,
			 b_surface_circle, b_surface_arc, b_surface

b_surface_polygon(b::Backend{K,T}, ps, mat) where {K,T} =
  # This only works for convex polygons
  b_ngon(b, ps, trig_center(ps[1], ps[2], ps[3]), false, mat)

@bdef(b_surface_polygon_with_holes(ps, qss, mat))

b_surface_rectangle(b::Backend{K,T}, c, dx, dy, mat) where {K,T} =
  b_quad(b, c, add_x(c, dx), add_xy(c, dx, dy), add_y(c, dy), mat)

b_surface_regular_polygon(b::Backend{K,T}, edges, c, r, angle, inscribed, mat) where {K,T} =
  b_ngon(b, regular_polygon_vertices(edges, c, r, angle, inscribed), c, false, mat)

b_surface_circle(b::Backend{K,T}, c, r, mat) where {K,T} =
  b_surface_regular_polygon(b, 32, c, r, 0, true, mat)

b_surface_arc(b::Backend{K,T}, c, r, α, Δα, mat) where {K,T} =
  b_ngon(b,
         [center + vpol(r, a, center.cs)
          for a in division(α, α + Δα, Δα*32/2/π, false)],
         c, false, mat)

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
			 b_cone

# Each solid can have just one material or multiple materials
b_generic_pyramid_frustum(b::Backend{K,T}, bs, ts, smooth, bmat, tmat, smat) where {K,T} =
  [b_surface_polygon(b, reverse(bs), bmat),
   b_quad_strip_closed(b, bs, ts, smooth, smat),
   b_surface_polygon(b, ts, tmat)]

b_generic_pyramid_frustum_with_holes(b::Backend{K,T}, bs, ts, smooth, bbs, tts, smooths, bmat, tmat, smat) where {K,T} =
  [b_surface_polygon_with_holes(b, reverse(bs), bbs, bmat),
   b_quad_strip_closed(b, bs, ts, smooth, smat),
   [b_quad_strip_closed(b, bs, ts, smooth, smat)
    for (bs, ts, smooth) in zip(bbs, tts, smooths)]...,
   b_surface_polygon_with_holes(b, ts, reverse.(tts), tmat)]

b_generic_pyramid(b::Backend{K,T}, bs, t, smooth, bmat, smat) where {K,T} =
	[b_surface_polygon(b, reverse(bs), bmat),
	 b_ngon(b, bs, t, smooth, smat)]

b_generic_prism(b::Backend{K,T}, bs, smooth, v, bmat, tmat, smat) where {K,T} =
  b_generic_pyramid_frustum(b, bs, translate(bs, v), smooth, bmat, tmat, smat)

b_generic_prism_with_holes(b::Backend{K,T}, bs, smooth, bss, smooths, v, bmat, tmat, smat) where {K,T} =
  b_generic_pyramid_frustum_with_holes(b, bs, translate(bs, v), smooth, bss, translate.(bss, v), smooths, bmat, tmat, smat)

b_pyramid_frustum(b::Backend{K,T}, bs, ts, mat) where {K,T} =
  b_pyramid_frustum(b, bs, ts, mat, mat, mat)

b_pyramid_frustum(b::Backend{K,T}, bs, ts, bmat, tmat, smat) where {K,T} =
  b_generic_pyramid_frustum(b, bs, ts, false, bmat, tmat, smat)

b_pyramid(b::Backend{K,T}, bs, t, mat) where {K,T} =
	b_pyramid(b, bs, t, mat, mat)
b_pyramid(b::Backend{K,T}, bs, t, bmat, smat) where {K,T} =
  b_generic_pyramid(b, bs, t, false, bmat, smat)

b_prism(b::Backend{K,T}, bs, v, mat) where {K,T} =
	b_prism(b, bs, v, mat, mat, mat)
b_prism(b::Backend{K,T}, bs, v, bmat, tmat, smat) where {K,T} =
  b_pyramid_frustum(b, bs, translate(bs, v), bmat, tmat, smat)

b_regular_pyramid_frustum(b::Backend{K,T}, edges, cb, rb, angle, h, rt, inscribed, mat) where {K,T} =
	b_regular_pyramid_frustum(b, edges, cb, rb, angle, h, rt, inscribed, mat, mat, mat)
b_regular_pyramid_frustum(b::Backend{K,T}, edges, cb, rb, angle, h, rt, inscribed, bmat, tmat, smat) where {K,T} =
  b_pyramid_frustum(
    b,
    regular_polygon_vertices(edges, cb, rb, angle, inscribed),
    regular_polygon_vertices(edges, add_z(cb, h), rt, angle, inscribed),
    bmat, tmat, smat)

b_regular_pyramid(b::Backend{K,T}, edges, cb, rb, angle, h, inscribed, mat) where {K,T} =
	b_regular_pyramid(b, edges, cb, rb, angle, h, inscribed, mat, mat)
b_regular_pyramid(b::Backend{K,T}, edges, cb, rb, angle, h, inscribed, bmat, smat) where {K,T} =
  b_pyramid(
  	b,
  	regular_polygon_vertices(edges, cb, rb, angle, inscribed),
  	add_z(cb, h),
  	bmat, smat)

b_regular_prism(b::Backend{K,T}, edges, cb, rb, angle, h, inscribed, mat) where {K,T} =
	b_regular_prism(b, edges, cb, rb, angle, h, inscribed, mat, mat, mat)
b_regular_prism(b::Backend{K,T}, edges, cb, rb, angle, h, inscribed, bmat, tmat, smat) where {K,T} =
	b_regular_pyramid_frustum(b, edges, cb, rb, angle, h, rt, inscribed, bmat, tmat, smat)

b_cylinder(b::Backend{K,T}, cb, r, h, mat) where {K,T} =
	b_cylinder(b, cb, r, h, mat, mat, mat)
b_cylinder(b::Backend{K,T}, cb, r, h, bmat, tmat, smat) where {K,T} =
  b_generic_prism(
  	b,
  	regular_polygon_vertices(32, cb, r, 0, true),
  	true,
    vz(h, cb.cs),
  	bmat, tmat, smat)

b_cuboid(b::Backend{K,T}, pb0, pb1, pb2, pb3, pt0, pt1, pt2, pt3, mat) where {K,T} =
  [b_quad(b, pb3, pb2, pb1, pb0, mat),
   b_quad_strip_closed(b, [pb0, pb1, pb2, pb3], [pt0, pt1, pt2, pt3], false, mat),
   b_quad(b, pt0, pt1, pt2, pt3, mat)]

b_box(b::Backend{K,T}, c, dx, dy, dz, mat) where {K,T} =
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

b_sphere(b::Backend{K,T}, c, r, mat) where {K,T} =
  let ϕs = division(0, 2π, 32, false)
    [b_ngon(b, [add_sph(c, r, ϕ, π/16) for ϕ in ϕs], add_sph(c, r, 0, 0), true, mat),
  	 [b_quad_strip_closed(b,
  			[add_sph(c, r, ϕ, ψ+π/16) for ϕ in ϕs],
  			[add_sph(c, r, ϕ, ψ) for ϕ in ϕs],
  			true, mat) for ψ in π/16:π/16:π-π/16]...,
  	 b_ngon(b, [add_sph(c, r, ϕ, π-π/16) for ϕ in ϕs], add_sph(c, r, 0, π), true, mat)]
	end

b_cone(b::Backend{K,T}, cb, r, h, mat) where {K,T} =
	b_cone(b, cb, r, h, mat, mat)

b_cone(b::Backend{K,T}, cb, r, h, bmat, smat) where {K,T} =
	b_generic_pyramid(
		b,
		regular_polygon_vertices(32, cb, r, 0, true),
		add_z(cb, h),
		true,
		bmat, smat)

b_cone_frustum(b::Backend{K,T}, cb, rb, h, rt, mat) where {K,T} =
	b_cone_frustum(b, cb, rb, h, rt, mat, mat, mat)

b_cone_frustum(b::Backend{K,T}, cb, rb, h, rt, bmat, tmat, smat) where {K,T} =
  b_generic_pyramid_frustum(
  	b,
  	regular_polygon_vertices(32, cb, rb, 0, true),
  	regular_polygon_vertices(32, add_z(cb, h), rt, 0, true),
	true,
  	bmat, tmat, smat)

##################################################################
# Paths and Regions
b_surface(b::Backend{K,T}, path::ClosedPath, mat) where {K,T} =
  b_surface_polygon(b, path_vertices(path), mat)

b_surface(b::Backend{K,T}, region::Region, mat) where {K,T} =
  b_surface_polygon_with_holes(
    b,
    path_vertices(outer_path(region)),
    [reverse(path_vertices(path)) for path in inner_paths(region)],
    mat)

# In theory, this should be implemented using a loft
b_path_frustum(b::Backend{K,T}, bpath, tpath, bmat, tmat, smat) where {K,T} =
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
b_extrude_profile(b::Backend{K,T}, cb, h, profile, mat) where {K,T} =
  b_extrude_profile(b, cb, h, profile, mat, mat, mat)

b_extrude_profile(b::Backend{K,T}, cb, h, profile, bmat, tmat, smat) where {K,T} =
  let path = profile
  	b_generic_prism(
  	  b,
  	  path_vertices_on(path, cb),
  	  is_smooth_path(path),
      vz(h, cb.cs),
  	  bmat, tmat, smat)
  end

b_extrude_profile(b::Backend{K,T}, cb, h, profile::CircularPath, bmat, tmat, smat) where {K,T} =
  b_cylinder(b, add_xy(cb, profile.center.x, profile.center.y), profile.radius, h, bmat, tmat, smat)

b_extrude_profile(b::Backend{K,T}, cb, h, profile::Region, bmat, tmat, smat) where {K,T} =
  let outer = outer_path(profile),
      inners = inner_paths(profile)
    isempty(inners) ?
      b_generic_prism(b,
        path_vertices_on(outer, cb),
        is_smooth_path(outer),
        vz(h, cb.cs),
        bmat, tmat, smat) :
      b_generic_prism_with_holes(b,
        path_vertices_on(outer, cb),
        is_smooth_path(outer),
        path_vertices_on.(inners, cb),
        is_smooth_path.(inners),
        vz(h, cb.cs),
        bmat, tmat, smat)
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

b_stroke(b::Backend{K,T}, path::CircularPath, mat) where {K,T} =
  b_circle(b, path.center, path.radius, mat)
b_stroke(b::Backend{K,T}, path::RectangularPath, mat) where {K,T} =
  b_rectangle(b, path.corner, path.dx, path.dy, mat)
b_stroke(b::Backend{K,T}, path::ArcPath, mat) where {K,T} =
  b_arc(b, path.center, path.radius, path.start_angle, path.amplitude, mat)
b_stroke(b::Backend{K,T}, path::OpenPolygonalPath, mat) where {K,T} =
  b_line(b, path.vertices, mat)
b_stroke(b::Backend{K,T}, path::ClosedPolygonalPath, mat) where {K,T} =
  b_polygon(b, path.vertices, mat)
b_stroke(b::Backend{K,T}, path::OpenSplinePath, mat) where {K,T} =
  b_spline(b, path.vertices, path.v0, path.v1, path.interpolator, mat)
b_stroke(b::Backend{K,T}, path::ClosedSplinePath, mat) where {K,T} =
  b_closed_spline(b, path.vertices, mat)
b_stroke(b::Backend{K,T}, path::Region, mat) where {K,T} =
  [b_stroke(b, path, mat) for path in path.paths]

b_fill(b::Backend{K,T}, path::CircularPath, mat) where {K,T} =
  b_surface_circle(b, path.center, path.radius, mat)
b_fill(b::Backend{K,T}, path::RectangularPath, mat) where {K,T} =
  b_surface_rectangle(b, path.corner, path.dx, path.dy, mat)
b_fill(b::Backend{K,T}, path::ClosedPolygonalPath, mat) where {K,T} =
  b_surface_polygon(b, path.vertices, mat)
b_fill(b::Backend{K,T}, path::ClosedSplinePath, mat) where {K,T} =
  b_surface_closed_spline(b, path.vertices, mat)
b_fill(b::Backend{K,T}, path::Region, mat) where {K,T} =
  b_surface(b, path, mat)

#=
backend_fill(b::Backend{K,T}, path::ClosedSplinePath) where {K,T} =
  backend_fill_curves(b, @remote(b, ClosedSpline(path.vertices)))

b_stroke_unite(b::Backend{K,T}, refs) where {K,T} = @remote(b, JoinCurves(refs))

backend_fill(b::Backend{K,T}, path::ClosedPolygonalPath) where {K,T} =
    @remote(b, SurfaceClosedPolyLine(path.vertices))
    backend_fill(b::Backend{K,T}, path::RectangularPath) where {K,T} =
        let c = path.corner,
            dx = path.dx,
            dy = path.dy
            @remote(b, SurfaceClosedPolyLine([c, add_x(c, dx), add_xy(c, dx, dy), add_y(c, dy)]))
        end
backend_fill(b::Backend{K,T}, path::RectangularPath) where {K,T} =
    let c = path.corner,
        dx = path.dx,
        dy = path.dy
        SurfaceClosedPolyLine(connection(b), [c, add_x(c, dx), add_xy(c, dx, dy), add_y(c, dy)])
    end

backend_fill_curves(b::Backend{K,T}, gs::Guids) where {K,T} = @remote(b, SurfaceFrom(gs))
backend_fill_curves(b::Backend{K,T}, g::Guid) where {K,T} = @remote(b, SurfaceFrom([g]))

b_stroke_line(b::Backend{K,T}, vs) where {K,T} = @remote(b, PolyLine(vs))

b_stroke_arc(b::Backend{K,T}, center::Loc, radius::Real, start_angle::Real, amplitude::Real) where {K,T} =
    let end_angle = start_angle + amplitude
        if end_angle > start_angle
            @remote(b, Arc(center, vx(1, center.cs), vy(1, center.cs), radius, start_angle, end_angle))
        else
            @remote(b, Arc(center, vx(1, center.cs), vy(1, center.cs), radius, end_angle, start_angle))
        end
    end
=#
##################################################################
# Materials
#=
In most cases, the material already exists in the backend and is
accessed by name.
=#
export b_get_material, b_new_material

@bdef(b_get_material(path))

# Default for backends that do not have materials
b_get_material(b::Backend{K,T}, mat::Nothing) where {K,T} = zero(T)

#=
It is also possible to algorithmically create materials by
specifying the material properties, such as color, roughness, etc.
Given that different kinds of materials need specialized treatment
(e.g., glass or metal), there are specific operations for these kinds.
=#

@bdef(b_new_material(path, color, specularity, roughness, transmissivity, transmitted_specular))

#=
Utilities for interactive development
=#

export b_all_refs, b_delete_all_refs, b_delete_refs, b_delete_ref

@bdef(b_all_refs())

b_delete_all_refs(b::Backend{K,T}) where {K,T} =
  b_delete_refs(b, b_all_shapes(b))

b_delete_refs(b::Backend{K,T}, rs::Vector{T}) where {K,T} =
  for r in rs
		b_delete_ref(b, r)
	end

b_delete_ref(b::Backend{K,T}, r::T) where {K,T} =
  missing_specialization(b, :b_delete_ref, r)

#=
BIM operations require some extra support from the backends.
Given that humans prefer to live in horizontal surfaces, one interesting idea is
to separate 3D coordinates in two parts, one for the horizontal 2D coordinates
x and y, and another for the 1D vertical coordinate z, known as level.
=#

level_height(b::Backend{K,T}, level) where {K,T} = level_height(level)

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

material_refs(b::Backend{K,T}, materials) where {K,T} =
  [material_ref(b, mat) for mat in materials]

b_slab(b::Backend{K,T}, profile, level, family) where {K,T} =
  b_extrude_profile(
    b,
    z(level_height(b, level) + slab_family_elevation(b, family)),
    slab_family_thickness(b, family),
    profile,
    material_refs(b, family_materials(b, family))[1:3]...)

b_beam(b::Backend{K,T}, c, h, family) where {K,T} =
  b_extrude_profile(
    b,
    c,
    h,
    family_profile(b, family),
	material_refs(b, family_materials(b, family))[1:3]...)

# A poor's man approach to deal with Z-fighting
const support_z_fighting_factor = 0.999
const wall_z_fighting_factor = 0.998

b_wall(b::Backend{K,T}, w_path, w_height, l_thickness, r_thickness, lmat, rmat, smat) where {K,T} =
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

b_wall(b::Backend{K,T}, w_path, w_height, l_thickness, r_thickness, family) where {K,T} =
  path_length(w_path) < path_tolerance() ?
    void_ref(b) :
    b_wall(b, w_path, w_height, l_thickness, r_thickness, family_materials(b, family)[1:3])

# Lights

@bdef ieslight(file::String, loc::Loc, dir::Vec, alpha::Real, beta::Real, gamma::Real)
@bdef pointlight(loc::Loc, color::RGB, range::Real, intensity::Real)
@bdef spotlight(loc::Loc, dir::Vec, hotspot::Real, falloff::Real)

# Trusses

b_truss_node(b::Backend{K,T}, p, family) where {K,T} =
  b_sphere(b, p, family.radius, material_ref(b, family_materials(b, family)[1]))

b_truss_node_support(b::Backend{K,T}, cb, family) where {K,T} =
  b_regular_pyramid(
    b, 4, add_z(cb, -3*family.radius),
    family.radius, 0, 3*family.radius, false,
    material_ref(b, family_materials(b, family)[1]))

b_truss_bar(b::Backend{K,T}, p, q, family) where {K,T} =
  let (c, h) = position_and_height(p, q)
    b_cylinder(
      b, c, family.radius, h,
      [material_ref(b, m) for m in family_materials(b, family)]...)
  end

# Layers

@bdef b_current_layer()
@bdef b_current_layer(layer)
@bdef b_create_layer(name::String, active::Bool, color::RGB)

# Analysis

@bdef b_truss_analysis(load::Vec, self_weight::Bool)
@bdef b_node_displacement_function(res::Any)

export b_truss_bars_volume
b_truss_bars_volume(b::Backend{K,T}) where {K,T} =
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

export add_current_backend
add_current_backend(b::Backend) =
  current_backends(tuple(b, current_backends()...))
# but for backward compatibility reasons, we might also select just one.
current_backend() =
	let bs = current_backends()
		isempty(bs) ?
			throw(UndefinedBackendException()) :
			bs[1]
	end
current_backend(b::Backend) = current_backends((b,))
has_current_backend() = !isempty(current_backends())

# Variables with backend-specific values can be useful.
# Basically, they are dictionaries

struct BackendParameter
	value::IdDict{Backend, Any}
	BackendParameter() = new(IdDict{Backend, Any}())
	BackendParameter(p::BackendParameter) = new(copy(p.value))
  	BackendParameter(ps::Pair{<:Backend}...) = new(IdDict{Backend, Any}(ps...))
end

(p::BackendParameter)(b::Backend=current_backend()) = get(p.value, b, nothing)
(p::BackendParameter)(b::Backend, newvalue) = p.value[b] = newvalue

Base.copy(p::BackendParameter) = BackendParameter(p)

export @backend
macro backend(b, expr)
  quote
	with(current_backend, $(esc(b))) do
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

# Many functions default the backend to the current_backend and throw an error if there is none.
# We will simplify their definition with a macro:
# @defop delete_all_shapes()
# that expands into
# delete_all_shapes(backend::Backend=current_backend()) = throw(UndefinedBackendException())
# Note that according to Julia semantics the previous definition actually generates two different ones:
# delete_all_shapes() = delete_all_shapes(current_backend())
# delete_all_shapes(backend::Backend) = throw(UndefinedBackendException())
# Hopefully, backends will specialize the function for each specific backend

#macro defop(name_params)
#    name, params = name_params.args[1], name_params.args[2:end]
#    quote
#        export $(esc(name))
#        $(esc(name))($(map(esc,params)...), backend::Backend=current_backend()) =
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
backend_chair(b::Backend, p, angle, f) =
  let mat = get_material(b, family_ref(b, f))
    realize_chair(b, mat, loc_from_o_phi(p, angle), f.length, f.width, f.height, f.seat_height, f.thickness)
  end

#@bdef create_layer(name::String, active::Bool, color::RGB)

#@bdef curtain_wall(s, path::Path, bottom::Real, height::Real, l_thickness::Real, r_thickness::Real, kind::Symbol)
backend_curtain_wall(b::Backend, s, path::Path, bottom::Real, height::Real, l_thickness::Real, r_thickness::Real, kind::Symbol) =
  let family = getproperty(s.family, kind),
      (lmat, rmat, smat) = material_refs(b, family_materials(b, family)[1:3])
    with_family_in_layer(b, family) do
      b_wall(b, translate(path, vz(bottom)), height, l_thickness, r_thickness, lmat, rmat, smat)
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

backend_pyramid(b::Backend, bot_vs::Locs, top::Loc) =
  let refs = [backend_surface_polygon(b, reverse(bot_vs))]
    for (v1, v2, v3) in zip(bot_vs, circshift(bot_vs, -1), repeated(top))
      push!(refs, backend_surface_polygon(b, [v1, v2, v3]))
    end
    refs
  end

#@bdef pyramid_frustum(bs::Locs, ts::Locs)
backend_pyramid_frustum(b::Backend, bot_vs::Locs, top_vs::Locs) =
  let refs = [backend_surface_polygon(b, reverse(bot_vs))]
    push!(refs, backend_surface_polygon(b, top_vs))
    for (v1, v2, v3, v4) in zip(bot_vs, circshift(bot_vs, -1), circshift(top_vs, -1), top_vs)
      push!(refs, backend_surface_polygon(b, [v1, v2, v3, v4]))
    end
    refs
  end

@bdef b_set_view(camera, target, lens, aperture)
@bdef b_get_view()

@bdef b_realistic_sky(altitude, azimuth, turbidity, withsun)
@bdef b_realistic_sky(date, latitude, longitude, elevation, meridian, turbidity, withsun)

@bdef b_render_view(path)

#@bdef revolve_curve(profile::Shape, p::Loc, n::Vec, start_angle::Real, amplitude::Real)
#@bdef revolve_point(profile::Shape, p::Loc, n::Vec, start_angle::Real, amplitude::Real)
#@bdef revolve_surface(profile::Shape, p::Loc, n::Vec, start_angle::Real, amplitude::Real)
@bdef right_cuboid(cb, width, height, h, angle, material)
@bdef right_cuboid(cb, width, height, h, material)

#@bdef slab(profile, holes, thickness, family)
backend_slab(b::Backend, profile, openings, thickness, family) =
  let (mattop, matbot, matside) = slab_materials(b, family_ref(b, family))
    realize_prism(
      b, mattop, matbot, matside,
      isempty(openings) ? profile : path_set(profile, openings...), thickness)
  end

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
        (start,
         center + vpol(op.radius, op.start_angle + op.amplitude),
         push!(refs, backend_stroke_arc(b, center, op.radius, op.start_angle, op.amplitude)))
     end


#

backend_stroke(b::Backend, m::Mesh) =
  let vs = m.vertices
    for face in m.faces
      backend_stroke_line(b, vs[face.+1]) #1-indexed
    end
  end


@bdef stroke_unite(refs)
#@bdef surface_boundary(s::Shape2D)
#@bdef surface_domain(s::Shape2D)
@bdef surface_grid(pts, closed_u, closed_v, smooth_u, smooth_v)
@bdef surface_mesh(vertices, faces)

#@bdef surface_polygon(mat, path::Path, acw::Bool)
backend_surface_polygon(b::Backend, mat, path::Path, acw) =
  backend_surface_polygon(b, mat, path_vertices(path), acw)

@bdef surface_polygon(mat, path::PathSet, acw::Bool)
@bdef surface_polygon(mat, vs::Locs, acw::Bool)
@bdef surface_polygon(vs::Locs)
#@bdef sweep(path::Shape, profile::Shape, rotation::Real, scale::Real)

#@bdef wall(path, height, l_thickness, r_thickness, family)

# A poor's man approach to deal with Z-fighting
const support_z_fighting_factor = 0.999
const wall_z_fighting_factor = 0.998

backend_wall(b::Backend, w_path, w_height, l_thickness, r_thickness, family) =
  path_length(w_path) < path_tolerance() ?
    realize(b, empty_shape()) : # not beatiful
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

SocketBackend{K,T}(c::LazyParameter{TCPSocket}) where {K,T} =
  SocketBackend{K,T}(c, NamedTuple{}())

connection(b::SocketBackend{K,T}) where {K,T} = b.connection()

reset_backend(b::SocketBackend{K,T}) where {K,T} =
  begin
    for f in b.remote
      reset_opcode(f)
    end
    close(b.connection())
    reset(b.connection)
  end

#One less dynamic option is to use a file-based backend. To that end, we implement
#the IOBuffer_Backend

struct IOBufferBackend{K,T} <: Backend{K,T}
  out::IOBuffer
end
connection(backend::IOBufferBackend{K,T}) where {K,T} = backend.out
