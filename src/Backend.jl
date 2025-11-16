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
backend_name(b::Backend) = string(typeof(b))
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

export new_refs
new_refs(b::Backend{K,T}) where {K,T} = T[]

############################################################
# Zeroth tier: curves. Not all backends support these.

export b_point, b_line, b_closed_line, b_polygon, b_regular_polygon,
       b_nurbs_curve,
       b_spline, b_closed_spline, b_circle, b_arc, b_ellipse, b_rectangle

@bdef(b_point(p, mat))

@bdef(b_line(ps, mat))

b_polygon(b::Backend, ps, mat) =
  b_line(b, [ps..., ps[1]], mat)

# Legacy
const b_closed_line = b_polygon

b_regular_polygon(b::Backend, edges, c, r, angle, inscribed, mat) =
  b_polygon(b, regular_polygon_vertices(edges, c, r, angle, inscribed), mat)

b_nurbs_curve(b::Backend, ps, order, cps, knots, weights, closed, mat) =
  closed ?
  	b_polygon(b, ps, mat) :
  	b_line(b, ps, mat)

b_spline(b::Backend, ps, mat) =
  b_spline(b, ps, false, false, mat)

b_spline(b::Backend, ps, v1, v2, mat) =
  let ci = curve_interpolator(ps, false),
      cps = curve_control_points(ci),
      n = length(cps),
      knots = curve_knots(ci)
    b_nurbs_curve(b, ps, 5, cps, knots, fill(1.0, n), false, mat)
  end

b_closed_spline(b::Backend, ps, mat) =
  let ci = curve_interpolator(ps, true),
      cps = curve_control_points(ci),
      n = length(cps),
      knots = curve_knots(ci)
    b_nurbs_curve(b, ps, 5, cps, knots, fill(1.0, n), true, mat)
  end

b_circle(b::Backend, c, r, mat) =
  b_closed_spline(b, regular_polygon_vertices(32, c, r, 0, true), mat)

b_arc(b::Backend, c, r, α, Δα, mat) =
  b_spline(b,
    [c + vpol(r, a, c.cs)
     for a in division(α, α + Δα, max(ceil(Int, Δα*32/2/π), 2), true)],
    nothing, nothing, # THIS NEEDS TO BE FIXED
    mat)

b_ellipse(b::Backend, c, rx, ry, mat) =
  b_closed_spline(b,
    [add_xy(c, rx*cos(ϕ), ry*sin(ϕ))
     for ϕ in division(0, 2pi, 64, false)], mat)
  
b_rectangle(b::Backend, c, dx, dy, mat) =
  b_polygon(b, [c, add_x(c, dx), add_xy(c, dx, dy), add_y(c, dy)], mat)

#############################################################
# First tier: everything is a triangle or a set of triangles
export b_trig, b_quad, b_ngon, b_quad_strip, b_quad_strip_closed, b_strip

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
  vcat([b_quad(b, ps[i], ps[i+1], qs[i+1], qs[i], mat) for i in 1:size(ps,1)-1]...)

b_quad_strip_closed(b::Backend, ps, qs, smooth, mat) =
  b_quad_strip(b, [ps..., ps[1]], [qs..., qs[1]], smooth, mat)

b_strip(b::Backend, path1, path2, mat) =
  let v1s = path_vertices(path1),
	    v2s = path_vertices(path2)
    length(v1s) != length(v2s) ?
  	  error("Paths with different resolution ($(length(v1s)) vs $(length(v2s)))") :
      is_closed_path(path1) && is_closed_path(path2) ?
        b_quad_strip_closed(
          b, v1s, v2s, is_smooth_path(path1) || is_smooth_path(path2), mat) :
        b_quad_strip(
          b, v1s, v2s, is_smooth_path(path1) || is_smooth_path(path2), mat)
  end

b_strip(b::Backend, path1::Region, path2::Region, mat) =
  [b_strip(b, outer_path(path1), outer_path(path2), mat),
   [b_strip(b, p2, p1, mat) for (p1, p2) in zip(inner_paths(path1), inner_paths(path2))]...
  ]

############################################################
# Second tier: surfaces
export b_surface_polygon, b_surface_polygon_with_holes, 
       b_surface_regular_polygon,
	     b_surface_circle, b_surface_arc, b_surface_ellipse, b_surface_closed_spline,
	     b_surface, b_surface_grid, b_smooth_surface_grid, b_surface_mesh

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
         [c + vpol(r, a, c.cs)
          for a in division(α, α + Δα, Δα*32/2/π, true)],
         c, false, mat)

b_surface_ellipse(b::Backend, c, rx, ry, mat) =
  b_surface_closed_spline(b,
    [add_xy(c, rx*cos(ϕ), ry*sin(ϕ))
     for ϕ in division(0, 2pi, 64, false)], mat)

@bdef(b_surface_closed_spline(ps, mat))

#=
This implementation is based on independent quadstrips, therefore, it is impossible
to be smooth in both u,v dimensions. We opt to be smooth in just the v dimension.
=#

b_surface_grid(b::Backend, ptss, closed_u, closed_v, smooth_u, smooth_v, mat) =
  let ptss = maybe_interpolate_grid(ptss, smooth_u, smooth_v),
      (nu, nv) = size(ptss)
	  closed_v ?
      vcat([b_quad_strip_closed(b, ptss[i,:], ptss[i+1,:], smooth_v, mat) for i in 1:nu-1]...,
	         (closed_u ? [b_quad_strip_closed(b, ptss[end,:], ptss[1,:], smooth_v, mat)] : new_refs(b))...) :
	    vcat([b_quad_strip(b, ptss[i,:], ptss[i+1,:], smooth_v, mat) for i in 1:nu-1]...,
	         (closed_u ? [b_quad_strip(b, ptss[end,:], ptss[1,:], smooth_v, mat)] : new_refs(b))...)
  end

export maybe_interpolate_grid
maybe_interpolate_grid(ptss, smooth_u, smooth_v) =
  smooth_u || smooth_v ? 
    let interpolator = grid_interpolator(ptss),
        (nu, nv) = size(ptss)
      [location_at(interpolator, u, v)
	     for u in division(0, 1, smooth_u ? 4*nu-7 : nu-1),
           v in division(0, 1, smooth_v ? 4*nv-7 : nv-1)] # 2->1, 3->5, 4->9, 5->13
    end :
    ptss

b_smooth_surface_grid(b::Backend, ptss, closed_u, closed_v, mat) =
  b_surface_grid(b, smooth_grid(ptss), closed_u, closed_v, true, true, mat)

b_surface_mesh(b::Backend, vertices, faces, mat) =
  map(faces) do face
	  if length(face) == 3
	    b_trig(b, vertices[face]..., mat)
    	elseif length(face) == 4
    	  b_quad(b, vertices[face]..., mat)
      else
	    b_surface_polygon(b, vertices[face], mat)
      end
    end

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

#=
We will use trigs, quads, quad_strips, and so on, but in order
to make a solid in the end, we will rely on a solidify operation that, 
by default, does nothing. However, on backends where solids do exist,
the solidify operation should convert those primitive shapes into a 
proper solid.
=#

export b_generic_pyramid_frustum, b_generic_pyramid, b_generic_prism,
       b_generic_pyramid_frustum_with_holes, b_generic_prism_with_holes,
  	   b_pyramid_frustum, b_pyramid, b_prism,
  	   b_regular_pyramid_frustum, b_regular_pyramid, b_regular_prism,
  	   b_cylinder,
  	   b_cuboid,
  	   b_box,
  	   b_sphere,
  	   b_cone,
  	   b_torus,
       b_solidify

b_solidify(b::Backend, refs) = refs

# Each solid can have just one material or multiple materials
b_generic_pyramid_frustum(b::Backend, bs, ts, smooth, bmat, tmat, smat) =
  b_solidify(b,
    vcat(isnothing(bmat) ? new_refs(b) : b_surface_polygon(b, reverse(bs), bmat),
         b_quad_strip_closed(b, bs, ts, smooth, smat),
         isnothing(tmat) ? new_refs(b) : b_surface_polygon(b, ts, tmat)))

b_generic_pyramid_frustum_with_holes(b::Backend, bs, ts, smooth, bbs, tts, smooths, bmat, tmat, smat) =
  b_solidify(b,
    [b_surface_polygon_with_holes(b, reverse(bs), bbs, bmat),
     b_quad_strip_closed(b, bs, ts, smooth, smat),
     [b_quad_strip_closed(b, bs, ts, smooth, smat)
      for (bs, ts, smooth) in zip(bbs, tts, smooths)]...,
     b_surface_polygon_with_holes(b, ts, reverse.(tts), tmat)])

b_generic_pyramid(b::Backend, bs, t, smooth, bmat, smat) =
  b_solidify(b,
    vcat(b_surface_polygon(b, reverse(bs), bmat),
	       b_ngon(b, bs, t, smooth, smat)))

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
	b_regular_pyramid_frustum(b, edges, cb, rb, angle, h, rb, inscribed, bmat, tmat, smat)

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
  b_solidify(b,
    vcat(b_surface_polygon(b, [pb3, pb2, pb1, pb0], mat), # quads do not work with concave quads
         b_quad_strip_closed(b, [pb0, pb1, pb2, pb3], [pt0, pt1, pt2, pt3], false, mat),
         b_surface_polygon(b, [pt0, pt1, pt2, pt3], mat)))

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
    b_solidify(b,
      [b_ngon(b, [add_sph(c, r, ϕ, π/16) for ϕ in ϕs], add_sph(c, r, 0, 0), true, mat),
    	 [b_quad_strip_closed(b,
    			[add_sph(c, r, ϕ, ψ+π/16) for ϕ in ϕs],
    			[add_sph(c, r, ϕ, ψ) for ϕ in ϕs],
    			true, mat) for ψ in π/16:π/16:π-π/16]...,
    	 b_ngon(b, reverse!([add_sph(c, r, ϕ, π-π/16) for ϕ in ϕs]), add_sph(c, r, 0, π), true, mat)])
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
  	 for ψ in division(0, 2π, 32, false), ϕ in division(0, 2π, 64, false)],
      true, true,
    	true, true,
    	mat)

export b_mesh_obj_fmt
@bdef(b_mesh_obj_fmt(obj_name))
export b_set_environment
@bdef(b_set_environment(env_name, set_background))

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

b_surface(b::Backend, frontier::Shapes, mat) =
  let path = foldr(join_paths, [convert(OpenPolygonalPath, shape_path(e)) for e in frontier])
    and_delete_shapes(b_surface_polygon(b, path_vertices(path), mat), frontier)
  end

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

# Extrusions, lofts, sweeps, etc
export b_extruded_point, b_extruded_curve, b_extruded_surface, b_sweep, b_loft

# Extruding a profile
b_extruded_point(b::Backend, path::PointPath, v, cb, mat) =
  let p = path_on(path, cb).location
    b_line(b, [p, p + v], mat)
  end

#=
b_extruded_point(b::Backend, pt::Shape0D, v, cb, mat) =
  b_extruded_point(b, convert(Path, pt), v, cb, mat)
=#

b_extruded_curve(b::Backend, path::OpenPolygonalPath, v, cb, mat) =
 	let bs = path_vertices_on(path, cb),
	  	ts = translate(bs, v)
  	b_quad_strip(b, bs, ts, is_smooth_path(path), mat)
  end

b_extruded_curve(b::Backend, path::ClosedPolygonalPath, v, cb, mat) =
  b_extruded_curve(b, convert(OpenPolygonalPath, path), v, cb, mat)

#=
b_extrusion(b::Backend, path, v, cb, bmat, tmat, smat) =
  b_generic_prism(
    b,
    path_vertices_on(path, cb),
    is_smooth_path(path),
    v,
    bmat, tmat, smat)
=#

b_extruded_curve(b::Backend, profile::CircularPath, v, cb, mat) =
  v.cs === cb.cs && iszero(v.x) && iszero(v.y) ?
    # HACK: This is wrong. It should be an open cylinder
  	b_cylinder(b, add_xy(cb, profile.center.x, profile.center.y), profile.radius, v.z, nothing, nothing, mat) :
	b_generic_prism(b,
	  path_vertices_on(profile, cb),
	  is_smooth_path(profile),
	  v,
	  nothing, nothing, mat)

b_extruded_surface(b::Backend, profile, v, cb, mat) =
  b_extruded_surface(b, profile, v, cb, mat, mat, mat)

#=
b_extruded_surface(b::Backend, path::Path, v, cb, bmat, tmat, smat) =
  b_extruded_surface(b, region(path), v, cb, bmat, tmat, smat)
=#

b_extruded_surface(b::Backend, profile::Region, v, cb, bmat, tmat, smat) =
  let outer = outer_path(profile),
      inners = inner_paths(profile)
    vcat(b_extruded_curve(b, outer, v, cb, smat),
         [b_extruded_curve(b, inner, v, cb, smat) for inner in inners]...,
         b_surface(b, path_on(profile, cb), bmat),
         b_surface(b, translate(path_on(profile, cb), v), tmat))
  end
    #=
      isempty(inners) ?
      b_generic_prism(
        b,
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
=#
b_extruded_curve(b::Backend, profile::PathSequence, v, cb, mat) =
  vcat([b_extruded_curve(b, subprofile, v, cb, mat) for subprofile in profile.paths]...)
b_extruded_curve(b::Backend, profile::Path, v, cb, mat) =
  b_extruded_curve(b, convert(OpenPolygonalPath, profile), v, cb, mat)

#=
b_extruded_curve(b::Backend, profile::Shape1D, v, cb, mat) =
  b_extruded_curve(b, convert(Path, profile), v, cb, mat)
b_extruded_surface(b::Backend, profile::Shape2D, v, cb, mat) =
  b_extruded_surface(b, convert(Region, profile), v, cb, mat)
=#

b_loft(b::Backend, profiles, closed, smooth, mat) =
  let ptss = path_vertices.(profiles),
	  n = mapreduce(length, max, ptss),
	  vss = map(profile->map_division(identity, profile, n), profiles)
	b_surface_grid(b, hcat(vss...), is_closed_path(profiles[1]), closed, is_smooth_path(profiles[1]), smooth, mat)
  end

b_swept_curve(b::Backend, path::Path, profile::Path, rotation, scaling, mat) =
  b_sweep(b, path, profile, rotation, scaling, mat)

#b_swept_curve(b::Backend, path::Shape1D, profile::Shape1D, rotation, scaling, mat) =
#  b_sweep(b, convert(Path, path), convert(Path, profile), rotation, scaling, mat)

b_swept_surface(b::Backend, path::Path, profile::Region, rotation, scaling, mat) =
  b_sweep(b, path, profile, rotation, scaling, mat)

#b_swept_surface(b::Backend, path::Shape1D, profile::Shape2D, rotation, scaling, mat) =
#  b_sweep(b, convert(Path, path), convert(Region, profile), rotation, scaling, mat)

b_sweep(b::Backend, path, profile::Region, rotation, scaling, mat) =
  let outer = outer_path(profile),
      inners = inner_paths(profile),
      frames = path_interpolated_frames(path),
      frames = rotation == 0 ?
        frames : 
        [loc_from_o_phi(frame, phi) for (frame, phi) in zip(frames, map_division(identity, 0, rotation, length(frames)-1))],
      profiles = map_division(s->scale(profile, s, u0()), 1, scaling, length(frames)-1)
    vcat(b_sweep(b, path, outer, rotation, scaling, mat),
         [b_sweep(b, path, inner, rotation, scaling, mat) for inner in inners]...,
         # HACK
         # If the final profile coincides with the first, then it doesn't make sense
         # to create the closing surfaces
         b_surface(b, path_on(profiles[1], frames[1]), mat),
         b_surface(b, path_on(profiles[end], frames[end]), mat))
  end

b_sweep(b::Backend, path, profile, rotation, scaling, mat) =
  let frames = rotation == 0 ?
                rotation_minimizing_frames(path_frames(path)) :
                let subframes = path_interpolated_frames(path, path_domain(path), 
                                                         collinearity_tolerance(),
                                                         ceil(Int, log(abs(rotation))*4))
                  map(loc_from_o_phi, subframes, map_division(identity, 0, rotation, length(subframes)-1))
                end,
      profiles = map_division(s->scale(profile, s, u0()), 1, scaling, length(frames)-1),
      verticess = map(path_vertices, profiles),
	    verticess = is_smooth_path(profile) ?
	  	  let n = mapreduce(length, max, verticess)-1
		      map(path->map_division(identity, path, n), profiles)
	      end :
	      verticess,
   	  points = hcat(map(on_cs, verticess, frames)...)
    #[b_sphere(b, p, 0.01, mat) for p in points]
    b_surface_grid(
      b,
      points,
  	  is_closed_path(profile),
      is_closed_path(path),
  	  is_smooth_path(profile),
      is_smooth_path(path),
	    mat)
  end

b_revolved_point(b::Backend, profile, p, n, start_angle, amplitude, mat) =
  let q = profile.position,
      pp = perpendicular_point(p, n, q)
    b_arc(b, loc_from_o_vz(pp, n), distance(pp, q), start_angle, amplitude, mat)
  end

b_revolved_curve(b::Backend, profile, p, n, start_angle, amplitude, mat) = 
  let profile = translate(convert(Path, profile), u0()-p),
      pp = loc_from_o_vz(p, n),
      vertices = path_frames(profile),
      frames = map_division(ϕ -> loc_from_o_phi(pp, start_angle + ϕ), 0, amplitude, ceil(Int, amplitude*10), amplitude < 2pi),
      points = hcat(map(frame->on_cs(vertices, frame), frames)...)
    b_surface_grid(
        b,
        points,
        is_closed_path(profile),
        amplitude >= 2pi,
        is_smooth_path(profile),
        true,
        mat)
  end

b_revolved_surface(b::Backend, profile, p, n, start_angle, amplitude, mat) =
  let profile = convert(Region, profile),
      outer = outer_path(profile),
      inners = inner_paths(profile)
    vcat(b_revolved_curve(b, outer, p, n, start_angle, amplitude, mat),
         [b_revolved_curve(b, inner, p, n, start_angle, amplitude, mat) for inner in inners]...,
         (amplitude < 2pi ?
          let pp = loc_from_o_vz(p, n),
              profile = translate(profile, u0()-p),
              frames = map_division(ϕ -> loc_from_o_phi(pp, start_angle + ϕ), 0, amplitude, 1)
           [b_surface(b, path_on(profile, frames[1]), mat),
            b_surface(b, path_on(profile, frames[end]), mat)]
          end :
          new_refs(b)))
  end


# Booleans

export b_subtracted, b_intersected, b_united,
       b_subtracted_surfaces, b_intersected_surfaces, b_united_surfaces, 
       b_subtracted_solids, b_intersected_solids, b_united_solids,
       b_slice, b_unite_refs, b_slice_ref

@bdef b_subtract_ref(sref, mref)
@bdef b_intersect_ref(sref, mref)
b_unite_ref(b::Backend, sref, mref) = vcat(sref, mref)

b_unite_refs(b::Backend, rs) =
  foldl((s, r)->b_unite_ref(b, s, r), rs)

b_subtracted_surfaces(b::Backend, source, mask, mat) =
  b_subtracted(b, source, mask, mat)

b_subtracted_solids(b::Backend, source, mask, mat) =
  b_subtracted(b, source, mask, mat)

b_subtracted(b::Backend, source, mask, mat) =
  and_mark_deleted(b,
    map_ref(b, source) do s
      and_mark_deleted(b,
        map_ref(b, mask) do r
          b_subtract_ref(b, s, r)
        end,
        [mask])
      end,
    [source])

b_intersected_surfaces(b::Backend, source, mask, mat) =
  b_intersected(b, source, mask, mat)
  
b_intersected_solids(b::Backend, source, mask, mat) =
  b_intersected(b, source, mask, mat)
  
b_intersected(b::Backend, source, mask, mat) =
  and_mark_deleted(b,
    map_ref(b, source) do s
      and_mark_deleted(b,
        map_ref(b, mask) do r
          b_intersect_ref(b, s, r)
        end,
        [mask])
      end,
    [source])

b_united_surfaces(b::Backend, source, mask, mat) =
  b_united(b, source, mask, mat)
    
b_united_solids(b::Backend, source, mask, mat) =
  b_united(b, source, mask, mat)
  
b_united(b::Backend, source, mask, mat) =
  and_mark_deleted(b,
    b_unite_refs(b, vcat(ref_values(b, source), ref_values(b, mask))),
    [source, mask])

b_slice(b::Backend, shape, p, v, mat) =
  and_mark_deleted(b,
    map_ref(b, shape) do s
      b_slice_ref(b, s, p, v)
    end,
    [shape])

b_slice_ref(b::Backend, r, p, v) =
  b_subtract_ref(b, r, b_regular_prism(b, 4, loc_from_o_vz(p, v), 1e5, 0, 1e5, true, void_ref(b)))

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
  b_spline(b, path.vertices, path.v0, path.v1, mat)
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
b_stroke(b::Backend, path::PathSequence, mat) =
  for path in path.paths
    b_stroke(b, path, mat)
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
b_fill(b::Backend, path::Mesh, mat) =
  b_surface_mesh(b, m.vertices, m.faces, mat)

export b_realize_path
b_realize_path(b::Backend, path::Region, mat) =
  b_fill(b, path, mat)
b_realize_path(b::Backend, path, mat) =
  b_stroke(b, path, mat)

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
export b_dimension, b_ext_line, b_dim_line, b_text, b_text_size, b_arc_dimension

b_dimension(b::Backend, p, q, str, size, offset, mat) =
  let qp = in_world(q - p),
	    phi = pol_phi(qp),
	    outside = pi/2 <= phi <= 3pi/2,
	    v = vpol(outside ? size : 2*size, phi-pi/2),
	    uv = unitized(v),
	    (si, se) = (offset*size, 2*offset*size),
	    (vi, ve) = (uv*si, uv*se),
	    (tp, tq, tv) = outside ? (q, p, vpol(1, phi + pi)) : (p, q, vpol(1, phi))
	  offset == 0 ?
	    b_dim_line(b, tp, tq, tv, str, size, outside, mat) :
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

b_arc_dimension(b::Backend, c, r, α, Δα, rstr, dstr, size, offset, mat) =
  error("To be finished")

##################################################################
# Text
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
	  refs = new_refs(b)
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
# Illustrations

@bdef(b_labels(p, data, mat))
@bdef(b_radii_illustration(c, rs, rs_txts, mat))
@bdef(b_vectors_illustration(p, a, rs, rs_txts, mat))
@bdef(b_angles_illustration(c, rs, ss, as, r_txts, s_txts, a_txts, mat))
@bdef(b_arcs_illustration(c, rs, ss, as, r_txts, s_txts, a_txts, mat))

##################################################################
# Materials
#=
A material is a description of the appearence of an object.

There are two default cases:
1. When the specification of the material is nothing, we use the void_ref to
indicate that we don't want to specify the material.
2. Otherwise, the material already exists in the backend and is accessed by name.

In general, backends need to specialize this function to address additional cases.
=#

export b_get_material

b_get_material(b::Backend, spec::Nothing) = void_ref(b)
#Is this really needed? Yes, e.g., POVRay.
b_get_material(b::Backend, spec::Any) = spec

#=
In other cases, the material can be algorithmically created.

There are numerous models for a material (Phong, Blinn–Phong, Cook–Torrance,
Lambert, Oren–Nayar, Minnaert, etc), covering wildly different materials such
as glass, metal, rubber, clay, etc. Unfortunately, different render engines
implement different subsets of those models, making it impossible to have portable
code that depends on a specific model. Additionally, each of these models have
different parameters, some of which might be difficult to specify by a non-expert.
Therefore, we prefer to provide a more intuitive approach, based on providing
different categories of materials, each with a minimal number of parameters.
Each backend is responsible for selecting the most adequate reflection model and
convert the generic material parameters into specific model parameters.

=#

@bdef b_new_material(b::Backend, name, base_color, metallic, specular, roughness,
	           	     clearcoat, clearcoat_roughness, ior,
                     transmission, transmission_roughness,
	           	     emission_color, emission_strength)
@bdef b_plastic_material(b::Backend, name, color, roughness)
@bdef b_metal_material(b::Backend, name, color, roughness, ior)
@bdef b_glass_material(b::Backend, name, color, roughness, ior)
@bdef b_mirror_material(b::Backend, name, color)

#=
Utilities for interactive development
=#

@bdef(b_all_shape_refs())

b_delete_all_shape_refs(b::Backend) =
  b_delete_refs(b, b_all_shape_refs(b))

b_delete_refs(b::Backend{K,T}, rs::Vector{T}) where {K,T} =
  for r in rs
	  b_delete_ref(b, r)
  end

b_delete_ref(b::Backend{K,T}, r::T) where {K,T} =
  missing_specialization(b, :b_delete_ref, r)


b_highlight_refs(b::Backend{K,T}, rs::Vector{T}) where {K,T} =
  for r in rs
 	  b_highlight_ref(b, r)
  end

b_highlight_ref(b::Backend{K,T}, r::T) where {K,T} =
  missing_specialization(b, :b_highlight_ref, r)

b_unhighlight_refs(b::Backend{K,T}, rs::Vector{T}) where {K,T} =
  for r in rs
   	b_unhighlight_ref(b, r)
  end

b_unhighlight_ref(b::Backend{K,T}, r::T) where {K,T} =
  missing_specialization(b, :b_unhighlight_ref, r)

b_unhighlight_all_refs(b::Backend) =
  b_unhighlight_refs(b, b_all_shape_refs(b))

# BIM
export b_slab, b_roof, b_beam, b_column, b_free_column, b_wall, b_curtain_wall

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
export material_ref, material_refs
material_refs(b::Backend, materials) =
  [material_ref(b, mat) for mat in materials]

materialize_path(b, c_r_w_path, c_l_w_path, mat) =
  with_material_as_layer(b, mat) do
    b_strip(b, c_l_w_path, c_r_w_path, material_ref(b, mat))
  end
materialize_path(b, path, mat) =
  with_material_as_layer(b, mat) do
    b_surface(b, path, material_ref(b, mat))
  end

# b_slab(b::Backend, profile, level, family) =
#   let tmat = family.top_material,
#       bmat = family.bottom_material,
#       smat = family.side_material,
#       bprof = translate(profile, vz(level_height(b, level) + slab_family_elevation(b, family))),
#       tprof = translate(bprof, vz(slab_family_thickness(b, family)))
#     [materialize_path(b, tprof, tmat),
#      materialize_path(b, bprof, tprof, smat)...,
#      materialize_path(b, reverse(bprof), bmat)]
#   end
# It is preferable to do an extrusion because separate discretization of the top and bottom
# paths do not ensure an equal number of vertices.
b_slab(b::Backend, profile, level, family) =
  let tmat = material_ref(b, family.top_material),
      bmat = material_ref(b, family.bottom_material),
      smat = material_ref(b, family.side_material),
      v = vz(slab_family_thickness(b, family))
    b_extruded_surface(b, profile, v, z(level_height(b, level) + slab_family_elevation(b, family)), bmat, tmat, smat)
   end

b_roof(b::Backend, region, level, family) =
  b_slab(b, region, level, family)

# b_panel(b::Backend, region, family) =
#   let th = family.thickness,
# 	    v = planar_path_normal(region),
#   	  left = translate(region, v*-th),
#   	  right = translate(region, v*th)
#     [materialize_path(b, right, family.right_material),
#      materialize_path(b, left, right, family.side_material)...,
#      materialize_path(b, reverse(left), family.left_material)]
#   end
b_panel(b::Backend, profile, family) =
  let lmat = material_ref(b, family.left_material),
      rmat = material_ref(b, family.right_material),
      smat = material_ref(b, family.side_material),
      th = family.thickness,
      v = planar_path_normal(profile)
    b_extruded_surface(b, profile, v*th, u0(v.cs)+v*(th/-2), lmat, rmat, smat)
  end

b_beam(b::Backend, c, h, angle, family) =
  let c = loc_from_o_phi(c, angle),
	    mat = material_ref(b, family.material)
  	with_material_as_layer(b, family.material) do
      b_extruded_surface(b, region(family_profile(b, family)), vz(h, c.cs), c,	mat, mat, mat)
	  end
  end

b_column(b::Backend, cb, angle, bottom_level, top_level, family) =
  let base_height = level_height(b, bottom_level),
      top_height = level_height(b, top_level)
    b_beam(b, add_z(loc_from_o_phi(cb, angle), base_height), top_height-base_height, 0, family)
  end

b_free_column(b::Backend, cb, h, angle, family) =
  b_beam(b, cb, h, angle, family)

b_wall(b::Backend, w_path, w_height, l_thickness, r_thickness, family) =
  path_length(w_path) < path_tolerance() ?
  	void_ref(b) :
    let w_paths = subpaths(w_path),
        r_w_paths = subpaths(offset(w_path, -r_thickness)),
        l_w_paths = subpaths(offset(w_path, l_thickness)),
        w_height = w_height*wall_z_fighting_factor,
  	    (lmat, rmat, smat) = (family.left_material, family.right_material, family.side_material),
        prevlength = 0,
        refs = new_refs(b)
      for (w_seg_path, r_w_path, l_w_path) in zip(w_paths, r_w_paths, l_w_paths)
        let currlength = prevlength + path_length(w_seg_path),
            c_r_w_path = closed_path_for_height(r_w_path, w_height),
            c_l_w_path = closed_path_for_height(l_w_path, w_height)
          #append!(refs, b_pyramid_frustum(b, path_vertices(c_r_w_path), path_vertices(c_l_w_path), rmat, lmat, smat))
          append!(refs, materialize_path(b, c_r_w_path, c_l_w_path, smat))
          push!(refs, materialize_path(b, reverse(c_l_w_path), lmat))
          push!(refs, materialize_path(b, c_r_w_path, rmat))
          prevlength = currlength
        end
      end
      refs
    end

b_curtain_wall(b::Backend, path, bottom_level, top_level, family, offset) =
  let th = family.panel.thickness,
      bfw = family.boundary_frame.width,
      bfd = family.boundary_frame.depth,
      bfdo = family.boundary_frame.depth_offset,
      mfw = family.mullion_frame.width,
      mfd = family.mullion_frame.depth,
      mdfo = family.mullion_frame.depth_offset,
      tfw = family.transom_frame.width,
      tfd = family.transom_frame.depth,
      tfdo = family.transom_frame.depth_offset,
      path = curtain_wall_panel_path(b, path, family),
      path_length = path_length(path),
      bottom = level_height(b, bottom_level),
      top = level_height(b, top_level),
      height = top - bottom,
      x_panels = ceil(Int, path_length/family.max_panel_dx),
      y_panels = ceil(Int, height/family.max_panel_dy),
      refs = new_refs(b)
    append!(refs, b_curtain_wall_element(b, subpath(path, bfw, path_length-bfw), bottom+bfw, height-2*bfw, th/2, th/2, getproperty(family, :panel)))
    append!(refs, b_curtain_wall_element(b, path, bottom, bfw, l_thickness(bfdo, bfd), r_thickness(bfdo, bfd), getproperty(family, :boundary_frame)))
    append!(refs, b_curtain_wall_element(b, path, top-bfw, bfw, l_thickness(bfdo, bfd), r_thickness(bfdo, bfd), getproperty(family, :boundary_frame)))
    append!(refs, b_curtain_wall_element(b, subpath(path, 0, bfw), bottom+bfw, height-2*bfw, l_thickness(bfdo, bfd), r_thickness(bfdo, bfd), getproperty(family, :boundary_frame)))
    append!(refs, b_curtain_wall_element(b, subpath(path, path_length-bfw, path_length), bottom+bfw, height-2*bfw, l_thickness(bfdo, bfd), r_thickness(bfdo, bfd), getproperty(family, :boundary_frame)))
    for i in 1:y_panels-1
      l = height/y_panels*i
      sub = subpath(path, bfw, path_length-bfw)
      append!(refs, b_curtain_wall_element(b, sub, bottom+l-tfw/2, tfw, l_thickness(tfdo, tfd), r_thickness(tfdo, tfd), getproperty(family, :transom_frame)))
    end
    for i in 1:x_panels-1
      l = path_length/x_panels*i
      append!(refs, b_curtain_wall_element(b, subpath(path, l-mfw/2, l+mfw/2), bottom+bfw, height-2*bfw, l_thickness(mdfo, mfd), r_thickness(mdfo, mfd), getproperty(family, :mullion_frame)))
    end
    refs
  end

curtain_wall_panel_path(b::Backend, path, family) =
  let path_length = path_length(path),
      x_panels = ceil(Int, path_length/family.max_panel_dx),
      pts = map(t->in_world(location_at_length(path, t)),
                division(0, path_length, x_panels))
    polygonal_path(pts)
  end

#AML TO BE CONTINUED!!!
#b_wall(b::Backend, w_path, w_openings, l_thickness, r_thickness, family) =

export b_toilet, b_sink, b_closet
b_toilet(b::Backend, c, host, family) =
  b_box(b, c - vxy(20, 20, c.cs), 40, 40, 40, nothing)

b_sink(b::Backend, c, host, family) =
  b_box(b, c - vxy(40, 40, c.cs), 80, 80, 80, nothing)

b_closet(b::Backend, c, host, family) =
  b_box(b, c - vxy(100, 40, c.cs), 200, 80, 200, nothing)

# Lights

@bdef b_ieslight(file, loc, dir, alpha, beta, gamma)
@bdef b_pointlight(loc, energy, color)
@bdef b_spotlight(loc, dir, hotspot, falloff)
@bdef b_arealight(loc, dir, size, energy, color)

# Trusses
export b_truss_node, b_truss_node_support, b_truss_bar

b_truss_node(b::Backend, p, family) =
  with_material_as_layer(b, family.material) do
  	b_sphere(b, p, family.radius, material_ref(b, family.material))
  end

b_truss_node_support(b::Backend, cb, family) =
  with_material_as_layer(b, family.material) do
    b_regular_pyramid(
      b, 4, add_z(cb, -3*family.radius),
      family.radius, 0, 3*family.radius, false,
      material_ref(b, family.material))
  end

b_truss_bar(b::Backend, p, q, family) =
  with_material_as_layer(b, family.material) do
    let (c, h) = position_and_height(p, q)
      b_cylinder(
        b, c, family.radius, h,
        material_ref(b, family.material))
    end
  end

# Analysis

@bdef b_truss_analysis(load::Vec, self_weight::Bool, point_loads::Dict)
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

#=
Moving Khepri to a multi-threaded model requires that each thread uses its own
current backend.

This means that current_backends should only be used to intentionally broadcast
an operation to several backends at the same time. In all other cases, current_backend
should be used to retrieve the backend for the current thread. 

TO BE CONTINUED!
=#

# We can have several backends active at the same time
const Backends = Tuple{Vararg{Backend}}
const current_backends = Parameter{Backends}(())
has_current_backend() = !isempty(current_backends())

export add_current_backend, delete_current_backend
add_current_backend(b::Backend) =
  current_backends(tuple(b, current_backends()...))
delete_current_backend(b::Backend) =
  current_backends(filter(!=(b), current_backends()))
 
# but for backward compatibility reasons, we might also select just one.
top_backend() =
  let bs = current_backends(),
      i = 0
    while isempty(bs) && i < 10
      @info("Waiting for a backend to be available...")
      sleep(5)
      bs = current_backends()
      i += 1
    end
    isempty(bs) ? throw(UndefinedBackendException()) : bs[1]
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

export purge_backends
purge_backends() =
  let bs = current_backends(),
      ok_bs = []
    for b in bs
      try
        with(current_backend, b) do
          get_view() # use a more efficient operation
        end
        push!(ok_bs, b)
      catch e
        @info("Backend $(b.name) is dead!")
      end
    end
    current_backends((ok_bs...,))
  end

# Variables with backend-specific values can be useful.
# Basically, they are dictionaries.
# but they also support a default value for the case
# where there is no backend-specific value available
export BackendParameter
struct BackendParameter
	value::IdDict{Type{<:Backend}, Any}
	BackendParameter(ps...) = new(IdDict{Type{<:Backend}, Any}(ps...))
	BackendParameter(p::BackendParameter) = new(copy(p.value))
end

(p::BackendParameter)(b::Backend=top_backend()) = error("Don't do this") #get(p.value, b, nothing)
(p::BackendParameter)(b::Backend, newvalue) = error("Don't do this") #p.value[b] = newvalue

(p::BackendParameter)(tb::Type{<:Backend}) = get(p.value, tb, nothing)
(p::BackendParameter)(tb::Type{<:Backend}, newvalue) = p.value[tb] = newvalue


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

b_curtain_wall_element(b::Backend, path, bottom, height, l_thickness, r_thickness, family) =
  b_wall(b, translate(path, vz(bottom)), height, l_thickness, r_thickness, family)

#@bdef curtain_wall(s, path, bottom, height, thickness, kind)

backend_fill(b, path) =
  backend_fill_curves(b, backend_stroke(b, path))

backend_frame_at(b, c, t) = throw(UndefinedBackendException())
backend_fill_curves(b, ids) = throw(UndefinedBackendException())

#@bdef fill(path::ClosedPathSequence)
backend_fill(b::Backend, path::ClosedPathSequence) =
  backend_fill_curves(b, map(path->backend_stroke(b, path), path.paths))


@bdef ground(level::Loc, color::RGB)

@bdef name()

@bdef b_zoom_extents()

export b_set_ground
b_set_ground(b::Backend, level, mat) =
  b_surface_regular_polygon(b, 16, z(level), 10000, 0, true, material_ref(b, mat))

b_realistic_sky(b::Backend, date, latitude, longitude, elevation, meridian, turbidity, sun) =
  b_realistic_sky(b, sun_pos(date, meridian, latitude, longitude)..., turbidity, sun)

# Rendering

export b_render_pathname, b_render_initial_setup, b_render_final_setup, b_setup_render, b_render_view

b_setup_render(b::Backend, kind) = kind

b_render_view(b::Backend, name) =
  let path = prepare_for_saving_file(b_render_pathname(b, name))
    b_render_final_setup(b, render_kind())
    b_render_and_save_view(b, path)
  end

#=
prepare_for_saving_file(path::String) =
  let p = normpath(path)
    mkpath(dirname(p))
    rm(p, force=true)
    isfile(p) ? # rm failed because file is locked
      let (base, ext) = splitext(path)
        prepare_for_saving_file(base*"_"*ext)
      end :
      p
  end
=#
prepare_for_saving_file(path::String) =
  let p = normpath(path)
    mkpath(dirname(p))
    try
      rm(p, force=true)
      p
    catch e
      if isa(e, Base.IOError)
        let (base, ext) = splitext(path)
          if ext == ".pdf"
            prepare_for_saving_file(base*"_"*ext)
          else
            throw(e)
          end
        end
      else
        throw(e)
      end
    end
  end

b_render_pathname(::Backend, name) = render_default_pathname(name)  
b_render_initial_setup(::Backend, kind) = kind
b_render_final_setup(::Backend, kind) = kind

@bdef b_render_and_save_view(path)

#######

b_right_cuboid(b::Backend, cb, width, height, h, mat) =
  b_box(b, add_xy(cb, -width/2, -height/2), width, height, h, mat)


backend_stroke(b::Backend, path::Union{OpenPathSequence,ClosedPathSequence}) =
  backend_stroke_unite(b, map(path->backend_stroke(b, path), path.paths))
backend_stroke(b::Backend, path::PathOps) =
  begin
      start, curr, refs = path.start, path.start, new_refs(b)
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
      refs = new_refs(b)
    for (w_seg_path, r_w_path, l_w_path) in zip(w_paths, r_w_paths, l_w_paths)
      let currlength = prevlength + path_length(w_seg_path),
          c_r_w_path = closed_path_for_height(r_w_path, w_height),
          c_l_w_path = closed_path_for_height(l_w_path, w_height)
        push!(refs, realize_pyramid_frustum(b, matright, matleft, matleft, c_l_w_path, c_r_w_path))
        prevlength = currlength
      end
    end
    refs
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

# Illustrations

@bdef(b_radius_illustration(c, r, c_text, r_text))
@bdef(b_arc_illustration(c, r, s, a, r_text, s_text, a_text))

#=
The backend might be in the same process as the frontend, or
it might be in a different process.
=#
export TargetType, LocalTarget, RemoteTarget
abstract type TargetType end
struct LocalTarget <: TargetType end
struct RemoteTarget <: TargetType end

# By default, we use a local target
target_type(::Type{<:Backend}) = LocalTarget()
target(b::T) where T = target(target_type(T), b)

target(::LocalTarget, b) = b.target
target(::RemoteTarget, b) = b.connection()

#=
Backends might not provide camera information. In that case
we need to provide it in the frontend.
=#
export ViewType, FrontendView, BackendView, view_type
abstract type ViewType end
struct FrontendView <: ViewType end
struct BackendView <: ViewType end

# By default, we use the backend view
view_type(::Type{<:Backend}) = BackendView()

b_set_view(b::T, camera, target, lens, aperture) where {T<:Backend} =
  b_set_view(view_type(T), b, camera, target, lens, aperture)

b_set_view_top(b::T) where {T<:Backend} = b_set_view_top(view_type(T), b)

b_get_view(b::T) where {T<:Backend} = b_get_view(view_type(T), b)

###############################################################
# For the frontend, we assume there is a property to store the 
# view details. A simple approach is to use a mutable struct

mutable struct View
  camera::Loc
  target::Loc
  lens::Real
  aperture::Real
  is_top_view::Bool
end

default_view() = View(xyz(10,10,10), xyz(0,0,0), 35, 22, false)
top_view() = View(xyz(0,0,10), xyz(0,0,0), 0, 0, true)

export View, default_view, top_view, b_get_view, b_set_view, b_set_view_top


b_set_view(::FrontendView, b, camera, target, lens, aperture) =
  begin
    b.view.camera = camera
    b.view.target = target
    b.view.lens = lens
	  b.view.aperture = aperture
	  b.view.is_top_view = norm(cross(target - camera, vz(1, world_cs))) < 1e-9  # aligned with Z?
  end

b_set_view_top(::FrontendView, b) =
  begin
    b.view.camera = z(1000)
    b.view.target = z(0)
    b.view.lens = 1000
	  b.view.is_top_view = true
  end

# For legacy reasons, we only return camera, target, and lens.
b_get_view(::FrontendView, b) =
  b.view.camera, b.view.target, b.view.lens

###############################################################
# For the backend, we always rely on the b_set_view function.

b_set_view_top(v::BackendView, b) =
  b_set_view(b, xyz(0.0,0.0,10), xyz(0.0,0.1,0.0), 50, 0)

###############################################################
# It might be useful to stop view update during batch processing
export with_batch_processing, b_start_batch_processing, b_stop_batch_processing

b_start_batch_processing(b::Backend) = nothing

b_stop_batch_processing(b::Backend) = nothing

with_batch_processing(f) =
  try
    foreach(b_start_batch_processing, current_backends())
    f()
  finally
    foreach(b_stop_batch_processing, current_backends())
  end


