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

############################################################
# First tier: everything is a triangle or a set of triangles
export b_trig, b_quad, b_ngon,
			 b_quad_strip, b_quad_strip_closed

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

b_surface_regular_polygon(b::Backend{K,T}, edges, c, r, angle, inscribed, mat) where {K,T} =
  b_ngon(b, regular_polygon_vertices(edges, c, r, angle, inscribed), c, false, mat)

b_surface_polygon(b::Backend{K,T}, ps, mat) where {K,T} =
  # This only works for convex polygons
  b_ngon(b, ps, trig_center(ps[1], ps[2], ps[3]), false, mat)

@bdef(b_surface_polygon_with_holes(ps, qss, mat))

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
    path_vertices.(inner_paths(region)),
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

##################################################################
# Materials
#=
In most cases, the material already exists in the backend and is
accessed by name.
=#
export b_get_material, b_new_material

@bdef(b_get_material(path))

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
    b_wall(b, w_path, w_height, l_thickness, r_thickness, family_materials(family)[1:3])

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
  	BackendParameter(ps::Pair{K}...) where {K<:Backend} = new(IdDict{Backend, Any}(ps...))
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

@bdef fill(path::ClosedPolygonalPath)
@bdef fill(path::ClosedSplinePath)
@bdef fill(path::RectangularPath)
@bdef ground(level::Loc, color::RGB)
@bdef ieslight(file::String, loc::Loc, dir::Vec, alpha::Real, beta::Real, gamma::Real)
@bdef line(vs::Locs)
#@bdef loft_curve_point(profile::Shape, point::Shape)
#@bdef loft_points(profiles::Shapes, rails::Shapes, ruled::Bool, closed::Bool)
#@bdef loft_surface_point(profile::Shape, point::Shape)
#@bdef loft_surfaces(profiles::Shapes, rails::Shapes, ruled::Bool, closed::Bool)
#@bdef map_division(f::Function, s::Shape1D, n::Int)
#@bdef map_division(f::Function, s::Shape2D, nu::Int, nv::Int)
#@bdef map_division(f::Function, s::SurfaceGrid, nu::Int, nv::Int)
@bdef name()
#@bdef panel(bot::Locs, top::Locs, family::PanelFamily)
@bdef pointlight(loc::Loc, color::RGB, range::Real, intensity::Real)
@bdef polygon(vs::Locs)

#@bdef pyramid(bs::Locs, t::Loc)
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
@bdef b_realistic_sky(date, latitude, longitude, meridian, turbidity, withsun)

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


@bdef spotlight(loc::Loc, dir::Vec, hotspot::Real, falloff::Real)


backend_stroke(b::Backend, path::RectangularPath) =
  let c = path.corner,
      dx = path.dx,
      dy = path.dy
    backend_stroke_line(b, (c, add_x(c, dx), add_xy(c, dx, dy), add_y(c, dy), c))
  end

backend_stroke(b::Backend, path::OpenPolygonalPath) =
	backend_stroke_line(b, path.vertices)

backend_stroke(b::Backend, path::ClosedPolygonalPath) =
  backend_stroke_line(b, [path.vertices...,path.vertices[1]])

backend_stroke(b::Backend, path::Union{OpenPathSequence,ClosedPathSequence}) =
  backend_stroke_unite(b, map(path->backend_stroke(b, path), path.paths))

@bdef stroke(path::ArcPath)
@bdef stroke(path::CircularPath)

#@bdef stroke(path::ClosedSplinePath)
#@bdef stroke(path::OpenSplinePath)

#@bdef stroke(path::PathOps)
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

#@bdef stroke(path::PathSet)
backend_stroke(b::Backend, path::PathSet) =
    for p in path.paths
        backend_stroke(b, p)
    end

@bdef stroke_arc(center::Loc, radius::Real, start_angle::Real, amplitude::Real)

#@bdef stroke_color(path::Path, color::RGB)
# By default, we ignore the color
backend_stroke_color(backend::Backend, path::Path, color::RGB) =
  backend_stroke(backend, path)

@bdef stroke_line(vs)
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









#=
Backends might use different communication mechanisms, e.g., sockets, COM,
RMI, etc. We start by defining socket-based communication.
=#

struct SocketBackend{K,T} <: Backend{K,T}
  connection::LazyParameter{TCPSocket}
  remote::NamedTuple
end

create_backend_connection(backend::AbstractString, port::Integer) =
  for i in 1:10
    try
      return connect(port)
    catch e
      @info("Please, start/restart $(backend).")
      sleep(8)
      if i == 9
        throw(e)
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
