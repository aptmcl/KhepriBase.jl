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

show(io::IO, b::Backend{K,T}) where {K,T} = print(io, backend_name(b))

#=
Operations that rely on a backend need to have a backend selected and will
generate an exception if there is none.
=#

struct UndefinedBackendException <: Exception end
showerror(io::IO, e::UndefinedBackendException) = print(io, "No current backend.")

# We can have several backends active at the same time
const current_backends = Parameter{Tuple{Vararg{Backend}}}(())
# but for backward compatibility reasons, we might also select just one.
current_backend() =
	let bs = current_backends()
		isempty(bs) ?
			throw(UndefinedBackendException()) :
			bs[1]
	end
current_backend(b::Backend) = current_backends((b,))
has_current_backend() = !isempty(current_backends())

# Backends need to implement operations or an exception is triggered
struct UnimplementedBackendOperationException <: Exception
	backend
	operation
end
showerror(io::IO, e::UnimplementedBackendOperationException) =
	print(io, "Operation $(e.operation) is not implemented for backend $(e.backend).")

# backend define
macro bdef(name_params)
  name = name_params.args[1]
  params = esc.(name_params.args[2:end])
  backend_name = esc(Symbol("backend_$name"))
  quote
    export $(backend_name)
    $(backend_name)(b::Backend, $(params...)) =
	  throw(UnimplementedBackendOperationException(b, $(backend_name)))
  end
end
# backend call
macro bcall(backend, name_args)
  name = name_args.args[1]
  args = name_args.args[2:end]
  backend_name = esc(Symbol("backend_$name"))
  quote
    $(backend_name)($(esc(backend)), $(esc.(args)...))
  end
end

# backends call
macro bscall(backends, name_args)
	name = name_args.args[1]
  args = name_args.args[2:end]
	backend_name = esc(Symbol("backend_$name"))
  quote
    for backend in $(esc(backends))
	  $(backend_name)(backend, $(esc.(args)...))
    end
  end
end

# current backends call
macro cbscall(name_args)
	name = name_args.args[1]
  args = name_args.args[2:end]
	backend_name = esc(Symbol("backend_$name"))
  quote
    for backend in current_backends()
	  $(backend_name)(backend, $(esc.(args)...))
    end
  end
end

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

@bdef create_layer(name::String, active::Bool, color::RGB)

#@bdef curtain_wall(s, path::Path, bottom::Real, height::Real, l_thickness::Real, r_thickness::Real, kind::Symbol)
backend_curtain_wall(b::Backend, s, path::Path, bottom::Real, height::Real, l_thickness::Real, r_thickness::Real, kind::Symbol) =
  let family = getproperty(s.family, kind),
      mat = get_material(b, family_ref(b, family))
    with_family_in_layer(b, family) do
      backend_wall_with_materials(b, translate(path, vz(bottom)), height, l_thickness, r_thickness, mat, mat)
    end
  end
@bdef curtain_wall(s, path::Path, bottom::Real, height::Real, thickness::Real, kind::Symbol)


@bdef sphere(c::Loc, r::Real)
#@bdef sphere(c::Loc, r::Real, material, layer)
@bdef cylinder(cb::Loc, r::Real, h::Real)

#@bdef cylinder(cb::Loc, r::Real, h::Real, material)
backend_cylinder(b::Backend, c::Loc, r::Real, h::Real, material) =
  backend_cylinder(b, c, r, h)



#@bdef delete_shapes(shapes::Shapes)
#@bdef extrusion(p::Point, v::Vec)
#@bdef extrusion(s::Shape, v::Vec)
#@bdef family(family::Family)

#@bdef fill(path)
backend_fill(b, path) =
  backend_fill_curves(b, backend_stroke(b, path))

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

@bdef realistic_sky(altitude, azimuth, turbidity, withsun)
@bdef realistic_sky(date, latitude, longitude, meridian, turbidity, withsun)

#@bdef rectangular_table(c, angle, family)
backend_rectangular_table(b::Backend, p, angle, f) =
  realize_table(b, get_material(b, family_ref(b, f)),
                loc_from_o_phi(p, angle), f.length, f.width, f.height, f.top_thickness, f.leg_thickness)

realize_table(b::Backend, mat, p::Loc, length::Real, width::Real, height::Real,
              top_thickness::Real, leg_thickness::Real) =
  let dx = length/2,
      dy = width/2,
      leg_x = dx - leg_thickness/2,
      leg_y = dy - leg_thickness/2,
      c = add_xy(p, -dx, -dy),
      table_top = realize_box(b, mat, add_z(c, height - top_thickness), length, width, top_thickness),
      pts = add_xy.(add_xy.(p, [+leg_x, +leg_x, -leg_x, -leg_x], [-leg_y, +leg_y, +leg_y, -leg_y]), -leg_thickness/2, -leg_thickness/2),
      legs = [realize_box(b, mat, pt, leg_thickness, leg_thickness, height - top_thickness) for pt in pts]
    [ensure_ref(b, r) for r in [table_top, legs...]]
  end

#@bdef rectangular_table_and_chairs(c, angle, family)
backend_rectangular_table_and_chairs(b::Backend, p, angle, f) =
  let tf = f.table_family,
      cf = f.chair_family,
      tmat = get_material(b, realize(b, tf).material),
      cmat = get_material(b, realize(b, cf).material)
    realize_table_and_chairs(b,
      loc_from_o_phi(p, angle),
      p->realize_table(b, tmat, p, tf.length, tf.width, tf.height, tf.top_thickness, tf.leg_thickness),
      p->realize_chair(b, cmat, p, cf.length, cf.width, cf.height, cf.seat_height, cf.thickness),
      tf.width,
      tf.height,
      f.chairs_top,
      f.chairs_bottom,
      f.chairs_right,
      f.chairs_left,
      f.spacing)
  end

realize_table_and_chairs(b::Backend, p::Loc, table::Function, chair::Function,
                         table_length::Real, table_width::Real,
                         chairs_on_top::Int, chairs_on_bottom::Int,
                         chairs_on_right::Int, chairs_on_left::Int,
                         spacing::Real) =
  let dx = table_length/2,
      dy = table_width/2,
      row(p, angle, n) = [loc_from_o_phi(add_pol(p, i*spacing, angle), angle+pi/2) for i in 0:n-1],
      centered_row(p, angle, n) = row(add_pol(p, -spacing*(n-1)/2, angle), angle, n)
    vcat(table(p),
         chair.(centered_row(add_x(p, -dx), -pi/2, chairs_on_bottom))...,
         chair.(centered_row(add_x(p, +dx), +pi/2, chairs_on_top))...,
         chair.(centered_row(add_y(p, +dy), -pi, chairs_on_right))...,
         chair.(centered_row(add_y(p, -dy), 0, chairs_on_left))...)
  end

@bdef render_view(path::String)

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
@bdef truss_analysis(load::Vec)


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
