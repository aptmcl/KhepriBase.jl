#=
Backends might use different communication mechanisms, e.g., sockets, COM,
RMI, etc.
=#

abstract type RemoteBackend{K,T} <: Backend{K,T} end

# There is a protocol for retrieving the connection
connection(b::RemoteBackend) =
  begin
    if ismissing(b.connection)
	  before_connecting(b)
      b.connection = start_connection(b)
	  after_connecting(b)
    end
	b.connection
  end

export RemoteBackend, before_connecting, after_connecting, start_connection, failed_connecting, retry_connecting
before_connecting(b::RemoteBackend) = nothing
after_connecting(b::RemoteBackend) = nothing
failed_connecting(b::RemoteBackend) =
  @error("Couldn't connect to $(b.name).")
retry_connecting(b::RemoteBackend) =
  (@info("Please, start/restart $(b.name)."); sleep(8))

#=
We start by defining socket-based communication.
=#

mutable struct SocketBackend{K,T} <: RemoteBackend{K,T}
  name::String
  port::Integer
  connection::Union{Missing,TCPSocket}
  remote::NamedTuple
end

SocketBackend{K,T}(name::AbstractString, port::Integer, remote::NamedTuple) where {K,T} =
  SocketBackend{K,T}(name, port, missing, remote)

backend_name(b::SocketBackend) = b.name

reset_backend(b::SocketBackend) =
  begin
    for f in b.remote
      reset_opcode(f)
    end
    close(b.connection)
    b.connection = missing
  end

start_connection(b::SocketBackend) =
  let attempts = 10
    for i in 1:attempts
      try
        return connect(b.port)
      catch e
        if i == attempts
		  failed_connecting(b)
        else
          retry_connecting(b)
        end
      end
    end
  end

# To simplify remote calls
macro remote(b, call)
  let op = call.args[1],
      args = map(esc, call.args[2:end]),
      b = esc(b)
    :(call_remote(getfield(getfield($(b), :remote), $(QuoteNode(op))), connection($(b)), $(args...)))
  end
end

macro get_remote(b, op)
  let b = esc(b)
    :(getfield(getfield($(b), :remote), $(QuoteNode(op))))
  end
end

################################################################
# Not all backends support all stuff. Some of it might need to be supported
# by ourselves. Layers are one example.

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

###############################################################
# We assume there is a property to store the view details
# A simple approach is to use a mutable struct
mutable struct View
  camera::Loc
  target::Loc
  lens::Real
  aperture::Real
end
default_view() = View(xyz(10,10,10), xyz(0,0,0), 35, 22)

export View, default_view, b_get_view, b_set_view, b_set_view_top

b_set_view(b::Backend, camera, target, lens, aperture) =
  begin
    b.view.camera = camera
    b.view.target = target
    b.view.lens = lens
	b.view.aperture = aperture
  end
b_set_view_top(b::Backend) =
  begin
    b.view.camera = z(1000)
    b.view.target = z(0)
    b.view.lens = 1000
  end
# For legacy reasons, we only return camera, target, and lens.
b_get_view(b::Backend) =
  b.view.camera, b.view.target, b.view.lens

################################################################
# Not all backends support all stuff. Some of it might need to be supported
# by ourselves. Render environment is another example.
export RenderEnvironment, RealisticSkyEnvironment, ClayEnvironment, default_render_environment

abstract type RenderEnvironment end
struct RealisticSkyEnvironment <: RenderEnvironment
  sun_altitude::Real
  sun_azimuth::Real
  turbidity::Real
  sun::Bool
end

struct ClayEnvironment <: RenderEnvironment
end

struct GeographicLocation
  latitude::Real
  longitude::Real
  elevation::Real
  meridian::Real
end

###############################################################################
#Another backend option is to save all shapes locally and then generate, e.g., a
#file-based description.

abstract type LocalBackend{K,T} <: LazyBackend{K,T} end
@kwdef mutable struct IOBufferBackend{K,T,E} <: LocalBackend{K,T}
  shapes::Shapes=Shape[]
  current_layer::Union{Nothing,AbstractLayer}=nothing
  layers::Dict{AbstractLayer,Vector{Shape}}=Dict{AbstractLayer,Vector{Shape}}()
  date::DateTime=DateTime(2020, 9, 21, 10, 0, 0)
  place::GeographicLocation=GeographicLocation(39, 9, 0, 0)
  render_env::RenderEnvironment=RealisticSkyEnvironment(90, 0, 5, true)
  ground_level::Float64=0.0
  ground_material::Union{Nothing,Material}=nothing
  view::View=default_view()
  cached::Bool=false
  buffer::IOBuffer=IOBuffer()
  extra::E=E()
end

connection(b::IOBufferBackend) = b.buffer

save_shape!(b::IOBufferBackend, s::Shape) =
  begin
    push!(b.shapes, s)
    if !isnothing(b.current_layer)
      push!(get!(b.layers, b.current_layer, Shape[]), s)
    end
	b.cached = false
    s
  end

export realize_shapes
realize_shapes(b::IOBufferBackend) =
  if ! b.cached
    take!(b.buffer)
    for s in b.shapes
      reset_ref(b, s)
  	  force_realize(b, s)
    end
    b.cached = true
  end

export used_materials
used_materials(b::IOBufferBackend) =
  let materials=Set{Material}()
	for s in b.shapes
  	  push!(materials, s.material)
    end
	if !isnothing(b.ground_material)
	  push!(materials, b.ground_material)
	end
	materials
  end

KhepriBase.b_delete_all_shapes(b::IOBufferBackend) =
  begin
    empty!(b.shapes)
    for ss in values(b.layers)
      empty!(ss)
    end
    b.cached = false
    nothing
  end

KhepriBase.b_delete_shape(b::IOBufferBackend, shape::Shape) =
  let f(s) = s!== shape
    filter!(f, b.shapes)
    for ss in values(b.layers)
      filter!(f, ss)
    end
  end

#=
KhepriBase.b_delete_shapes(b::POVRay, shapes::Shapes) =
  let f(s) = isnothing(findfirst(s1->s1===s, shapes))
    filter!(f, b.shapes)
    for ss in values(b.layers)
      filter!(f, ss)
    end
  end
=#

# HACK: This should be filtered on the plugin, not here.
KhepriBase.b_all_shapes(b::IOBufferBackend) = b.shapes
KhepriBase.b_all_shapes_in_layer(b::IOBufferBackend, layer) = b.layers[layer]

b_set_time_place(b::IOBufferBackend, date, latitude, longitude, elevation, meridian) =
  begin
	b.date = date
    b.place = GeographicLocation(latitude, longitude, elevation, meridian)
  end

b_set_ground(b::IOBufferBackend, level, mat) =
  begin
	b.ground_level=level
	b.ground_material=mat
  end

b_clay_model(b::IOBufferBackend) =
  b.render_env = ClayEnvironment()
