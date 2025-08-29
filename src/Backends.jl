#=
Backends might use different communication mechanisms, e.g., sockets, COM,
RMI, etc.
=#

abstract type RemoteBackend{K,T} <: Backend{K,T} end
# We assume there is a property for the name
backend_name(b::RemoteBackend) = b.name


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

#=
There is a protocol to reset the connection
It also envolves reseting several resources that are affected such materials, layers, families, etc.
=#

reset_backend(b::RemoteBackend) =
  let c = b.connection
    for f in b.remote
      reset_opcode(f)
    end
    b.connection = missing
    close(c) # This might err, so it goes last
  end

export RemoteBackend, before_connecting, after_connecting, start_connection, failed_connecting, retry_connecting
before_connecting(::RemoteBackend) = nothing
after_connecting(::RemoteBackend) = nothing
failed_connecting(b::RemoteBackend) =
  @error("Couldn't connect to $(b.name).")
retry_connecting(b::RemoteBackend) =
  (@info("Please, start/restart $(b.name)."); sleep(8))

#=
Backends might have resources that are shared by many shapes, such as materials, layers, families, etc.
Moreover, these resources tend to live longer than the shapes themselves.
This means that shapes can be deleted without necessarily affecting the underlying resources.
=#

#=
struct References{T}
  material_refs::Dict{Material,T}
  shape_refs::Dict{Shape,T}
  annotation_refs::Dict{Shape,T}
  References{T}() where {T} =
    new{T}(Dict{Material,T}(), Dict{Shape,T}(), Dict{Shape,T}())
end
=#

const References{K,T} = IdDict{Proxy,GenericRef{K,T}}
#=
We start by defining socket-based communication.
=#

mutable struct SocketBackend{K,T} <: RemoteBackend{K,T}
  name::String
  port::Integer
  connection::Union{Missing,TCPSocket}
  static_remote::NamedTuple
  remote::NamedTuple
  refs::References{K,T}
end

SocketBackend{K,T}(name, port, remote) where {K,T} =
  SocketBackend{K,T}(name, port, missing, remote)

SocketBackend{K,T}(name, port, connection, remote) where {K,T} =
  SocketBackend{K,T}(name, port, connection, remote, remote_functions(remote), References{K,T}())

# AML: replace with the Base.retry function?
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

#=
In some cases, it might be useful to have multiple backends running at the same time, controlled by a single frontend.
This entails treating the frontend as a server and each backend as a client.
Another option is simply to have a socket server running on the Julia side, that waits for connection requests.
After connecting, we can expect a string that specifies the backend to instantiate and then we simply initialize it.
=#

const default_khepri_server_port = Parameter(12345)
export default_khepri_server_port
const khepri_server_task = Parameter{Union{Nothing,Task}}(nothing)

export ensure_khepri_server_running
ensure_khepri_server_running() =
  if isnothing(khepri_server_task())
    khepri_server_task(errormonitor(Threads.@spawn start_khepri_server()))
  end


const backend_init_map = Dict{String, Function}()
add_client_backend_init_function(name, init_function) = begin
  backend_init_map[name] = init_function
  ensure_khepri_server_running()
end
get_client_backend_init_function(name) =
  get(backend_init_map, name) do 
    error("Requested backend '$name' is not available!")
  end
export add_client_backend_init_function

#=
Upon connection and initialization, it might be
necessary to place some initial geometry on each
client that connects.
To that end, we will use the main generic function,
as a metaphor for the startup function that is the
entry program of C, Java, etc, programs.
=#
export main
main(b::Backend) = nothing


export start_khepri_server
start_khepri_server(port=default_khepri_server_port()) =
  let server = listen(port)
    #println("Listening on port $port")
    while true
      println("Waiting for a new client")
      let conn = accept(server)
        println("Client connected")
        let backend_name = decode(Val(:CS), Val(:string), conn),
            init_func = get_client_backend_init_function(backend_name),
            backend = invokelatest(init_func, conn)
          invokelatest(before_connecting, backend)
    	    invokelatest(after_connecting, backend)
          add_current_backend(backend)
          invokelatest(main, backend)
        end
      end
    end
  end

#=
Another option is to use WebSockets.
=#

struct WebSocketConnection <: IO
  server::HTTP.Server
  router::HTTP.Router
  connections::Set{HTTP.WebSockets.WebSocket}
end

for_connection(f, s) = begin
  for websocket in s.connections
    if HTTP.WebSockets.isclosed(websocket)
      @warn("Removing closed websocket connection!")
      delete!(s.connections, websocket)
    else
      try
        return f(websocket)
      catch e
        if isa(e, Base.IOError)
          delete!(s.connections, websocket)
        else
          rethrow(e)
        end
      end
    end
  end
  error("There are no open WebSocket connections to write to!")
end

from_connection(f, s) = begin
  for websocket in s.connections
    if HTTP.WebSockets.isclosed(websocket)
      @warn("Removing closed websocket connection!")
      delete!(s.connections, websocket)
    else
      try
        return f(websocket)
      catch e
        if isa(e, Base.IOError)
          delete!(s.connections, websocket)
        else
          rethrow(e)
        end
      end
    end
  end
  error("There are no open WebSocket connections to read from!")
end

broadcast_write(c, data) =
  for_connection(c) do websocket
    HTTP.WebSockets.send(websocket, data)
  end

# These are needed to avoid ambiguous dispatch
Base.write(c::WebSocketConnection, data::Union{Float16,Float32,Float64,Int128,Int16,Int32,Int64,UInt128,UInt16,UInt32,UInt64}) = broadcast_write(c, data)
Base.write(c::WebSocketConnection, data::Array) = broadcast_write(c, data)
Base.write(c::WebSocketConnection, data::Union{SubString{String},String}) = broadcast_write(c, data)

Base.read(c::WebSocketConnection, T::Union{Type{Float16},Type{Float32},Type{Float64},Type{Int128},Type{Int16},Type{Int32},Type{Int64},Type{UInt128},Type{UInt16},Type{UInt32},Type{UInt64}}) =
  let ans = from_connection(HTTP.WebSockets.receive, c),
      io = IOBuffer(ans)
    read(io, T)
  end

# Why can't the following one be included in the previous union?
Base.read(c::WebSocketConnection, T::Type{UInt8}) =
  let ans = from_connection(HTTP.WebSockets.receive, c),
      io = IOBuffer(ans)
    read(io, T)
  end
Base.close(c::WebSocketConnection) =
  begin
    close.(c.connections)
    empty!(c.connections)
    if isopen(c.server)
      HTTP.close(c.server)
    end
  end

# Finally, the backend itself:
mutable struct WebSocketBackend{K,T} <: RemoteBackend{K,T}
  name::String
  host::String
  port::Int
  connection::Union{WebSocketConnection,Missing}
  static_remote::NamedTuple
  remote::NamedTuple
  refs::References{K,T}
end

WebSocketBackend{K,T}(name, host, port, remote) where {K,T} =
  WebSocketBackend{K,T}(name, host, port, missing, remote, remote_functions(remote), References{K,T}())

#=
Given that this is a server, we can also allow clients to make requests by using, 
e.g., the fetch API. To that end, we need to be able to add request handlers.
Be careful about deadlocks and such.
=#
export register_handler

register_handler(c::WebSocketBackend{K,T}, method, path, handler) where {K,T} =
  HTTP.register!(c.connection.router, method, path, handler)

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
b_layer(b::Backend, name, active, color) = BasicLayer(name, active, color)
b_current_layer(b::Backend) = b.current_layer
b_current_layer(b::Backend, layer) = b.current_layer = layer
b_all_shapes_in_layer(b::Backend, layer) = b.layers[layer]
b_delete_all_shapes_in_layer(b::Backend, layer) = b_delete_shapes(b_all_shapes_in_layer(b, layer))


################################################################
# Not all backends support all stuff. Some of it might need to be supported
# by ourselves. Render environment is another example.
export RenderEnvironment, RealisticSkyEnvironment, ClayEnvironment

abstract type RenderEnvironment end
struct RealisticSkyEnvironment <: RenderEnvironment
  turbidity::Real
  sun::Bool
end

struct ClayEnvironment <: RenderEnvironment
end

export GeographicLocation
struct GeographicLocation
  latitude::Real
  longitude::Real
  elevation::Real
  meridian::Real
end

###############################################################################
# Another backend option is to save all shapes locally and then generate, e.g., a
# file-based description.

abstract type LocalBackend{K,T} <: LazyBackend{K,T} end
@kwdef mutable struct IOBackend{K,T,E} <: LocalBackend{K,T}
  shapes::Shapes=Shape[]
  refs::References{K,T}=References{K,T}()
  current_layer::Union{Nothing,AbstractLayer}=nothing
  layers::Dict{AbstractLayer,Vector{Shape}}=Dict{AbstractLayer,Vector{Shape}}()
  date::DateTime=DateTime(2020, 9, 21, 10, 0, 0)
  place::GeographicLocation=GeographicLocation(39, 9, 0, 0)
  render_env::RenderEnvironment=RealisticSkyEnvironment(5, true)
  ground_level::Float64=0.0
  ground_material::Union{Nothing,Material}=nothing
  view::View=default_view()
  cached::Bool=false # REMOVE?
  io::IO=IOBuffer()
  extra::E=E()
end

view_type(::Type{<:LocalBackend}) = FrontendView()

connection(b::IOBackend) = b.io

save_shape!(b::IOBackend, s::Shape) =
  begin
    push!(b.shapes, s)
    if !isnothing(b.current_layer)
      push!(get!(b.layers, b.current_layer, Shape[]), s)
    end
    s
  end

export realize_shapes
realize_shapes(b::IOBackend) =
  for s in b.shapes
    reset_ref(b, s)
	  force_realize(b, s)
  end

export used_materials
used_materials(b::IOBackend) =
  let materials=Set{Material}()
	for s in b.shapes
	  for m in used_materials(s)
  	  	push!(materials, m)
	  end
    end
	if !isnothing(b.ground_material)
	  push!(materials, b.ground_material)
	end
	materials
  end

KhepriBase.b_delete_all_refs(b::IOBackend) =
  begin
    empty!(b.shapes)
    for ss in values(b.layers)
      empty!(ss)
    end
    empty!(b.refs)
    nothing
  end

KhepriBase.b_delete_shape(b::IOBackend, shape::Shape) =
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
KhepriBase.b_all_shapes(b::IOBackend) = b.shapes
KhepriBase.b_all_shapes_in_layer(b::IOBackend, layer) = b.layers[layer]

KhepriBase.b_realistic_sky(b::IOBackend, date, latitude, longitude, elevation, meridian, turbidity, sun) =
  begin
	  b.date = date
    b.place = GeographicLocation(latitude, longitude, elevation, meridian)
	  b.render_env = RealisticSkyEnvironment(turbidity, sun)
  end

b_set_ground(b::IOBackend, level, mat) =
  begin
	  b.ground_level=level
	  b.ground_material=mat
  end
