#=

Backends need to store references to resources such as shapes, materials, layers, families, etc.
Moreover, some of these resources live longer than others. For example, a shape can be deleted 
but its material should be preserved, as it can be referenced from other shapes.
To handle this, these references are going to be classified according to their type.
=#

struct References{K,T}
  shapes::Dict{Shape,GenericRef{K,T}}
  materials::Dict{Material,GenericRef{K,T}}
  layers::Dict{Layer,GenericRef{K,T}}
  annotations::Dict{Annotation,GenericRef{K,T}}
  families::Dict{Family,GenericRef{K,T}}
  levels::Dict{Level,GenericRef{K,T}}
  References{K,T}() where {K,T} =
    new{K,T}(Dict{Shape,GenericRef{K,T}}(), 
             Dict{Material,GenericRef{K,T}}(),
             Dict{Layer,GenericRef{K,T}}(),
             Dict{Annotation,GenericRef{K,T}}(),
             Dict{Family,GenericRef{K,T}}(),
             Dict{Level,GenericRef{K,T}}())
end

# Constraint: all backends must have a References field named refs
export shape_refs_storage, material_refs_storage, layer_refs_storage, annotation_refs_storage, family_refs_storage, level_refs_storage, refs_storage
shape_refs_storage(b::Backend) = b.refs.shapes
material_refs_storage(b::Backend) = b.refs.materials
layer_refs_storage(b::Backend) = b.refs.layers
annotation_refs_storage(b::Backend) = b.refs.annotations
family_refs_storage(b::Backend) = b.refs.families
level_refs_storage(b::Backend) = b.refs.levels

# To dispatch on the type of proxy
refs_storage(b::Backend, p::Shape) = shape_refs_storage(b)
refs_storage(b::Backend, p::Material) = material_refs_storage(b)
refs_storage(b::Backend, p::Layer) = layer_refs_storage(b)
refs_storage(b::Backend, p::Annotation) = annotation_refs_storage(b)
refs_storage(b::Backend, p::Family) = family_refs_storage(b)
refs_storage(b::Backend, p::Level) = level_refs_storage(b)

# Get the shape from the reference
export get_or_create_shape_from_ref_value

get_or_create_from_ref_value(b::Backend, r, storage, create) =
  let dict = storage(b)
    for (proxy, ref) in dict
      if contains(ref, r)
        return proxy
      end
    end
    let new_sh = 
          with(current_backends, ()) do # To avoid realizing shapes
            create(b, r)
          end
      ref!(b, new_sh, r)
      new_sh
    end
  end

get_or_create_shape_from_ref_value(b::Backend, r) =
  get_or_create_from_ref_value(b, r, shape_refs_storage, b_create_shape_from_ref_value)

get_or_create_layer_from_ref_value(b::Backend, r) =
  get_or_create_from_ref_value(b, r, layer_refs_storage, b_create_layer_from_ref_value)

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
We start by defining socket-based communication.
=#

mutable struct SocketBackend{K,T} <: RemoteBackend{K,T}
  name::String
  port::Integer
  connection::Union{Missing,TCPSocket}
  static_remote::NamedTuple
  remote::NamedTuple
  transaction::Parameter{Transaction}
  refs::References{K,T}
end

SocketBackend{K,T}(name, port, remote) where {K,T} =
  SocketBackend{K,T}(name, port, missing, remote)

SocketBackend{K,T}(name, port, connection, remote) where {K,T} =
  SocketBackend{K,T}(name, port, connection, remote, remote_functions(remote), Parameter{Transaction}(AutoCommitTransaction()), References{K,T}())

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
  websocket::HTTP.WebSockets.WebSocket
  buffer::IOBuffer
  WebSocketConnection(server, router, websocket) =
    new(server, router, websocket, IOBuffer(UInt8[], read=true, write=true))
end

# Sending stuff is trivial.

send(c::WebSocketConnection, buffer) = 
  HTTP.WebSockets.send(c.websocket, take!(buffer))

receive(c::WebSocketConnection) = 
  let bytes = HTTP.WebSockets.receive(c.websocket)
    #println("Received data of length $(length(bytes))")
    take!(c.buffer) # Clear the buffer
    write(c.buffer, bytes)
    seekstart(c.buffer)
    c.buffer
  end

# Finally, the backend itself:
mutable struct WebSocketBackend{K,T} <: RemoteBackend{K,T}
  name::String
  host::String
  port::Int
  connection::Union{WebSocketConnection,Missing}
  static_remote::NamedTuple
  remote::NamedTuple
  transaction::Parameter{Transaction}
  refs::References{K,T}
  handlers::Vector{Function}

  WebSocketBackend{K,T}(name, host, port, remote) where {K,T} =
    new(name, host, port,
        missing, remote, remote_functions(remote),
        Parameter{Transaction}(AutoCommitTransaction()), References{K,T}(), Function[])
end

#=
Given that this is a server, we can also allow clients to make requests by using,
e.g., the fetch API. To that end, we need to be able to add request handlers.
Be careful about deadlocks and such.
=#
export register_handler, register_http_handler, register_websocket_handler

const using_http_requests = Parameter(false)

register_handler(c::WebSocketBackend{K,T}, target, handler) where {K,T} =
  if using_http_requests()
    let request_str = "/api/"*randstring()
      register_http_handler(c, request_str, handler)
      request_str
    end
  else
    register_websocket_handler(c, target, handler)
  end

register_websocket_handler(c::WebSocketBackend{K,T}, target, handler) where {K,T} =
  (push!(c.handlers, handler); length(c.handlers))

call_handler(c::WebSocketBackend{K,T}, target, args...) where {K,T} =
  let handler = c.handlers[target]
    handler(args...)
  end

export process_requests
#=
Note that we cannot process client requests while server requests are being processed.
This means that the handler being called must do its job and return as soon as possible, 
so that other requests can be processed.
=#
process_requests(c::WebSocketBackend{K,T}) where {K,T} =
  let namespace = c.static_remote[1].namespace # An awful way of retrieving the namespace
    while true
      let buf = receive(c.connection),
          target = decode(namespace, Val(:size), buf),
          args_len = decode(namespace, Val(:size), buf),
          args = Any[]
        for i in 1:args_len
          push!(args, decode(namespace, Val(:Any), buf))
        end
        println("Calling '$target' with $(args)")
        let result_code = -1 # Default: no result
          try
            call_handler(c, target, args...)
          catch e
            println("Error occurred while processing request '$target': $e")
            showerror(stdout, e, catch_backtrace())
            result_code = -2 # Error
          finally
            # Signal that the request has been processed
            let buf = IOBuffer()
              encode(namespace, Val(:int), buf, result_code)
              send(c.connection, buf)
            end
          end
        end
      end
    end
  end

#=
Another option is to use HTTP handlers.
=#
export register_http_handler

# Backends encode the parameters as query parameters p0, p1, p2, ...
# e.g., /api/handler?p0=val0&p1=val1&...
request_parameters(req) =
  let dict = HTTP.queryparams(HTTP.URI(HTTP.Messages.getfield(req, :target)))
    map(i -> haskey(dict, "p$i") ? dict["p$i"] : error("Missing parameter p$i"), 0:(length(dict)-1))
  end

register_http_handler(c::WebSocketBackend{K,T}, target, handler) where {K,T} =
  HTTP.register!(c.connection.router, "GET", target, req -> (handler(request_parameters(req)...); HTTP.Response(200, "0")))

#=
It is going to be useful to have an algebra of handlers.
=#
export action_handler, sequence_handler, wrapper_handler, update_parameter_handler

action_handler(f) =
  (args...) -> f()

sequence_handler(handlers...) = 
  (args...) -> for handler in handlers
                  handler(args...)
               end

wrapper_handler(wrapper, f) =
  (args...) -> wrapper(f, args...)

#=
A particular case of handler is the one that updates a parameter.
=#

update_parameter_handler(parameter::Parameter{T}) where {T} = 
  using_http_requests() ?
    (str) -> parameter(parse(T, str)) :
    (val) -> parameter(convert(T, val))


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

export b_layer, b_current_layer_ref,
       b_all_shapes_in_layer, b_delete_all_shapes_in_layer

# Default implementation assumes that backends have properties for current_layer and layers (a dict)
b_layer(b::Backend, name, active, color) = BasicLayer(name, active, color)
b_current_layer_ref(b::Backend) = b.current_layer
b_current_layer_ref(b::Backend, layer) = b.current_layer = layer
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

abstract type LocalBackend{K,T} <: Backend{K,T} end
@kwdef mutable struct IOBackend{K,T,E} <: LocalBackend{K,T}
  shapes::Shapes=Shape[]
  transaction::Parameter{Transaction}=Parameter{Transaction}(ManualCommitTransaction())
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

KhepriBase.b_delete_all_shape_refs(b::IOBackend) =
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
