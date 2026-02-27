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

Base.empty!(r::References) =
  begin
    empty!(r.shapes)
    empty!(r.materials)
    empty!(r.layers)
    empty!(r.annotations)
    empty!(r.families)
    empty!(r.levels)
    r
  end

# Constraint: all backends must have a References field named refs
export References
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

const socket_backend_init_map = Dict{String, Function}()
const socket_backend_init_lock = ReentrantLock()
add_socket_backend_init_function(name, init_function) = begin
  lock(socket_backend_init_lock) do
    socket_backend_init_map[name] = init_function
  end
  ensure_khepri_socket_server_running()
end
get_socket_backend_init_function(name) =
  lock(socket_backend_init_lock) do
    get(socket_backend_init_map, name) do
      error("Requested socket backend '$name' is not available!")
    end
  end
export add_socket_backend_init_function

# The server code

const default_khepri_socket_server_host = GlobalParameter(ip"127.0.0.1")
const default_khepri_socket_server_port = GlobalParameter(12345)
export default_khepri_socket_server_host, default_khepri_socket_server_port
const khepri_socket_server_task = GlobalParameter{Union{Nothing,Task}}(nothing)

export ensure_khepri_socket_server_running
const _server_launch_lock = ReentrantLock()
ensure_khepri_socket_server_running() =
  lock(_server_launch_lock) do
    if isnothing(khepri_socket_server_task())
      khepri_socket_server_task(errormonitor(Threads.@spawn run_khepri_socket_server()))
    end
  end

#=
Upon connection and initialization, it might be
necessary to place some initial geometry on each
client that connects.
To that end, we will use the main generic function,
as a metaphor for the startup function that is the
entry program of C, Java, etc, programs.
=#
export main, main_callback
main_callback = GlobalParameter{Function}((b)->nothing)

#=
Parameter isolation across clients is handled by task-local storage: each
@spawn-ed client task gets its own values for current_backends, current_cs,
default_wall_family, user-defined Parameters, etc.
Each client also gets its own backend struct, so refs, websocket I/O, and
other backend-internal state are per-client with no sharing.
No global lock is needed — clients run concurrently in separate tasks.
=#

main(b::Backend) =
  with(current_backends, (b,)) do
    main_callback()(b)
  end

export run_khepri_socket_server
run_khepri_socket_server(host=default_khepri_socket_server_host(), port=default_khepri_socket_server_port()) =
  let server = listen(host, port)
    while true
      println("Waiting for a connection")
      let conn = accept(server)
        println("Connected!")
        let backend_name = decode(Val(:CS), Val(:string), conn),
            init_func = get_socket_backend_init_function(backend_name),
            backend = invokelatest(init_func, conn)
          invokelatest(before_connecting, backend)
    	    invokelatest(after_connecting, backend)
          add_global_backend(backend)
          invokelatest(main, backend)
        end
      end
    end
  end

#=
Another option is to use WebSockets. In this case, the backend is strictly a client.
This also needs an HTTP server to handle the initial handshake and a few more features.
=#

struct WebSocketBackend{K,T} <: RemoteBackend{K,T}
  name::String
  websocket::HTTP.WebSockets.WebSocket
  buffer::IOBuffer
  static_remote::NamedTuple
  remote::NamedTuple
  transaction::Parameter{Transaction}
  refs::References{K,T}
  handlers::Vector{Function}

  WebSocketBackend{K,T}(name, websocket, static_remote) where {K,T} =
    new{K,T}(name, websocket, IOBuffer(UInt8[], read=true, write=true), 
        static_remote, remote_functions(static_remote), 
        Parameter{Transaction}(AutoCommitTransaction()), 
        References{K,T}(), Function[])
end

# A WebSocketBackend is always connected
connection(b::WebSocketBackend) = b

send(b::WebSocketBackend, buffer) = 
  HTTP.WebSockets.send(b.websocket, take!(buffer))

receive(b::WebSocketBackend) = 
  let bytes = HTTP.WebSockets.receive(b.websocket)
    take!(b.buffer) # Clear the buffer
    write(b.buffer, bytes)
    seekstart(b.buffer)
    b.buffer
  end

# Finally, the server
mutable struct WebSocketServer
  server::HTTP.Server
  router::HTTP.Router
  handlers::Vector{Function}

  WebSocketServer(server, router) =
    new(server, router, Function[])
end

export register_http_handler
register_http_handler(c::WebSocketServer, target, handler) =
  let request_str = "/api/"*randstring()
    HTTP.register!(c.router, "GET", target, req -> (handler(request_parameters(req)...); HTTP.Response(200, "0")))
    request_str
  end

# Backends encode the parameters as query parameters p0, p1, p2, ...
# e.g., /api/handler?p0=val0&p1=val1&...
request_parameters(req) =
  let dict = HTTP.queryparams(HTTP.URI(HTTP.Messages.getfield(req, :target)))
    map(i -> haskey(dict, "p$i") ? dict["p$i"] : error("Missing parameter p$i"), 0:(length(dict)-1))
  end

const websocket_backend_init_map = Dict{String, Function}()
const websocket_backend_init_lock = ReentrantLock()
add_websocket_backend_init_function(name, init_function) = begin
  lock(websocket_backend_init_lock) do
    websocket_backend_init_map[name] = init_function
  end
end
get_websocket_backend_init_function(name) =
  lock(websocket_backend_init_lock) do
    get(websocket_backend_init_map, name) do
      error("Requested websocket backend '$name' is not available!")
    end
  end
export add_websocket_backend_init_function

const default_khepri_websocket_server_host = GlobalParameter(ip"127.0.0.1")
const default_khepri_websocket_server_port = GlobalParameter(12346)
export default_khepri_websocket_server_host, default_khepri_websocket_server_port

run_khepri_websocket_server(host=default_khepri_websocket_server_host(), port=default_khepri_websocket_server_port()) =
  let router = HTTP.Router(),
      server = HTTP.listen!(host, port) do http
                 if HTTP.WebSockets.isupgrade(http.message)
                    @info("WebSocket connection request for $(http.message.target)")
                    let backend_name = http.message.target[2:end] # Remove leading '/'
                      HTTP.WebSockets.upgrade(http) do websocket
                        let init_func = get_websocket_backend_init_function(backend_name),
                            backend = invokelatest(init_func, websocket)
                          invokelatest(before_connecting, backend)
    	                    invokelatest(after_connecting, backend)
                          add_global_backend(backend)
                          invokelatest(main, backend)
                        end
                        wait()
                      end
                    end
                 else
                    @info("HTTP request for $(http.message.target)")
                    HTTP.streamhandler(router)(http)
                 end
               end
    @info("Khepri started on URL:http://$(host):$(port)")
    khepri_websocket_server(WebSocketServer(server, router))
  end

export khepri_websocket_server
const khepri_websocket_server = LazyParameter(run_khepri_websocket_server)

#=
Because this relies on HTTP, we can also serve files, such as HTML, JS, CSS, images, etc.
=#

const content_type_header = Dict(
  "html"=>"text/html",
  "js"  =>"application/javascript",
  "css" =>"text/css",
  "png" =>"image/png",
  "jpg" =>"image/jpeg",
  "jpeg"=>"image/jpeg",
  "obj" =>"text/plain",
  "mtl" =>"text/plain",
  "hdr" =>"image/hdr",
  "gltf"=>"model/gltf+json",
  "glb" =>"model/gltf-binary",
  "bin" =>"application/octet-stream",
  )

get_file_content_type(path) =
  let ext = lowercase(lstrip(splitext(path)[2], '.'))
    ["Content-Type" => get(content_type_header, ext, "application/octet-stream")]
  end

export http_response_with_file, http_response_with_resource_file
http_response_with_file(path) =
  HTTP.Response(200, get_file_content_type(path), read(path))

#=
We need to support a PATH-based approach to resources.
=#
export resources_folder, add_resource_folder!
const resources_folder = [joinpath(@__DIR__, "..", "resources")]
const resources_folder_lock = ReentrantLock()

add_resource_folder!(path) =
  lock(resources_folder_lock) do
    pushfirst!(filter!(==(path), resources_folder), path)
  end

http_response_with_resource_file(filename) =
  let folders = lock(resources_folder_lock) do
                  copy(resources_folder)
                end
    for res_path in folders
      let full_path = joinpath(res_path, filename)
        if isfile(full_path)
          return http_response_with_file(full_path)
        end
      end
    end
    HTTP.Response(404, "File not found: $filename")
  end

#=
Each backend can register handlers for client requests.
=#
export register_handler
register_handler(c::WebSocketBackend{K,T}, target, handler) where {K,T} =
  (push!(c.handlers, handler); length(c.handlers))

call_handler(c::WebSocketBackend{K,T}, target, args...) where {K,T} =
  let handler = c.handlers[target]
    handler(args...)
  end

#=
Note that we cannot process client requests while server requests are being processed.
This means that the handler being called must do its job and return as soon as possible, 
so that other requests can be processed.
=#
export process_requests
process_requests(c::WebSocketBackend{K,T}) where {K,T} =
  let namespace = c.static_remote[1].namespace # An awful way of retrieving the namespace
    while true
      let buf = try 
                  receive(c) 
                catch e
                  if WebSockets.isok(e)
                    @warn("Connection to backend '$(c.name)' lost.")
                    delete_global_backend(c)
                    break
                  else
                    rethrow(e)
                  end
                end,
          target = decode(namespace, Val(:size), buf),
          args_len = decode(namespace, Val(:size), buf),
          args = Any[]
        for i in 1:args_len
          push!(args, decode(namespace, Val(:Any), buf))
        end
        @warn("Calling '$target' with $(args)")
        let result_code = -1 # Default: no result
          try
            with(current_backends, (c,)) do
              call_handler(c, target, args...)
            end
          catch e
            @warn("Error occurred while processing request '$target': $e")
            showerror(stdout, e, catch_backtrace())
            result_code = -2 # Error
          finally
            # Signal that the request has been processed
            let buf = IOBuffer()
              encode(namespace, Val(:int), buf, result_code)
              send(c, buf)
            end
          end
        end
      end
    end
  end

export start_processing_requests
start_processing_requests(c::WebSocketBackend{K,T}) where {K,T} = begin
  parent_tls = copy(task_local_storage())
  Threads.@spawn begin
    merge!(task_local_storage(), parent_tls)
    process_requests(c)
  end
  c
end

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
  #using_http_requests() ?
  #  (str) -> parameter(parse(T, str)) :
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
  color::RGBA
end

export b_layer, b_current_layer_ref,
       b_all_shapes_in_layer, b_delete_all_shapes_in_layer,
       b_set_layer_material

# Default implementation assumes that backends have properties for current_layer and layers (a dict)
b_layer(b::Backend, name, active, color) = BasicLayer(name, active, color)
b_current_layer_ref(b::Backend) = b.current_layer
b_current_layer_ref(b::Backend, layer) = b.current_layer = layer
b_all_shapes_in_layer(b::Backend, layer) = b.layers[layer]
b_delete_all_shapes_in_layer(b::Backend, layer) = b_delete_shapes(b_all_shapes_in_layer(b, layer))
b_set_layer_material(b::Backend, layer_ref, material_ref) = nothing


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
# Field-group structs for backend composition via @defbackend mixins.
# These are reusable building blocks — backends compose them instead of
# duplicating fields. Property forwarding is generated by @defbackend.

export LocalShapes, RenderState, IOState

@kwdef mutable struct LocalShapes
  shapes::Shapes = Shape[]
  current_layer::Union{Nothing, AbstractLayer} = nothing
  layers::Dict{AbstractLayer, Vector{Shape}} = Dict{AbstractLayer, Vector{Shape}}()
end

@kwdef mutable struct RenderState
  date::DateTime = DateTime(2020, 9, 21, 10, 0, 0)
  place::GeographicLocation = GeographicLocation(39, 9, 0, 0)
  render_env::RenderEnvironment = RealisticSkyEnvironment(5, true)
  ground_level::Float64 = 0.0
  ground_material::Union{Nothing, Material} = nothing
end

@kwdef mutable struct IOState
  io::IO = IOBuffer()
end

# Standalone functions that operate on field-group structs.
# These are used by @defbackend-generated default operations, and can also
# be called directly by backends that override only part of an operation.

export save_shape_local!, delete_all_local!, delete_shape_local!,
       set_realistic_sky_local!, set_ground_local!,
       realize_shapes_local!, used_materials_local!

save_shape_local!(ls::LocalShapes, s::Shape) =
  begin
    push!(ls.shapes, s)
    if !isnothing(ls.current_layer)
      push!(get!(ls.layers, ls.current_layer, Shape[]), s)
    end
    s
  end

delete_all_local!(ls::LocalShapes) =
  begin
    empty!(ls.shapes)
    for ss in values(ls.layers)
      empty!(ss)
    end
    nothing
  end

delete_shape_local!(ls::LocalShapes, shape::Shape) =
  let f(s) = s !== shape
    filter!(f, ls.shapes)
    for ss in values(ls.layers)
      filter!(f, ss)
    end
  end

set_realistic_sky_local!(rs::RenderState, date, latitude, longitude, elevation, meridian, turbidity, sun) =
  begin
    rs.date = date
    rs.place = GeographicLocation(latitude, longitude, elevation, meridian)
    rs.render_env = RealisticSkyEnvironment(turbidity, sun)
  end

set_ground_local!(rs::RenderState, level, mat) =
  begin
    rs.ground_level = level
    rs.ground_material = mat
  end

realize_shapes_local!(b, ls::LocalShapes) =
  for s in ls.shapes
    reset_ref(b, s)
    force_realize(b, s)
  end

used_materials_local!(ls::LocalShapes, rs::RenderState) =
  let materials = Set{Material}()
    for s in ls.shapes
      for m in used_materials(s)
        push!(materials, m)
      end
    end
    if !isnothing(rs.ground_material)
      push!(materials, rs.ground_material)
    end
    materials
  end

# Mixin registry: maps mixin name to (field_name, struct_type, forwarded_field_names)
const MIXIN_REGISTRY = Dict{Symbol, @NamedTuple{field::Symbol, type::Any, fields::Vector{Symbol}}}(
  :local_shapes => (field=:_local_shapes, type=:LocalShapes, fields=[:shapes, :current_layer, :layers]),
  :render_state => (field=:_render_state, type=:RenderState, fields=[:date, :place, :render_env, :ground_level, :ground_material]),
  :io           => (field=:_io_state,     type=:IOState,     fields=[:io]),
)

###############################################################################
# Another backend option is to save all shapes locally and then generate, e.g., a
# file-based description.

abstract type LocalBackend{K,T} <: Backend{K,T} end
# DEPRECATED: Use @defbackend with mixins instead of IOBackend.
# IOBackend is retained for backward compatibility but will be removed in a future version.
# Migration: replace `const MyBackend = IOBackend{K, T, E}` with:
#   @defbackend MyName Alias begin
#     parent = LocalBackend
#     mixin(local_shapes)   # shapes, current_layer, layers
#     mixin(render_state)   # date, place, render_env, ground_level, ground_material
#     mixin(io)             # io::IOBuffer
#     # fields from E go here directly
#   end
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
  io::IO=IOBuffer()
  extra::E=E()
end

view_type(::Type{<:LocalBackend}) = FrontendView()

# ─── LocalBackend default operations ─────────────────────────────────────────
# These serve as defaults for any LocalBackend subtype (including @defbackend
# backends with parent = LocalBackend). Field access goes through getproperty,
# so mixin forwarding works transparently.

connection(b::LocalBackend) = b.io

save_shape!(b::LocalBackend, s::Shape) =
  begin
    push!(b.shapes, s)
    if !isnothing(b.current_layer)
      push!(get!(b.layers, b.current_layer, Shape[]), s)
    end
    s
  end

export realize_shapes
realize_shapes(b::LocalBackend) =
  for s in b.shapes
    reset_ref(b, s)
    force_realize(b, s)
  end

export used_materials
used_materials(b::LocalBackend) =
  let materials = Set{Material}()
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

KhepriBase.b_delete_all_shape_refs(b::LocalBackend) =
  begin
    empty!(b.shapes)
    for ss in values(b.layers)
      empty!(ss)
    end
    empty!(b.refs)
    nothing
  end

KhepriBase.b_delete_shape(b::LocalBackend, shape::Shape) =
  let f(s) = s !== shape
    filter!(f, b.shapes)
    for ss in values(b.layers)
      filter!(f, ss)
    end
  end

KhepriBase.b_all_shapes(b::LocalBackend) = b.shapes
KhepriBase.b_all_shapes_in_layer(b::LocalBackend, layer) = b.layers[layer]

KhepriBase.b_realistic_sky(b::LocalBackend, date, latitude, longitude, elevation, meridian, turbidity, sun) =
  begin
    b.date = date
    b.place = GeographicLocation(latitude, longitude, elevation, meridian)
    b.render_env = RealisticSkyEnvironment(turbidity, sun)
  end

b_set_ground(b::LocalBackend, level, mat) =
  begin
    b.ground_level = level
    b.ground_material = mat
  end

# LocalBackend stores Shape proxies locally instead of immediately realizing them.
# Non-Shape proxies (UniqueProxy: Level, Material, Layer, Family) fall through
# to the transaction-based maybe_realize(b, s) in Shapes.jl.
maybe_realize(b::LocalBackend, s::Shape) = save_shape!(b, s)

# IOBackend (deprecated) inherits all LocalBackend operations above.

###############################################################################
# LazyBackend is for backends that delay shape realization by storing shapes
# locally for later batch processing (e.g., structural analysis backends).
# Unlike RemoteBackend which sends shapes to external applications, LazyBackend
# collects shapes in memory and processes them later.

export LazyBackend
abstract type LazyBackend{K,T} <: Backend{K,T} end

# LazyBackend stores Shape proxies using save_shape! instead of immediately realizing them.
# Non-Shape proxies (UniqueProxy: Level, Material, Layer, Family) fall through
# to the transaction-based maybe_realize(b, s) in Shapes.jl.
maybe_realize(b::LazyBackend, s::Shape) = save_shape!(b, s)

# Default save_shape! for LazyBackend - backends should specialize this
save_shape!(b::LazyBackend, s::Shape) = s
