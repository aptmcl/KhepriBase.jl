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
public References
public shape_refs_storage, material_refs_storage, layer_refs_storage, annotation_refs_storage, family_refs_storage, level_refs_storage, refs_storage
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
public get_or_create_shape_from_ref_value

get_or_create_from_ref_value(b::Backend, r, storage, create) =
  let dict = storage(b)
    for (proxy, ref) in dict
      if contains(ref, r)
        return proxy
      end
    end
    with_introspection(b) do
      let new_sh = create(b, r)
        ref!(b, new_sh, r)
        new_sh
      end
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


#=
Discard all state whose lifetime is tied to a single C# Processor instance:

- remote opcodes — assigned by `ProvideOperation` per-Processor; an integer
  cached from a prior Processor will dispatch to the wrong handler in a fresh
  one (silent corruption, since opcodes are just ints with no fingerprint).
- shape/material/layer refs — backend-side handles only valid for the C#
  session that issued them.
- family refs — same lifetime as the refs above.

This is called from every path that ends a session (`reset_backend`,
`retire_dead_backend`) AND from `connection` after a fresh socket is
established. The latter is the load-bearing one: it makes "if the connection
is new, the session state is fresh" hold by construction, which closes the
silent-opcode-drift failure mode where a teardown path forgot to clear.

Idempotent: calling it twice (e.g., in reset_backend then again on the next
connection) is a no-op the second time.

See also: `connection`, `reset_backend`, `retire_dead_backend`.
=#
discard_session_state!(b::RemoteBackend) =
  begin
    for f in b.remote
      reset_opcode(f)
    end
    empty!(b.refs)
    invalidate_family_refs(b)
  end

# There is a protocol for retrieving the connection
connection(b::RemoteBackend) =
  begin
    if ismissing(b.connection)
      before_connecting(b)
      b.connection = start_connection(b)
      # Must run before after_connecting: hooks like set_material and
      # set_backend_family populate b.refs in the new session.
      discard_session_state!(b)
      after_connecting(b)
    end
    b.connection
  end

#=
Teardown layers:

- close_connection!: transport-level — close the socket, mark the field
  missing. No knowledge of session state.
- reset_backend: explicit user reset. Discards session state and closes the
  transport, but leaves the backend in current_backends() so the next op
  reconnects automatically.
- retire_dead_backend: auto-recovery on IOError/EOFError mid-RPC. Same as
  reset_backend, but also removes the backend from current_backends() — the
  user must explicitly re-add it to use it again.
=#

close_connection!(b::RemoteBackend) =
  begin
    try close(b.connection) catch end
    b.connection = missing
  end

reset_backend(b::RemoteBackend) =
  begin
    discard_session_state!(b)
    close_connection!(b)
  end

retire_dead_backend(b::RemoteBackend) =
  begin
    @warn "Backend $(backend_name(b)) disconnected. Removing from active backends."
    delete_global_backend(b)
    discard_session_state!(b)
    close_connection!(b)
  end

#=
Julia surfaces a write to a closed socket as
ArgumentError("stream is closed or unusable") — the SAME exception type as
the most common generic programming error.  The retire policy used to
classify by type alone (Union{IOError, ArgumentError, EOFError}), so ANY
ArgumentError raised inside a b_* implementation — KhepriBase itself throws
ArgumentError from dozens of geometry-validation sites reachable during
realize — silently retired the backend: session refs wiped, healthy socket
closed, and in the @defcbs/maybe_realize loops the error fully swallowed
while the script kept "succeeding" with degraded output.

The fix is to classify at the transport boundary instead of by type: the
transport primitives (send/receive on TCPSocket in Primitives.jl and on
WebSocketBackend below) wrap their bodies in rethrow_disconnected, which
converts IOError, EOFError, and the closed-stream ArgumentError into
BackendDisconnected.  Only errors raised by the transport itself can carry
that type, so handle_backend_error's retire method no longer needs (and no
longer has) plain ArgumentError in its set — a b_* implementation bug now
propagates as an error on the backend where it occurred.

See also: handle_backend_error, retire_dead_backend, send/receive
(Primitives.jl).
=#
public BackendDisconnected
"The connection to a remote backend was lost mid-operation; `cause` holds the underlying transport exception."
struct BackendDisconnected <: Exception
  cause::Any
end

Base.showerror(io::IO, e::BackendDisconnected) =
  begin
    print(io, "BackendDisconnected: the connection to the backend was lost (")
    showerror(io, e.cause)
    print(io, ")")
  end

is_disconnection(e::Union{Base.IOError, EOFError}) = true
is_disconnection(e::ArgumentError) = occursin("stream is closed or unusable", e.msg)
is_disconnection(e) = false

# To be called from a catch block wrapping a transport primitive: converts
# transport-death exceptions into BackendDisconnected and rethrows anything else.
rethrow_disconnected(e) =
  is_disconnection(e) ? throw(BackendDisconnected(e)) : rethrow()

handle_backend_error(e, b::Backend) = rethrow()
handle_backend_error(e::Union{BackendDisconnected, Base.IOError, EOFError}, b::RemoteBackend) =
  retire_dead_backend(b)

public RemoteBackend, before_connecting, after_connecting, start_connection, failed_connecting, retry_connecting,
       retire_dead_backend, handle_backend_error
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

SocketBackend{K,T}(name, port, static_remote) where {K,T} =
  SocketBackend{K,T}(name, port, missing, static_remote)

SocketBackend{K,T}(name, port, connection, static_remote) where {K,T} =
  SocketBackend{K,T}(name, port, connection, static_remote, remote_functions(static_remote), Parameter{Transaction}(AutoCommitTransaction()), References{K,T}())

# AML: replace with the Base.retry function?
#=
Use `Sockets.connect` explicitly rather than the unqualified name: KhepriBase
also exports a `connect` annotation combinator from Designs/combinators.jl,
which shadows `Sockets.connect` inside this module and silently turns every
socket-backend connection attempt into a MethodError — the backend then logs
"Couldn't connect" despite the backend process listening normally.
=#
start_connection(b::SocketBackend) =
  let attempts = 10,
      last_error = Ref{Any}(nothing)
    for i in 1:attempts
      try
        let conn = Sockets.connect(b.port)
          Sockets.nagle(conn, false)
          return conn
        end
      catch e
        last_error[] = e
        if i == attempts
          failed_connecting(b)
        else
          retry_connecting(b)
        end
      end
    end
    isnothing(last_error[]) ?
      error("Could not connect to $(b.name).") :
      throw(last_error[])
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
public add_socket_backend_init_function

# The server code

const default_khepri_socket_server_host = GlobalParameter(ip"127.0.0.1")
const default_khepri_socket_server_port = GlobalParameter(12345)
public default_khepri_socket_server_host, default_khepri_socket_server_port
const khepri_socket_server_task = GlobalParameter{Union{Nothing,Task}}(nothing)

public ensure_khepri_socket_server_running
const _server_launch_lock = ReentrantLock()
ensure_khepri_socket_server_running() =
  lock(_server_launch_lock) do
    let task = khepri_socket_server_task()
      if isnothing(task) || istaskfailed(task) || istaskdone(task)
        khepri_socket_server_task(errormonitor(Threads.@spawn run_khepri_socket_server()))
      end
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
public main, main_callback
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

public run_khepri_socket_server
run_khepri_socket_server(host=default_khepri_socket_server_host(), port=default_khepri_socket_server_port()) =
  let server = listen(host, port)
    while true
      println("Waiting for a connection")
      let conn = accept(server)
        Sockets.nagle(conn, false)
        try
          println("Connected!")
          let backend_name = decode(Val(:CS), Val(:string), conn),
              init_func = get_socket_backend_init_function(backend_name),
              backend = invokelatest(init_func, conn)
            invokelatest(before_connecting, backend)
            invokelatest(after_connecting, backend)
            add_global_backend(backend)
            invokelatest(main, backend)
          end
        catch e
          @warn "Error handling client connection" exception=(e, catch_backtrace())
          try close(conn) catch end
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

# WebSocketBackend is its own connection object (send/receive dispatch on it)
connection(b::WebSocketBackend) = b

close_connection!(b::WebSocketBackend) =
  try close(b.websocket) catch end

# Transport primitives: IOError/EOFError/closed-stream ArgumentError become
# BackendDisconnected (see the prose block above BackendDisconnected).
# WebSocketError is deliberately NOT reclassified: process_requests below
# distinguishes clean closes via WebSockets.isok and must see it raw.
send(b::WebSocketBackend, buffer) =
  try
    HTTP.WebSockets.send(b.websocket, take!(buffer))
  catch e
    rethrow_disconnected(e)
  end

receive(b::WebSocketBackend) =
  try
    let bytes = HTTP.WebSockets.receive(b.websocket)
      take!(b.buffer) # Clear the buffer
      write(b.buffer, bytes)
      seekstart(b.buffer)
      b.buffer
    end
  catch e
    rethrow_disconnected(e)
  end

# Finally, the server
mutable struct WebSocketServer
  server::HTTP.Server
  router::HTTP.Router
end

public register_http_handler
register_http_handler(c::WebSocketServer, target, handler) =
  let request_str = "/api/"*randstring()
    HTTP.register!(c.router, "GET", request_str, req -> (handler(request_parameters(req)...); HTTP.Response(200, "0")))
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
public add_websocket_backend_init_function

const default_khepri_websocket_server_host = GlobalParameter(ip"127.0.0.1")
const default_khepri_websocket_server_port = GlobalParameter(12346)
public default_khepri_websocket_server_host, default_khepri_websocket_server_port

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

public khepri_websocket_server
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

public http_response_with_file, http_response_with_resource_file
http_response_with_file(path) =
  HTTP.Response(200, get_file_content_type(path), read(path))

#=
We need to support a PATH-based approach to resources.
=#
public resources_folder, add_resource_folder!
const resources_folder = [joinpath(@__DIR__, "..", "resources")]
const resources_folder_lock = ReentrantLock()

add_resource_folder!(path) =
  lock(resources_folder_lock) do
    pushfirst!(filter!(!=(path), resources_folder), path)
  end

function resource_relative_path(filename)
  filename isa AbstractString || return nothing
  isempty(filename) && return nothing
  occursin('\0', filename) && return nothing
  decoded = try
    HTTP.URIs.unescapeuri(replace(filename, '\\' => '/'))
  catch
    return nothing
  end
  any(==(".."), splitpath(decoded)) && return nothing
  rel = normpath(decoded)
  (isempty(rel) || rel == "." || isabspath(rel)) && return nothing
  any(==(".."), splitpath(rel)) && return nothing
  rel
end

function resource_file_in_root(root, rel)
  root_abs = abspath(root)
  candidate = abspath(joinpath(root_abs, rel))
  from_root = relpath(candidate, root_abs)
  (from_root == ".." || startswith(from_root, string("..", Base.Filesystem.path_separator)) || isabspath(from_root)) &&
    return nothing
  candidate
end

http_response_with_resource_file(filename) =
  let rel = resource_relative_path(filename)
    rel === nothing && return HTTP.Response(400, "Invalid resource path")
    let folders = lock(resources_folder_lock) do
                  copy(resources_folder)
                end
      for res_path in folders
        let full_path = resource_file_in_root(res_path, rel)
          if full_path !== nothing && isfile(full_path)
            return http_response_with_file(full_path)
          end
        end
      end
      HTTP.Response(404, "File not found: $rel")
    end
  end

#=
Each backend can register handlers for client requests.
=#
public register_handler
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
public process_requests
process_requests(c::WebSocketBackend{K,T}) where {K,T} =
  let namespace = c.static_remote[1].namespace # An awful way of retrieving the namespace
    while true
      let buf = try 
                  receive(c) 
                catch e
                  # Symmetric cleanup on every error path: the non-isok branch
                  # previously rethrew without removing the backend or closing the
                  # socket, leaving a zombie backend registered with a dead socket and
                  # a dead task. retire_dead_backend does the full delete+discard+close.
                  retire_dead_backend(c)
                  WebSockets.isok(e) ? break : rethrow(e)
                end
        # Decode + dispatch + reply all inside one try so a MALFORMED FRAME
        # (bad target / arg-count / arg payload) is answered with an error
        # result_code in the finally instead of escaping the loop — otherwise a
        # timeout-less client blocks forever on a reply that never comes, and this
        # request task dies silently. target starts unresolved for the error log.
        let result_code = -1, target = -1 # Default: no result
          try
            target = decode(namespace, Val(:size), buf)
            let args_len = decode(namespace, Val(:size), buf),
                args = Any[]
              for i in 1:args_len
                push!(args, decode(namespace, Val(:Any), buf))
              end
              @warn("Calling '$target' with $(args)")
              with(current_backends, (c,)) do
                call_handler(c, target, args...)
              end
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

public start_processing_requests
start_processing_requests(c::WebSocketBackend{K,T}) where {K,T} = begin
  errormonitor(spawn_inheriting(() -> process_requests(c)))   # child inherits TLS; errormonitor surfaces a silent task-death
  c
end

#=
It is going to be useful to have an algebra of handlers.
=#
public action_handler, sequence_handler, wrapper_handler, update_parameter_handler

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
public AbstractLayer, BasicLayer
abstract type AbstractLayer end
struct BasicLayer <: AbstractLayer
  name::String
  visible::Bool
  color::RGBA
end

public b_layer, b_current_layer_ref,
       b_all_shapes_in_layer, b_delete_all_shapes_in_layer,
       b_set_layer_material, b_set_layer_visible, b_set_layer_opacity

# Default implementation assumes that backends have properties for current_layer and layers (a dict)
b_layer(b::Backend, name, visible, color) = BasicLayer(name, visible, color)
b_current_layer_ref(b::Backend) = b.current_layer
b_current_layer_ref(b::Backend, layer) = b.current_layer = layer
# `b.layers` is populated lazily (a key appears only once a shape is added to that
# layer, via `get!(layers, cur, Proxy[])`).  A layer that exists but holds no shapes
# has no key, so index with `get(...)` and fall back to an empty collection rather
# than raising KeyError.  `Proxy[]` is the broadest element type: layers may hold
# both Shapes and Annotations (both <: Proxy), and it is assignment-compatible with
# every concrete `b.layers` value type (Vector{Proxy} or Vector{Shape}).
b_all_shapes_in_layer(b::Backend, layer) = get(b.layers, layer, Proxy[])
#=
The default delete-all is composed from the two per-item ops the backend
already has: `b_all_shapes_in_layer` enumerates and `b_delete_shape`
handles each shape's ref lifecycle (the generic method routes through
maybe_delete, which deletes the native refs and resets the proxy's ref
entry; LocalBackend's override also removes the shape from its local
storage and layer index).  Backends that can delete a whole layer
remotely in one call (AutoCAD, Rhino, ...) override this op.

The `copy` is load-bearing: `b_all_shapes_in_layer` returns the *live*
layer collection on the default and LocalBackend paths, and
`b_delete_shape` may mutate that very collection (LocalBackend filter!s
it), so iterating the original would skip every other shape.

See also: delete_all_shapes_in_layer (Shapes.jl), b_delete_shape.
=#
b_delete_all_shapes_in_layer(b::Backend, layer) =
  for s in copy(b_all_shapes_in_layer(b, layer))
    b_delete_shape(b, s)
  end
#=
The three layer-property setters below are *deliberate* no-ops in the
generic fallback, not unfinished stubs.  Layer styling is an optional
capability: many backends model layer visibility/opacity but have no
per-layer *material* concept (Blender, Three.js, Unity and Unreal
override `b_set_layer_visible`/`b_set_layer_opacity` yet intentionally
leave `b_set_layer_material` at this default), and pure-geometry or
file-output backends may expose no layer styling at all.  For those a
styling request must be silently ignored rather than raised: under the
backend-portability rule the *same* script has to render on every
backend, so an unsupported style should degrade to "unstyled", never
to an error.  This mirrors `b_view_settings` (Frontend.jl), whose
`nothing` fallback is documented the same way.

A backend that genuinely supports a property overrides the matching
method (see AutoCAD/Rhino `b_set_layer_material`, and
`b_set_layer_visible`/`b_set_layer_opacity` in most socket backends);
a backend that must *reject* unsupported styling can override with an
`error(...)`.  The base default chooses portability over noise.
=#
b_set_layer_material(b::Backend, layer_ref, material_ref) = nothing
b_set_layer_visible(b::Backend, layer, visible) = nothing
b_set_layer_opacity(b::Backend, layer, opacity) = nothing


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

public LocalShapes, RenderState, IOState

#=
`shapes` holds both `Shape` *and* `Annotation` instances.  An
`Annotation` (`label`, `radius_illustration`, …) is realised by the
backend exactly like a shape — the realisation calls `b_labels` /
`b_radii_illustration` / … on the backend — but it carries no
geometry of its own, only decorations on top of whatever shapes were
emitted earlier.  Storing it alongside shapes lets the backend's
render path realise everything in one pass; a separate annotation
queue would force every backend to commit a transaction at the
right moment and for the right reasons, which historically ended
up forgotten in the SVG and TikZ pipelines.

The local-shape accessors (`local_shape_storage`, etc.) take
`Proxy` rather than `Shape` so an `Annotation` flows through the
same path; backends that want to special-case annotations can still
filter on `is_illustration(...)` or `isa Annotation` when they
render.  See also: KhepriSVG / KhepriTikZ `sort_illustrations!` —
they expect annotations to live in `b.shapes`.
=#
@kwdef mutable struct LocalShapes
  shapes::Vector{Proxy} = Proxy[]
  current_layer::Union{Nothing, AbstractLayer} = nothing
  layers::Dict{AbstractLayer, Vector{Proxy}} = Dict{AbstractLayer, Vector{Proxy}}()
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
# backends with parent = LocalBackend).  Historically the helpers below read
# `b.shapes`/`b.layers` directly, which only works for the legacy IOBackend
# with direct fields.  The new @defbackend+mixin style exposes the same names
# via `getproperty` forwarding, but Julia specialises `save_shape!(b::LocalBackend,
# s)` against the `Base.getproperty(::Any, ::Symbol) = getfield(…)` fallback
# when the mixin-typed backend was not yet loaded at KhepriBase compile time,
# and the specialisation misses the per-type getproperty method added by
# @defbackend.  Going through `local_shape_storage` and `local_shape_refs`
# sidesteps that: the accessor dispatches on `hasfield` at compile time so
# legacy and mixin backends both resolve to the right container.
#=
See also: `@defbackend` (src/Backend.jl) and the `_local_shapes` mixin
(LocalShapes, above), which is where the struct these accessors bridge to
is declared.
=#

# Returns the `Shape` vector the backend stores locally.
local_shape_storage(b) =
  hasfield(typeof(b), :_local_shapes) ? b._local_shapes.shapes : b.shapes

# Returns the backend's current layer (may be `nothing`).
local_current_layer(b) =
  hasfield(typeof(b), :_local_shapes) ? b._local_shapes.current_layer : b.current_layer

# Returns the `Dict{AbstractLayer,Vector{Shape}}` the backend keeps.
local_layer_index(b) =
  hasfield(typeof(b), :_local_shapes) ? b._local_shapes.layers : b.layers

connection(b::LocalBackend) = b.io

public save_shape!
save_shape!(b::LocalBackend, s::Proxy) =
  begin
    push!(local_shape_storage(b), s)
    cur = local_current_layer(b)
    if !isnothing(cur)
      push!(get!(local_layer_index(b), cur, Proxy[]), s)
    end
    s
  end

public realize_shapes
#=
Realizing one shape can delete *other* queued shapes: b_surface over a
frontier of curves and_delete_shapes the boundary shapes (Backend.jl), and
b_delete_shape on a LocalBackend filter!s local_shape_storage — the very
vector this loop iterates. An in-place iteration would shift the survivors
over the deleted slots and silently skip them (predioSinusoidal lost all
four extruded solids and six cylinders this way before profile consumption
moved to construction time). Iterate a snapshot and skip, by identity, any
entry a previous realization removed from the queue.

See also: b_consume_shape(::LocalBackend) and b_delete_shape below; the
extruded/swept after_init overrides in Shapes.jl.
=#
realize_shapes(b::LocalBackend) =
  let queued = copy(local_shape_storage(b))
    for s in queued
      any(q -> q === s, local_shape_storage(b)) || continue
      reset_ref(b, s)
      force_realize(b, s)
    end
  end

public used_materials
used_materials(b::LocalBackend) =
  let materials = Set{Material}()
    for s in local_shape_storage(b)
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
    empty!(local_shape_storage(b))
    for ss in values(local_layer_index(b))
      empty!(ss)
    end
    empty!(b.refs.shapes)
    nothing
  end

KhepriBase.b_delete_shape(b::LocalBackend, shape::Proxy) =
  let f(s) = s !== shape
    filter!(f, local_shape_storage(b))
    for ss in values(local_layer_index(b))
      filter!(f, ss)
    end
  end

#=
A LocalBackend queues shapes (save_shape!) and realizes them in a later
realize_shapes pass, so a profile consumed at construction by an
extrusion/sweep proxy must be unqueued here or its bytes leak into the
output ahead of its consumer (the file-output goldens bless its absence).
Delegating to b_delete_shape is exactly "unqueue, don't destroy": the
override above only filter!s the local storage and layer index — it never
touches refs, which realize_shapes re-mints on every pass anyway.
=#
KhepriBase.b_consume_shape(b::LocalBackend, s::Shape) =
  b_delete_shape(b, s)

KhepriBase.b_all_shapes(b::LocalBackend) = local_shape_storage(b)
# Mirror the generic default: the layer index is filled lazily by `save_shape!`
# (`get!(local_layer_index(b), cur, Proxy[])`), so an empty layer has no key.
# Use `get` to return an empty `Proxy[]` instead of throwing KeyError.
KhepriBase.b_all_shapes_in_layer(b::LocalBackend, layer) = get(local_layer_index(b), layer, Proxy[])

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

# LocalBackend stores Shape *and* Annotation proxies locally instead of
# immediately realizing them.  Annotations need the same deferred
# realization as shapes — backends like KhepriSVG / KhepriTikZ build their
# document inside `b_render_and_save_view` by setting `b.io` to a temporary
# buffer and then calling `realize_shapes`; if annotations were realised
# eagerly via the transaction path, their output would land in the wrong
# IO and be silently dropped.  Routing them through `save_shape!` puts them
# on the same `local_shape_storage` queue as shapes and they get realised
# in the render pass.  See also: `LocalShapes` definition and the
# `sort_illustrations!` hooks in KhepriSVG / KhepriTikZ.
#
# Other Proxy subtypes (UniqueProxy: Level, Material, Layer, Family) still
# fall through to the transaction-based `maybe_realize(b, s)` in Shapes.jl
# because they are realised once and then reused — caching, not queuing.
maybe_realize(b::LocalBackend, s::Shape)      = save_shape!(b, s)
maybe_realize(b::LocalBackend, s::Annotation) = save_shape!(b, s)

# IOBackend (deprecated) inherits all LocalBackend operations above.

###############################################################################
# LazyBackend is for backends that delay shape realization by storing shapes
# locally for later batch processing (e.g., structural analysis backends).
# Unlike RemoteBackend which sends shapes to external applications, LazyBackend
# collects shapes in memory and processes them later.

public LazyBackend
abstract type LazyBackend{K,T} <: Backend{K,T} end

# LazyBackend stores Shape proxies using save_shape! instead of immediately realizing them.
# Non-Shape proxies (UniqueProxy: Level, Material, Layer, Family) fall through
# to the transaction-based maybe_realize(b, s) in Shapes.jl.
maybe_realize(b::LazyBackend, s::Shape) = save_shape!(b, s)

#=
A LazyBackend overrides realization entirely (see maybe_realize above): instead
of going through the transaction/force_realize path, every Shape is handed to
`save_shape!` for deferred batch processing.  A backend therefore MUST decide,
per shape type, what happens to it.  The historical default silently returned
the shape unstored (`= s`), which meant any shape type a LazyBackend forgot to
handle was dropped without a trace — exactly the silent-no-op stub the project's
stub policy forbids (CLAUDE.md: a default that discards input must error() or
carry a prose block stating the silence is deliberate).

We make the unhandled case a loud, actionable error.  Backends that genuinely
want to ignore some shape types (e.g. a structural-analysis backend dropping
decorative geometry) can still do so — but they must opt in explicitly with a
documented catch-all `save_shape!(b::MyBackend, s::Shape) = s`, which makes the
discard a visible design decision rather than an accident.  Specialised methods
for handled types (e.g. Frame4DD's TrussNode/TrussBar) win by dispatch and never
reach this fallback.

See also: maybe_realize(b::LazyBackend, s::Shape) above; save_shape!(b::LocalBackend, s::Proxy);
KhepriFrame4DD/src/Frame4DD.jl (save_shape! for TrussNode/TrussBar).
=#
save_shape!(b::LazyBackend, s::Shape) =
  error("Backend $(backend_name(b)) (a LazyBackend) received a $(nameof(typeof(s))) " *
        "but defines no save_shape! method for it. A LazyBackend stores shapes for " *
        "later batch processing; an unhandled shape type would be silently dropped. " *
        "Add a `save_shape!(b::$(typeof(b)), s::$(nameof(typeof(s))))` method, or an " *
        "explicit catch-all `save_shape!(b::$(typeof(b)), s::Shape) = s` if discarding " *
        "such shapes is intended.")
