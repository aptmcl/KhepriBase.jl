#Backends might use different communication mechanisms, e.g., sockets, COM, RMI, etc

#We start with socket-based communication
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

struct IOBufferBackend{K,T} <: Backend{K,T}
  out::IOBuffer
end
connection(backend::IOBufferBackend) = backend.out
