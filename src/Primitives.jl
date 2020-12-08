#=
const julia_type_for_c_type = Dict(
  :byte => :Int8,
  :double => :Float64,
  :float => :Float64,
  :int => :Int,
  :bool => :Bool,
  :Point3d => :XYZ,
  :Point2d => :XYZ,
  :Vector3d => :VXYZ,
  :string => :ASCIIString,
  :ObjectId => :Int32,
  :Entity => :Int32,
  :BIMLevel => :Int32,
  :FloorFamily => :Int32,
    ) #Either Int32 or Int64, depending on the architecture

julia_type(ctype, is_array) = is_array ? :(Vector{$(julia_type(ctype, false))}) : julia_type_for_c_type[ctype]
=#
export show_rpc, step_rpc
const show_rpc = Parameter(false)
const step_rpc = Parameter(false)
function initiate_rpc_call(conn, opcode, name)
    if step_rpc()
        print(stderr, "About to call $(name) (opcode $(opcode)) [press ENTER]")
        readline()
    end
    if show_rpc()
        print(stderr, "$(name) (opcode $(opcode))")
    end
end
function complete_rpc_call(conn, opcode, result)
    if show_rpc()
        println(stderr, result == nothing ? "-> nothing" : "-> $(result)")
    end
    result
end

function my_symbol(prefix, name)
  Symbol(prefix * replace(string(name), ":" => "_"))
end

function translate(sym)
  Symbol(replace(string(sym), r"[:<>]" => "_", ))
end

#=
We parameterize signatures to support different programming languages, e.g.
Val{:CPP} is for C++ functions, while Val{:CS} is for C# static methods

A basic type is represented with an identically-named symbol, e.g.,
int -> :int
An array type is represented with a tuple, e.g.,
int[] -> (:array, :int)
A parametric type is represented with a tuple, e.g.,
Dict<int, string> -> (:Dict, :int, :string)
=#
#=
# C++
parse_signature(::Val{:CPP}, sig::AbstractString) =
  let func_name(name) = replace(name, ":" => "_"),
      type_name(name) = replace(name, r"[:<>]" => "_"),
      m = match(r"^ *(public|) *(\w+) *([\[\]]*) +((?:\w|:|<|>)+) *\( *(.*) *\)", sig),
      ret = type_name(m.captures[2]),
      array_level = count(c -> c=='[', something(m.captures[3], "")),
      name = m.captures[4],
      params = split(m.captures[5], r" *, *", keepempty=false),
      parse_c_decl(decl) =
        let m = match(r"^ *((?:\w|:|<|>)+) *([\[\]]*) *(\w+)$", decl)
          (type_name(m.captures[1]), count(c -> c=='[', something(m.captures[2], "")), Symbol(m.captures[3]))
        end
    (func_name(name), name, [parse_c_decl(decl) for decl in params], (ret, array_level))
  end

# C#
parse_signature(::Val{:CS}, sig::AbstractString) =
  let m = match(r"^ *(public|) *(\w+) *([\[\]]*) +(\w+) *\( *(.*) *\)", sig),
      ret = m.captures[2],
      array_level = count(c -> c=='[', something(m.captures[3], "")),
      name = m.captures[4],
      params = split(m.captures[5], r" *, *", keepempty=false),
      parse_c_decl(decl) =
        let m = match(r"^ *(\w+) *([\[\]]*) *(\w+)$", decl)
          (m.captures[1], count(c -> c=='[', something(m.captures[2], "")), Symbol(m.captures[3]))
    end
    (name, name, [parse_c_decl(decl) for decl in params], (ret, array_level))
  end

# Python
parse_signature(::Val{:PY}, sig::AbstractString) =
  let m = match(r"^ *def +(\w+) *(\(.*\)) *-> *(.+) *: *", sig),
      name = m.captures[1],
      params = m.captures[2],
      ret = m.captures[3],
      parse_type(t) =
        if t isa Symbol
          (t, 0)
        elseif t.head === :ref
          if t.args[1] === :List
            (s, l) = parse_type(t.args[2])
            (s, l + 1)
          elseif t.args[1] === :Tuple
            (Symbol(:T, join(t.args[2:end], "_"), :T), 0)
          else
            error("Uknown expression $(t)")
          end
        else
          error("Uknown expression $(t)")
        end,
      parse_params(ast) =
        let parse_param(p) = (parse_type(p.args[3])..., p.args[2])
          if ast.head === :tuple
            map(parse_param, ast.args)
          elseif ast.head === :call
            [parse_param(ast)]
          else
            error("Uknown expression $(t)")
          end
        end
    (name, name, parse_params(Meta.parse(params)), parse_type(Meta.parse(ret)))
  end
=#
###########################################################
# New version
# C++
parse_signature(::Val{:CPP}, sig::AbstractString) =
  let func_name(name) = replace(name, ":" => "_"),
      type_name(name) = replace(name, r"[:<>]" => "_"),
      packtype(t, n) = foldr((i,v)->Vector{v}, 1:n, init=Val{Symbol(t)}),
      m = match(r"^ *(public|) *(\w+) *([\[\]]*) +((?:\w|:|<|>)+) *\( *(.*) *\)", sig),
      ret = packtype(type_name(m.captures[2]), count(c -> c=='[', something(m.captures[3], ""))),
      name = m.captures[4],
      params = split(m.captures[5], r" *, *", keepempty=false),
      parse_c_decl(decl) =
        let m = match(r"^ *((?:\w|:|<|>)+) *([\[\]]*) *(\w+)$", decl)
          (packtype(type_name(m.captures[1]), count(c -> c=='[', something(m.captures[2], ""))), Symbol(m.captures[3]))
        end
    (func_name(name), name, [parse_c_decl(decl) for decl in params], ret)
  end

# C#
parse_signature(::Val{:CS}, sig::AbstractString) =
  let packtype(t, n) = foldr((i,v)->Vector{v}, 1:n, init=Val{Symbol(t)}),
      m = match(r"^ *(public|) *(\w+) *([\[\]]*) +(\w+) *\( *(.*) *\)", sig),
      ret = packtype(m.captures[2], count(c -> c=='[', something(m.captures[3], ""))),
      name = m.captures[4],
      params = split(m.captures[5], r" *, *", keepempty=false),
      parse_c_decl(decl) =
        let m = match(r"^ *(\w+) *([\[\]]*) *(\w+)$", decl)
          (packtype(m.captures[1], count(c -> c=='[', something(m.captures[2], ""))), Symbol(m.captures[3]))
    end
    (name, name, [parse_c_decl(decl) for decl in params], ret)
  end

# Python
parse_signature(::Val{:PY}, sig::AbstractString) =
  let m = match(r"^ *def +(\w+) *(\(.*\)) *-> *(.+) *: *", sig),
      name = m.captures[1],
      params = m.captures[2],
      ret = m.captures[3],
      parse_type(t) =
        if t isa Symbol
          Val{t}
        elseif t.head === :ref
          if t.args[1] === :List
            Vector{parse_type(t.args[2])}
          elseif t.args[1] === :Tuple
            Tuple{map(parse_type, t.args[2:end])...}
          else
            error("Uknown expression $(t)")
          end
        else
          error("Uknown expression $(t)")
        end,
      parse_params(ast) =
        let parse_param(p) = (parse_type(p.args[3]), p.args[2])
          if ast.head === :tuple
            map(parse_param, ast.args)
          elseif ast.head === :call
            [parse_param(ast)]
          else
            error("Uknown expression $(t)")
          end
        end
    (name, name, parse_params(Meta.parse(params)), parse_type(Meta.parse(ret)))
  end


#=
A remote function encapsulates the information needed for communicating with remote
applications, including the opcode that represents the remote function and that
is generated by the remote application from the remote_name upon request.
=#

mutable struct RemoteFunction{T} <: Function
  namespace::T
  signature::String
  local_name::String
  remote_name::String
  opcode::Int32
  encoder::Function
  buffer::IOBuffer
end

remote_function(ns::T,
                sig::AbstractString,
                local_name::AbstractString,
                remote_name::AbstractString,
                encoder::Function) where {T} =
  RemoteFunction(
    ns,
    string(sig),
    string(local_name),
    string(remote_name),
    Int32(-1),
    encoder,
    IOBuffer())

request_operation(f::RemoteFunction{T}, conn::IO) where {T} = begin
  encode(f.namespace, Val(:int), conn, 0)
  encode(f.namespace, Val(:string), conn, f.remote_name)
  op = decode(f.namespace, Val(:size), conn)
  if op == -1
    error(f.remote_name * " is not available")
  else
    op
  end
end

interrupt_processing(conn::IO) = write(conn, Int32(-1))


ensure_opcode(f::RemoteFunction, conn) =
  f.opcode == -1 ?
    f.opcode = Int32(request_operation(f, conn)) :
    f.opcode

reset_opcode(f::RemoteFunction) =
  f.opcode = -1

call_remote(f::RemoteFunction, conn, args...) =
  f.encoder(ensure_opcode(f, conn), conn, f.buffer, args...)

(f::RemoteFunction)(conn, args...) = call_remote(f, conn, args...)

#=
The most important part is the lang_rpc function. It parses the string describing the
signature of the remote function, generates a function to encode arguments and
retrieve results and, finally, creates the remote_function object.
=#
#=
remote_function_meta_program(nssym, sig, local_name, remote_name, params, ret) =
  let packtype(t, n) =
        let init = t isa Tuple ? :(Tuple{$(t...)}) : t
          foldr((i,v)->:(Vector{$v}), 1:n, init=:(Val{$(QuoteNode(Symbol(t)))}))
        end,
      namespace = :(Val{$(nssym)}())
    :(remote_function(
           $(namespace),
           $(sig),
           $(local_name),
           $(remote_name),
           (opcode, conn, buf, $([:($(p[3])) for p in params]...)) -> begin
              initiate_rpc_call(conn, opcode, $(remote_name))
              take!(buf) # Reset the buffer just in case there was an encoding error on a previous call
              encode($(namespace), Val(:int), buf, opcode)
              $([:(encode($(namespace),
                          $(packtype(p[1], p[2]))(),
                          buf,
                          $(p[3])))
                 for p in params]...)
              write(conn, take!(buf))
              complete_rpc_call(conn, opcode,
                decode($(namespace),
                       $(packtype(ret[1], ret[2]))(), conn))
            end))
  end
=#
###################################################
# New version
remote_function_meta_program(nssym, sig, local_name, remote_name, params, ret) =
  let namespace = :(Val{$(nssym)}())
    :(remote_function(
           $(namespace),
           $(sig),
           $(local_name),
           $(remote_name),
           (opcode, conn, buf, $([p[2] for p in params]...)) -> begin
              initiate_rpc_call(conn, opcode, $(remote_name))
              take!(buf) # Reset the buffer just in case there was an encoding error on a previous call
              encode($(namespace), Val(:int), buf, opcode)
              $([:(encode($(namespace),
                          $(p[1])(),
                          buf,
                          $(p[2])))
                 for p in params]...)
              write(conn, take!(buf))
              complete_rpc_call(conn, opcode,
                decode($(namespace),
                       $(ret)(), conn))
            end))
  end

#=
let lang = :CS,
    str = "public Entity Sphere(Point3d c, double r)",
    (local_name, remote_name, params, ret) = parse_signature(Val(lang), str)
  (Symbol(local_name), remote_function_meta_program(lang, str, local_name, remote_name, params, ret))
 end

let lang = :CS,
   str = "public Entity[][] Spheres(Point3d c, double[][] r)",
   (local_name, remote_name, params, ret) = parse_signature(Val(lang), str)
 (Symbol(local_name), remote_function_meta_program(lang, str, local_name, remote_name, params, ret))
end
=#

lang_rpc(lang, sigs) =
  [let (local_name, remote_name, params, ret) = parse_signature(Val(lang.value), str)
    (Symbol(local_name), remote_function_meta_program(lang, str, local_name, remote_name, params, ret))
   end
   for str in split(sigs, "\n") if str != ""]

#=
Given that remote apps might fail, it might be necessary to reset a connection.
This can be done either by resetting the opcodes of all remote functions or by
simply recreating all of them. This second alternative looks better because it
also allows to access multiple instances of the same remote app (as long as it)
support multiple separate connections.

=#

#=
The idea is that a particular remote application will store all of its functions
in a struct, as follows:

@remote_functions :CS """
  int add(int a, int b)
  int sub(int a, int b)
"""
=#

macro remote_functions(lang, str)
  let remotes = lang_rpc(lang, str)
    Expr(:tuple, [Expr(:(=), remote...) for remote in remotes]...)
  end
end

#=
@macroexpand @remote_functions :CS """
  int add(int a, int b)
  int sub(int a, int b)
"""
=#

# We need to detect errors by recognizing a particular value
# Note that this is parametric on namespace ond type
decode_or_error(ns::Val{NS}, t::Val{T}, c::IO, err) where {NS,T} =
  let v = decode(ns, t, c)
    v == err ? backend_error(ns, c) : v
  end

struct BackendError
  msg::String
  backtrace
end

show(io::IO, e::BackendError) =
  print(io, "Backend Error: $(e.msg)")

# In order to read arrays or process error messages,
# we need a few basic types, namely :size and :string,
# that need to be specified for each backend.

backend_error(ns::Val{NS}, c::IO) where {NS} =
  throw(BackendError(decode(ns, Val(:string), c), backtrace()))

# Encoding and decoding vectors
encode(ns::Val{NS}, t::Vector{T}, c::IO, v) where {NS,T} = begin
  sub = T()
  encode(ns, Val(:size), c, length(v))
  for e in v encode(ns, sub, c, e) end
end
decode(ns::Val{NS}, t::Vector{T}, c::IO) where {NS,T} = begin
  sub = T()
  len = decode(ns, Val(:size), c)
  [decode(ns, sub, c) for i in 1:len]
end

# Some generic conversions for C#, C++, and Python
encode(ns::Union{Val{:CS},Val{:CPP},Val{:PY}}, t::Val{:size}, c::IO, v) =
  encode(ns, Val(:int), c, v)
decode(ns::Union{Val{:CS},Val{:CPP},Val{:PY}}, t::Val{:size}, c::IO) =
  decode_or_error(ns, Val(:int), c, -1)

encode(ns::Val{:CS}, t::Val{:bool}, c::IO, v::Bool) =
  encode(ns, Val(:byte), c, v ? UInt8(1) : UInt8(0))
decode(ns::Val{:CS}, t::Val{:bool}, c::IO) =
  decode_or_error(ns, Val(:byte), c, UInt8(127)) == UInt8(1)

encode(::Union{Val{:CS},Val{:CPP},Val{:PY}}, t::Val{:byte}, c::IO, v) =
  write(c, convert(UInt8, v))
decode(::Union{Val{:CS},Val{:CPP},Val{:PY}}, t::Val{:byte}, c::IO) =
  convert(UInt8, read(c, UInt8))

# Assuming int is four bytes in C++ and Python
encode(::Union{Val{:CS},Val{:CPP},Val{:PY}}, t::Val{:int}, c::IO, v) =
  write(c, convert(Int32, v))
decode(::Union{Val{:CS},Val{:CPP},Val{:PY}}, t::Val{:int}, c::IO) =
  convert(Int, read(c, Int32))

# Assuming long is eight bytes in C++
encode(::Union{Val{:CS},Val{:CPP}}, t::Val{:long}, c::IO, v) =
  write(c, convert(Int64, v))
decode(::Union{Val{:CS},Val{:CPP}}, t::Val{:long}, c::IO) =
  convert(Int, read(c, Int64))

# Assuming float is four bytes in C++
encode(::Union{Val{:CS},Val{:CPP}}, t::Val{:float}, c::IO, v) =
  write(c, convert(Float32, v))
decode(ns::Union{Val{:CS},Val{:CPP}}, t::Val{:float}, c::IO) =
  let d = read(c, Float32)
    isnan(d) ? backend_error(ns, c) : convert(Float64, d)
  end

# Assuming float is eight bytes in Python
encode(::Val{:PY}, t::Val{:float}, c::IO, v) =
  write(c, convert(Float64, v))
decode(ns::Val{:PY}, t::Val{:float}, c::IO) =
  let d = read(c, Float64)
    isnan(d) ? backend_error(ns, c) : convert(Float64, d)
  end

# Assuming double is eight bytes in C++
encode(::Union{Val{:CS},Val{:CPP}}, t::Val{:double}, c::IO, v) =
  write(c, convert(Float64, v))
decode(ns::Union{Val{:CS},Val{:CPP}}, t::Val{:double}, c::IO) =
  let d = read(c, Float64)
    isnan(d) ? backend_error(ns, c) : d
  end

# The binary_stream we use with C++ and Python replicates C# behavior
encode(::Union{Val{:CS},Val{:CPP},Val{:PY}}, ::Union{Val{:string},Val{:String},Val{:str}}, c::IO, v) = begin
  str = string(v)
  size = length(str)
  array = UInt8[]
  while true
    byte = size & 0x7f
    size >>= 7
    if size > 0
      push!(array, byte | 0x80)
    else
      push!(array, byte)
      break
    end
  end
  write(c, array)
  write(c, str)
end

#=
decode(::Val{:CS}, ::Val{:string}, c::IO) = begin
  loop(size::Int, shift::Int) = begin
    b = convert(Int, read(c, UInt8))
    size = size | ((b & 0x7f) << shift)
    if (b & 0x80) == 0
      String(read(c, size))
    else
      loop(size, shift + 7)
    end
  end
  loop(0, 0)
end
=#
decode(ns::Union{Val{:CS},Val{:CPP},Val{:PY}}, ::Union{Val{:string},Val{:String},Val{:str}}, c::IO) = begin
  size::Int = 0
  shift::Int = 0
  while true
    b = convert(Int, read(c, UInt8))
    size = size | ((b & 0x7f) << shift)
    if (b & 0x80) == 0
      str = String(read(c, size))
      str == "This an error!" ?
        backend_error(ns, c) :
        return str
    else
      shift += 7
    end
  end
end

decode(ns::Val{:CS}, t::Val{:void}, c::IO) =
  decode_or_error(ns, Val(:byte), c, 0x7f) == 0x00

decode(ns::Val{:PY}, t::Val{:None}, c::IO) =
  decode_or_error(ns, Val(:byte), c, 0x7f) == 0x00

# Useful CS types
export Guid, Guids
const Guid = Vector{UInt8}
const Guids = Vector{Guid}

encode(::Val{:CS}, ::Val{:Guid}, c::IO, v::Guid) =
  write(c, v)
decode(ns::Val{:CS}, ::Val{:Guid}, c::IO) =
  let guid = read(c, 16)
    iszero(guid) ? backend_error(ns, c) : guid
  end

#=
@code_typed decode(Val(:CS), [Val(:Guid)], IOBuffer())
=#

# It is frequently necessary to encode/decode an abstract type that is
# implemented os some primitive type
macro encode_decode_as(ns, from, to)
  esc(
    quote
      encode(ns::Val{$ns}, ::$from, c::IO, v) = encode(ns, $to(), c, v)
      decode(ns::Val{$ns}, ::$from, c::IO) = decode(ns, $to(), c)
    end)
end

#=
There are lots of cases where doubles need to be encoded/decoded
(e.g. (points, vectors))
encode_double3(c::IO, v0::Real, v1::Real, v2::Real) = begin
    encode_double(c, v0)
    encode_double(c, v1)
    encode_double(c, v2)
end
encode_float3(c::IO, v0::Real, v1::Real, v2::Real) = begin
    encode_float(c, v0)
    encode_float(c, v1)
    encode_float(c, v2)
end
=#

encode(ns::Val{NS}, ::Val{:float3}, c::IO, (v1, v2, v3)) where {NS} = begin
  encode(ns, Val(:float), c, v1)
  encode(ns, Val(:float), c, v2)
  encode(ns, Val(:float), c, v3)
end

decode(ns::Val{NS}, ::Val{:float3}, c::IO) where {NS} = begin
  decode(ns, Val(:float), c),
  decode(ns, Val(:float), c),
  decode(ns, Val(:float), c)
end

encode(ns::Val{NS}, ::Val{:double3}, c::IO, (v1, v2, v3)) where {NS} = begin
  encode(ns, Val(:double), c, v1)
  encode(ns, Val(:double), c, v2)
  encode(ns, Val(:double), c, v3)
end

decode(ns::Val{NS}, ::Val{:double3}, c::IO) where {NS} = begin
  decode(ns, Val(:double), c),
  decode(ns, Val(:double), c),
  decode(ns, Val(:double), c)
end

decode_id(c::IO) =
  let id = decode_int(c)
    if id == -1
      backend_error(c)
    else
      id
    end
  end

encode(ns::Val{NS}, ::Val{RGB}, c::IO, v) where {NS} =
  encode(ns, Val(:float3), c, (v.red, v.green, v.blue))

decode(ns::Val{NS}, ::Val{RGB}, c::IO) where {NS} =
  rgb(decode(ns, Val(:float3), c)...)
