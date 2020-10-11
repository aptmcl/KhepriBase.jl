module KhepriBase
using ColorTypes
using LinearAlgebra
using StaticArrays
using Dierckx
using Dates
using Sockets

import Base:
    +, -, *, /, length,
    show, showerror,
    zero, iterate, convert,
    getindex, firstindex, lastindex, broadcastable,
    union, fill
import LinearAlgebra:
    cross, dot, norm

include("Parameters.jl")
include("Utils.jl")
include("Coords.jl")
include("Paths.jl")
include("Geometry.jl")
include("Backends.jl")
include("Shapes.jl")
include("Primitives.jl")
include("BIM.jl")
include("Camera.jl")

# From ColorTypes
export RGB, RGBA, rgb


export encode, decode, @encode_decode_as
end
