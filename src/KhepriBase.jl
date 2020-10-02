module KhepriBase
using ColorTypes
using LinearAlgebra
using StaticArrays
using Dierckx
using Dates
using Sockets

import Base: +, -, *, /, length, show, zero, iterate, convert
import Base: getindex, firstindex, lastindex, broadcastable
import Base: union, fill
import LinearAlgebra: cross, dot, norm

include("Parameters.jl")
include("Utils.jl")
include("Coords.jl")
include("Paths.jl")
include("Geometry.jl")
include("Shapes.jl")
include("BIM.jl")
include("Camera.jl")

# From ColorTypes
export RGB, RGBA, rgb
end
