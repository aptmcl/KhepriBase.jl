module KhepriBase
using ColorTypes
using LinearAlgebra
using StaticArrays
using Dierckx
using Dates

import Base.+, Base.-, Base.*, Base./, Base.length
import Base.show, Base.zero, Base.iterate
import Base.convert
import Base.getindex, Base.firstindex, Base.lastindex, Base.broadcastable
import Base.union
import Base.fill
import LinearAlgebra.cross, LinearAlgebra.dot, LinearAlgebra.norm

include("Parameters.jl")
include("Utils.jl")
include("Coords.jl")
include("Paths.jl")
include("Geometry.jl")
include("Shapes.jl")
include("BIM.jl")
include("Camera.jl")
end
