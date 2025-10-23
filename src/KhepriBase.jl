module KhepriBase
using ColorTypes
using LinearAlgebra
using StaticArrays
using Dierckx
using Dates
using Sockets
using HTTP
using Base.Iterators
using Base: @kwdef

#=
Dierckx does not yet support derivatives. Until they update it,
we extend it here to have that support.
=#
export evaluate_derivative
function evaluate_derivative(spline::Spline2D, x::AbstractVector, y::AbstractVector, dx::Int, dy::Int)
    m = length(x)
    @assert length(y) == m
    xin = convert(Vector{Float64}, x)
    yin = convert(Vector{Float64}, y)
    zout = Vector{Float64}(undef, m)
    nx = length(spline.tx)
    ny = length(spline.ty)
    lwrk = (nx*ny)+(spline.kx+1)*m+(spline.ky+1)*m
    wrk = Vector{Float64}(undef, lwrk)
    ier = Ref{Int32}()
    kwrk = 2*m
    iwrk = Vector{Int32}(undef, kwrk)
    ccall((:pardeu_, Dierckx.libddierckx), Nothing,
          (Ref{Float64}, Ref{Int32},  # ty, ny
           Ref{Float64}, Ref{Int32},  # tx, nx
           Ref{Float64},              # c
           Ref{Int32}, Ref{Int32},    # ky, kx
           Ref{Int32}, Ref{Int32},    # nuy, nux
           Ref{Float64},              # y
           Ref{Float64},              # x
           Ref{Float64},              # z
           Ref{Int32},                # m
           Ref{Float64}, Ref{Int32},  # wrk, lwrk
           Ref{Int32}, Ref{Int32},    # iwrk, kwrk
           Ref{Int32}),               # ier
          spline.ty, ny,
          spline.tx, nx,
          spline.c,
          spline.ky, spline.kx,
          dy, dx,
          yin, xin, zout, m,
          wrk, lwrk,
          iwrk, kwrk,
          ier)
    ier[] == 0 || error(Dierckx._eval2d_message)
    return zout
end
function evaluate_derivative(spline::Spline2D, x::Number, y::Number, dx::Int, dy::Int)
  if dx >= spline.kx
    ϵ = 1e-9
    (evaluate(spline, x+ϵ, y) - evaluate(spline, x-ϵ, y))/2/ϵ
  elseif dy >= spline.kx
    ϵ = 1e-9
    (evaluate(spline, x, y+ϵ) - evaluate(spline, x, y-ϵ))/2/ϵ
  else
    evaluate_derivative(spline, [x], [y], dx, dy)[1]
  end
end

import Base:
    +, -, *, /, length,
    show, showerror,
    zero, iterate, convert,
    getindex, firstindex, lastindex, broadcastable,
    union, fill
import LinearAlgebra:
    cross, dot, norm

include("Types.jl")
include("Parameters.jl")
include("Utils.jl")
include("Coords.jl")
include("Regions.jl")
include("Paths.jl")
include("Geometry.jl")
include("Backend.jl")
include("Frontend.jl")
include("Shapes.jl")
include("BIM.jl")
include("Backends.jl")
#include("Materials.jl")
include("Primitives.jl")
include("Camera.jl")

export khepribase_interface_file
khepribase_interface_file() = joinpath(@__DIR__, "Interface.jl")

export resources_pathname, models_pathname, environments_pathname
const resources_pathname = joinpath(@__DIR__, "..", "resources")
const models_pathname = joinpath(resources_pathname, "models")
const environments_pathname = joinpath(resources_pathname, "environments")

# From ColorTypes
export RGB, RGBA, rgb, rgba, red, green, blue, alpha


export and_mark_deleted,
       @remote_functions,
       parse_signature,
       encode,
       decode,
       decode_or_error,
       @encode_decode_as,
       SocketBackend, WebSocketBackend, WebSocketConnection,
       current_backends,
       @remote,
       @get_remote,
       connection,
       reset_backend,
       IOBackend,
       dimension,
       ref,
       show_truss_deformation

# Ports for socket-based backends are defined here to avoid conflicts
export autocad_port, revit_port, rhino_port, unity_port, unreal_port, blender_port, freecad_port, a3dsmax_port, threejs_port
const autocad_port = 11000
const revit_port = 11001
const unity_port = 11002
const blender_port = 11003
const freecad_port = 11004
const a3dsmax_port = 11005
const unreal_port = 11010
const rhino_port = 12000
const threejs_port = 8900

function __init__()
end
end

