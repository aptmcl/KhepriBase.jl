module KhepriBase
using ColorTypes
using LinearAlgebra
using StaticArrays
using Dierckx
using Dates
using Sockets
using Base.Iterators

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
include("Regions.jl")
include("Paths.jl")
include("Geometry.jl")
include("Backends.jl")
include("Shapes.jl")
include("Primitives.jl")
include("BIM.jl")
include("Camera.jl")

export khepribase_interface_file
khepribase_interface_file() = joinpath(@__DIR__, "Interface.jl")

# From ColorTypes
export RGB, RGBA, rgb


export @remote_functions,
       parse_signature,
       encode,
       decode,
       @encode_decode_as,
       SocketBackend,
       create_backend_connection,
       current_backends,
       @remote,
       @get_remote,
       connection,
       reset_backend,
       IOBufferBackend,
       backend_frame_at,
       acad_revolution,
       backend_bounding_box,
       backend_cylinder,
       backend_delete_shapes,
       backend_fill_curves,
       backend_loft_curve_point,
       backend_loft_curves,
       backend_loft_surface_point,
       backend_loft_surfaces,
       backend_map_division,
       backend_node_displacement_function,
       backend_panel,
       backend_sphere,
       backend_regular_pyramid,
       backend_revolve_curve,
       backend_revolve_point,
       backend_revolve_surface,
       backend_surface_boundary,
       backend_surface_domain,
       backend_truss_analysis,
       captured_shape,
       captured_shapes,
       create_block,
       dimension,
       generate_captured_shape,
       generate_captured_shapes,
       pre_selected_shapes_from_se,
       ref,
       save_file,
       shape_from_ref,
       show_truss_deformation,
       slice_ref

end
