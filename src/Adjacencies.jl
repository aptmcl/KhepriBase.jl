#####################################################################
# Adjacencies — find shared edges between placed spaces on one level.
#
# The core edge-classification algorithm lives in `Spaces.jl`
# (`classify_all_edges`, `classify_edge`, `shared_boundary`,
# `exterior_edges`, `neighbors`) and operates on `Loc`-typed vertex
# coordinates. This file is the thin record-shape adapter on top:
# it flattens those classified segments into the
# `AdjacencyRelation{Symbol, NTuple{2, Float64}}` shape consumed by
# AA's generator and the declarative engine. Keeping the math in one
# place (Spaces.jl) means there is now a single authority for "which
# edges are shared between which spaces."

export AdjacencyRelation, detect_adjacencies, adjacencies

"""
    AdjacencyRelation

Records that two spaces share an edge on the same level. `space_b`
is `nothing` when the shared edge faces the exterior.
"""
struct AdjacencyRelation
  space_a::Symbol
  space_b::Union{Symbol, Nothing}
  shared_edge::NTuple{2, NTuple{2, Float64}}
  level_z::Float64
end

"""
    adjacencies(l::Layout)

Compute per-storey adjacency relations for a `Layout` on demand.
Each entry is an [`AdjacencyRelation`](@ref) stamped with the
storey's z-elevation.
"""
function adjacencies(l::Layout)
  out = AdjacencyRelation[]
  for s in l.storeys
    append!(out, _storey_adjacencies(s, storey_z(s)))
  end
  out
end

"""
    detect_adjacencies(spaces)

Find shared edges between placed spaces grouped by z-elevation.
`spaces` is any iterable of `Space` (e.g. a `Dict` of
`Symbol => Space` or a bare `Vector`). Spaces on different storey
elevations never share an edge.

Prefer [`adjacencies`](@ref) when you already have a `Layout` — this
form exists for callers who hold a loose collection of spaces.
"""
function detect_adjacencies(spaces)
  groups = Dict{Float64, Vector{Space}}()
  for sp in _iter_spaces(spaces)
    z = round(sp.origin_z, digits=6)
    push!(get!(groups, z, Space[]), sp)
  end
  out = AdjacencyRelation[]
  for (z, members) in groups
    append!(out, _storey_adjacencies(_synthetic_storey(members, z), z))
  end
  out
end

# Normalise "iterable of Spaces" to a Space iterator. A `Dict` yields
# values; anything else is assumed to iterate spaces directly.
_iter_spaces(s::AbstractDict) = values(s)
_iter_spaces(s) = s

# A minimal in-memory Storey built from a Space list at elevation `z`,
# used by `detect_adjacencies` to funnel into `classify_all_edges`.
# `height`, `wall_family`, `slab_family`, `generate_slabs` are never
# inspected by the adjacency pass, so defaults are safe.
_synthetic_storey(members::Vector{Space}, z::Float64) =
  Storey(members, SpaceConnection[], level(z), 0,
         default_wall_family(), default_slab_family(), false)

# Run `classify_all_edges` on a storey and convert each classified
# segment into an `AdjacencyRelation`. `Loc` coordinates are projected
# down to world-xy `NTuple{2, Float64}`.
function _storey_adjacencies(storey::Storey, z::Real)
  out = AdjacencyRelation[]
  tol = collinearity_tolerance()
  for (p1, p2, kind, sp_a, sp_b) in classify_all_edges(storey, tol)
    b_id = (kind == :interior && !isnothing(sp_b)) ? sp_b.id : nothing
    # classify_all_edges dedupes interior edges via objectid(sp_a) <
    # objectid(sp_b); emit only one record per shared edge here too,
    # plus separate AdjacencyRelations for each exterior edge.
    if kind == :interior
      push!(out, AdjacencyRelation(sp_a.id, b_id, _xy_pair(p1, p2), Float64(z)))
    else
      push!(out, AdjacencyRelation(sp_a.id, nothing, _xy_pair(p1, p2), Float64(z)))
    end
  end
  out
end

# Two `Loc`s → an ((x1, y1), (x2, y2)) tuple in world coordinates.
_xy_pair(p1, p2) =
  ((cx(in_world(p1)), cy(in_world(p1))),
   (cx(in_world(p2)), cy(in_world(p2))))
