#####################################################################
# Designs — declarative description of architectural layouts
#
# A `Design` is an immutable tree whose leaves are rooms, voids, and
# envelopes, and whose internal nodes are composition (`beside`,
# `above`, `grid`, `repeat_unit`), transformation (`scale`, `mirror`,
# `with_height`, `with_props`), annotation (`connect`, `disconnect`,
# `no_windows`), and top-down subdivision (`subdivide_x`,
# `partition_x`, `carve`, `refine`, `assign`, `subdivide_remaining`,
# `split_x`). The tree compiles to a Level 1 `Layout` through the
# layout engine — for now still living in AlgorithmicArchitecture.jl,
# but destined to move here in a subsequent refactor that also merges
# AA's `PlacedSpace` with KhepriBase's `Space`.
#
# This module owns only the *declarative* layer — the tree, the
# combinators, the tree queries. Architectural opinions (typologies,
# building codes, the architectural constraint library, the element
# generation pipeline) stay in AA so KhepriBase remains domain-neutral.
#
# The historical file organisation from AA is preserved under
# `src/Designs/`; this top-level file is the public module surface
# that `include`s them and re-exports everything.
#
# See also `concepts/levels_of_abstraction.md` for how this Level 2
# slots above the `Layout` (Level 1) and BIM primitives (Level 0).

include("Designs/types.jl")
include("Designs/combinators.jl")
include("Designs/operators.jl")

# ---- Tree types ----
export SpaceDesc, Room, Void, Envelope
export BesideX, BesideY, Above, Repeated, GridLayout
export Scaled, Mirrored, HeightOverride, PropsOverlay, Annotated
export Subdivided, Partitioned, Carved, Refined, Assigned, SubdivideRemaining
export PolarEnvelope, SubdividedPolar, PartitionedPolar

# ---- Annotation types ----
export DesignAnnotation, ConnectAnnotation, ConnectExteriorAnnotation
export DisconnectAnnotation, NoWindowsAnnotation

# ---- Leaf constructors ----
export room, void, envelope, polar_envelope

# ---- Combinators ----
export beside, beside_x, beside_y, above
export repeat_unit
export scale, mirror_x, mirror_y, with_height, with_props
export tag_wall_family, tag_slab_family
export connect, connect_exterior, disconnect, no_windows

# ---- Subdivision ----
export subdivide_x, subdivide_y, split_x, split_y, partition_x, partition_y
export carve, refine, assign, assign_all, subdivide_remaining
export subdivide_radial, subdivide_angular, partition_angular, partition_radial

# ---- Tree queries ----
export desc_width, desc_depth, desc_height
export collect_ids, collect_annotations, update_room_by_id
