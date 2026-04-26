# ---- Core Types ----
# Immutable value types forming the SpaceDesc tree.
# Leaf nodes: Room, Void, Envelope
# Composite nodes: BesideX, BesideY, Above, Repeat, Grid, Scale, Mirror, Annotated

"""
    SpaceDesc

Abstract supertype for all spatial description tree nodes.
Leaf nodes represent rooms, voids, or envelopes; composite nodes combine them.
"""
abstract type SpaceDesc end

# ---- Leaf Nodes ----

"""
    Room <: SpaceDesc

A leaf node representing a named room with a use type, dimensions, and optional properties.
"""
struct Room <: SpaceDesc
  id::Symbol
  use::Symbol
  width::Float64
  depth::Float64
  height::Float64
  props::NamedTuple
end

"""
    room(id, use, w, d; height=2.8, props=(;))

Convenience constructor for [`Room`](@ref). Converts dimensions to `Float64`.
"""
room(id::Symbol, use::Symbol, w, d; height=2.8, props=(;)) =
  Room(id, use, Float64(w), Float64(d), Float64(height), props)

"""
    Void <: SpaceDesc

A leaf node representing empty space with optional width and depth.
Used as a spacer or placeholder in layouts.
"""
struct Void <: SpaceDesc
  width::Float64
  depth::Float64
end

"""
    void(w=0.0, d=0.0)

Convenience constructor for [`Void`](@ref). Defaults to zero-size empty space.
"""
void(w=0.0, d=0.0) = Void(Float64(w), Float64(d))

"""
    Envelope <: SpaceDesc

A leaf node representing a building envelope (exterior shell) with dimensions and properties.
"""
struct Envelope <: SpaceDesc
  id::Symbol
  width::Float64
  depth::Float64
  height::Float64
  props::NamedTuple
end

"""
    envelope(w, d, h; id=:envelope, props=(;))

Convenience constructor for `Envelope`. Converts dimensions to `Float64`.
"""
envelope(w, d, h; id=:envelope, props=(;)) =
  Envelope(id, Float64(w), Float64(d), Float64(h), props)

# ---- Polar Leaves ----
#
# Rectangular `Envelope` nodes live in a Cartesian `(x, y)` frame: the
# layout engine threads `(x, y)` down the tree and each subtree places
# itself relative to that cursor. Polar envelopes break this contract
# — they are absolute-positioned (their `center` is a world-space
# `Loc`) and self-contain the polar extents. That makes them useful
# for arc-shaped buildings (see the `IsembergTopDown` tutorial), but
# also means polar subtrees do not compose with rectangular
# combinators via shared `(x, y)` arithmetic; only `z` is threaded so
# that `above(polar_floor, polar_floor)` still stacks correctly.
#
# The polar metadata (`center`, radii, angles) is written into every
# placed Space's `props[:_polar]` so that [`refine`](@ref) can
# dispatch polymorphically — rebuilding a polar envelope for the
# transform rather than a rectangular one.

"""
    PolarEnvelope <: SpaceDesc

A leaf node representing a polar-bounded zone (an annular sector)
around an absolute `center` between radii `r_inner..r_outer` and
angles `theta_start..theta_end`. Used as the root of arc-shaped
designs and as the intermediate zone type produced by polar
subdivision operators.
"""
struct PolarEnvelope <: SpaceDesc
  id::Symbol
  center::Loc
  r_inner::Float64
  r_outer::Float64
  theta_start::Float64
  theta_end::Float64
  height::Float64
  use::Symbol
  props::NamedTuple
  n_arc::Int
end

"""
    polar_envelope(center, r_inner, r_outer, theta_start, theta_end, height;
                   id=:envelope, use=:envelope, props=(;), n_arc=16)

Convenience constructor for [`PolarEnvelope`](@ref). Converts numeric
fields to `Float64` and wraps `center` into a `Loc` if needed.
"""
polar_envelope(center, r_inner, r_outer, theta_start, theta_end, height;
               id::Symbol=:envelope, use::Symbol=:envelope,
               props::NamedTuple=(;), n_arc::Integer=16) =
  PolarEnvelope(id, center, Float64(r_inner), Float64(r_outer),
                Float64(theta_start), Float64(theta_end), Float64(height),
                use, props, Int(n_arc))

"""
    SubdividedPolar <: SpaceDesc

Splits a [`PolarEnvelope`](@ref) along its `:radial` or `:angular`
axis into named zones with proportional `ratios`. Each zone becomes
a polar Space (with `:_polar` metadata in its props).
"""
struct SubdividedPolar <: SpaceDesc
  base::SpaceDesc
  axis::Symbol              # :radial or :angular
  ratios::Vector{Float64}
  ids::Vector{Symbol}
end

"""
    subdivide_radial(base, ratios, ids)

Split a polar envelope into concentric annular bands by proportional
radial `ratios` (must sum to 1.0), naming each band from `ids`.
"""
subdivide_radial(base::SpaceDesc, ratios, ids) =
  SubdividedPolar(base, :radial, Float64.(collect(ratios)),
                  [id isa Symbol ? id : Symbol(id) for id in ids])

"""
    subdivide_angular(base, ratios, ids)

Split a polar envelope into angular wedges by proportional `ratios`
(must sum to 1.0), naming each wedge from `ids`.
"""
subdivide_angular(base::SpaceDesc, ratios, ids) =
  SubdividedPolar(base, :angular, Float64.(collect(ratios)),
                  [id isa Symbol ? id : Symbol(id) for id in ids])

"""
    PartitionedPolar <: SpaceDesc

Splits a [`PolarEnvelope`](@ref) into `count` equal parts along its
`:angular` or `:radial` axis, naming the resulting zones
`Symbol(id_prefix, "_", i)`.
"""
struct PartitionedPolar <: SpaceDesc
  base::SpaceDesc
  axis::Symbol
  count::Int
  id_prefix::Symbol
end

"""
    partition_angular(base, count, id_prefix)

Split a polar envelope into `count` equal angular wedges, naming the
zones `<id_prefix>_1`, `<id_prefix>_2`, …
"""
partition_angular(base::SpaceDesc, count::Integer, id_prefix::Symbol) =
  PartitionedPolar(base, :angular, Int(count), id_prefix)

"""
    partition_radial(base, count, id_prefix)

Split a polar envelope into `count` equal concentric annular bands,
naming the zones `<id_prefix>_1`, `<id_prefix>_2`, …
"""
partition_radial(base::SpaceDesc, count::Integer, id_prefix::Symbol) =
  PartitionedPolar(base, :radial, Int(count), id_prefix)

# ---- Composite Nodes ----

"""
    BesideX <: SpaceDesc

Composite node placing two spaces side by side along the X axis.
`align` controls depth alignment (`:start`, `:center`, or `:end`).
"""
struct BesideX <: SpaceDesc
  left::SpaceDesc
  right::SpaceDesc
  shared_wall::Bool
  align::Symbol  # :start, :center, :end
end

"""
    BesideY <: SpaceDesc

Composite node placing two spaces side by side along the Y axis (front to back).
`align` controls width alignment (`:start`, `:center`, or `:end`).
"""
struct BesideY <: SpaceDesc
  front::SpaceDesc
  back::SpaceDesc
  shared_wall::Bool
  align::Symbol
end

"""
    Above <: SpaceDesc

Composite node stacking two spaces vertically. `slab_between` controls
whether a floor slab is inserted between the two levels.
"""
struct Above <: SpaceDesc
  below::SpaceDesc
  above::SpaceDesc
  slab_between::Bool
end

"""
    Repeated <: SpaceDesc

Repeats a unit space `count` times along `axis` (`:x` or `:y`).
When `mirror_alternate` is true, every other copy is mirrored.
"""
struct Repeated <: SpaceDesc
  unit::SpaceDesc
  count::Int
  axis::Symbol        # :x or :y
  mirror_alternate::Bool
end

"""
    GridLayout <: SpaceDesc

A grid of spaces defined by a function `cell_fn(row, col) -> SpaceDesc`.
"""
struct GridLayout <: SpaceDesc
  cell_fn::Function   # (row, col) -> SpaceDesc
  rows::Int
  cols::Int
end

"""
    Scaled <: SpaceDesc

Wraps a space description with scale factors `sx` and `sy` along X and Y axes.
"""
struct Scaled <: SpaceDesc
  base::SpaceDesc
  sx::Float64
  sy::Float64
end

"""
    Mirrored <: SpaceDesc

Wraps a space description to mirror it along `axis` (`:x` or `:y`).
"""
struct Mirrored <: SpaceDesc
  base::SpaceDesc
  axis::Symbol  # :x or :y
end

"""
    HeightOverride <: SpaceDesc

Wraps a space description to override its height with a fixed value.
"""
struct HeightOverride <: SpaceDesc
  base::SpaceDesc
  height::Float64
end

"""
    PropsOverlay <: SpaceDesc

Wraps a space description to merge a `NamedTuple` of properties into every
placed space below it at layout time. Existing props on a space take
precedence; new keys are added from the overlay.
"""
struct PropsOverlay <: SpaceDesc
  base::SpaceDesc
  props::NamedTuple
end

# ---- Annotation Types ----

"""
    DesignAnnotation

Abstract supertype for annotations attached to a space description tree.
Annotations describe connections, disconnections, and other metadata.
The name is qualified to avoid clashing with `KhepriBase.Annotation`,
which is the BIM annotation type used for labels, dimensions, and so on.
"""
abstract type DesignAnnotation end

"""
    ConnectAnnotation <: DesignAnnotation

Declares a connection (door, window, or arch) between two named spaces.
"""
struct ConnectAnnotation <: DesignAnnotation
  from::Symbol
  to::Symbol
  kind::Symbol        # :door, :window, :arch
  width::Union{Float64, Nothing}
  height::Union{Float64, Nothing}
end

"""
    ConnectExteriorAnnotation <: DesignAnnotation

Declares an exterior opening (door or window) on a specific face of a space.
"""
struct ConnectExteriorAnnotation <: DesignAnnotation
  space_id::Symbol
  kind::Symbol        # :door, :window
  face::Symbol        # :north, :south, :east, :west, :auto
  count::Union{Int, Nothing}
  width::Union{Float64, Nothing}
  height::Union{Float64, Nothing}
end

"""
    DisconnectAnnotation <: DesignAnnotation

Removes any default connection between two adjacent spaces.
"""
struct DisconnectAnnotation <: DesignAnnotation
  from::Symbol
  to::Symbol
end

"""
    NoWindowsAnnotation <: DesignAnnotation

Suppresses automatic window generation for the given space.
"""
struct NoWindowsAnnotation <: DesignAnnotation
  space_id::Symbol
end

# ---- Annotated Wrapper Node ----

"""
    Annotated <: SpaceDesc

Wraps a space description with a single [`DesignAnnotation`](@ref).
Multiple annotations are represented by nesting `Annotated` nodes.
"""
struct Annotated <: SpaceDesc
  base::SpaceDesc
  annotation::DesignAnnotation
end

# ---- Subdivision Nodes ----

"""
    Subdivided <: SpaceDesc

Splits a space along `axis` into named zones with proportional `ratios`.
"""
struct Subdivided <: SpaceDesc
  base::SpaceDesc
  axis::Symbol              # :x or :y
  ratios::Vector{Float64}
  ids::Vector{Symbol}
end

"""
    Partitioned <: SpaceDesc

Splits a space into `count` equal parts along `axis`, naming them with `id_prefix`.
"""
struct Partitioned <: SpaceDesc
  base::SpaceDesc
  axis::Symbol
  count::Int
  id_prefix::Symbol
end

"""
    Carved <: SpaceDesc

Carves out a rectangular sub-region from a space at position `(x, y)` with given dimensions.
"""
struct Carved <: SpaceDesc
  base::SpaceDesc
  id::Symbol
  use::Symbol
  x::Float64
  y::Float64
  width::Float64
  depth::Float64
end

"""
    Refined <: SpaceDesc

Replaces a named zone within `base` by applying `transform(zone) -> SpaceDesc`.
"""
struct Refined <: SpaceDesc
  base::SpaceDesc
  zone_id::Symbol
  transform::Function
end

"""
    Assigned <: SpaceDesc

Assigns a use type and properties to a named zone within `base`.
"""
struct Assigned <: SpaceDesc
  base::SpaceDesc
  zone_id::Symbol
  use::Symbol
  props::NamedTuple
end

"""
    SubdivideRemaining <: SpaceDesc

Given a base containing a single central carved hole (via [`carve`](@ref)),
produce named perimeter blocks around it. Each `(id, position)` pair places
a zone at `:north`, `:south`, `:east`, or `:west` of the hole, covering the
remaining area in that direction.
"""
struct SubdivideRemaining <: SpaceDesc
  base::SpaceDesc
  blocks::Vector{Tuple{Symbol, Symbol}}  # (block_id, position)
end

# ---- Grid helpers ----
# Per-column widths (max cell width in each column) and per-row depths (max
# cell depth in each row). Used by both the tree queries and the layout
# engine so heterogeneous grids stay consistent between size and placement.

_grid_col_widths(g::GridLayout) =
  [maximum(desc_width(g.cell_fn(r, c)) for r in 1:g.rows) for c in 1:g.cols]

_grid_row_depths(g::GridLayout) =
  [maximum(desc_depth(g.cell_fn(r, c)) for c in 1:g.cols) for r in 1:g.rows]

# ---- Tree Query Utilities ----

"""
    desc_width(desc::SpaceDesc) -> Float64

Compute the total width (X extent) of a space description tree.
"""
desc_width(r::Room) = r.width
desc_width(v::Void) = v.width
desc_width(e::Envelope) = e.width
desc_width(b::BesideX) = desc_width(b.left) + desc_width(b.right)
desc_width(b::BesideY) = max(desc_width(b.front), desc_width(b.back))
desc_width(a::Above) = max(desc_width(a.below), desc_width(a.above))
desc_width(r::Repeated) = r.axis == :x ? desc_width(r.unit) * r.count : desc_width(r.unit)
desc_width(g::GridLayout) = sum(_grid_col_widths(g))
desc_width(s::Scaled) = desc_width(s.base) * s.sx
desc_width(m::Mirrored) = desc_width(m.base)
desc_width(h::HeightOverride) = desc_width(h.base)
desc_width(p::PropsOverlay) = desc_width(p.base)
desc_width(a::Annotated) = desc_width(a.base)
desc_width(s::Subdivided) = desc_width(s.base)
desc_width(p::Partitioned) = desc_width(p.base)
desc_width(c::Carved) = desc_width(c.base)
desc_width(r::Refined) = desc_width(r.base)
desc_width(a::Assigned) = desc_width(a.base)
desc_width(sr::SubdivideRemaining) = desc_width(sr.base)
# Polar subtrees are self-contained (absolute `center`), so they
# contribute 0 to Cartesian `(x, y)` budgeting.
desc_width(::PolarEnvelope) = 0.0
desc_width(::SubdividedPolar) = 0.0
desc_width(::PartitionedPolar) = 0.0

"""
    desc_depth(desc::SpaceDesc) -> Float64

Compute the total depth (Y extent) of a space description tree.
"""
desc_depth(r::Room) = r.depth
desc_depth(v::Void) = v.depth
desc_depth(e::Envelope) = e.depth
desc_depth(b::BesideX) = max(desc_depth(b.left), desc_depth(b.right))
desc_depth(b::BesideY) = desc_depth(b.front) + desc_depth(b.back)
desc_depth(a::Above) = max(desc_depth(a.below), desc_depth(a.above))
desc_depth(r::Repeated) = r.axis == :y ? desc_depth(r.unit) * r.count : desc_depth(r.unit)
desc_depth(g::GridLayout) = sum(_grid_row_depths(g))
desc_depth(s::Scaled) = desc_depth(s.base) * s.sy
desc_depth(m::Mirrored) = desc_depth(m.base)
desc_depth(h::HeightOverride) = desc_depth(h.base)
desc_depth(p::PropsOverlay) = desc_depth(p.base)
desc_depth(a::Annotated) = desc_depth(a.base)
desc_depth(s::Subdivided) = desc_depth(s.base)
desc_depth(p::Partitioned) = desc_depth(p.base)
desc_depth(c::Carved) = desc_depth(c.base)
desc_depth(r::Refined) = desc_depth(r.base)
desc_depth(a::Assigned) = desc_depth(a.base)
desc_depth(sr::SubdivideRemaining) = desc_depth(sr.base)
desc_depth(::PolarEnvelope) = 0.0
desc_depth(::SubdividedPolar) = 0.0
desc_depth(::PartitionedPolar) = 0.0

"""
    desc_height(desc::SpaceDesc) -> Float64

Compute the total height (Z extent) of a space description tree.
"""
desc_height(r::Room) = r.height
desc_height(v::Void) = 0.0
desc_height(e::Envelope) = e.height
desc_height(b::BesideX) = max(desc_height(b.left), desc_height(b.right))
desc_height(b::BesideY) = max(desc_height(b.front), desc_height(b.back))
desc_height(a::Above) = desc_height(a.below) + desc_height(a.above)
desc_height(r::Repeated) = r.axis == :z ? desc_height(r.unit) * r.count : desc_height(r.unit)
desc_height(g::GridLayout) =
  maximum(desc_height(g.cell_fn(r, c)) for r in 1:g.rows, c in 1:g.cols)
desc_height(s::Scaled) = desc_height(s.base)
desc_height(m::Mirrored) = desc_height(m.base)
desc_height(h::HeightOverride) = h.height
desc_height(p::PropsOverlay) = desc_height(p.base)
desc_height(a::Annotated) = desc_height(a.base)
desc_height(s::Subdivided) = desc_height(s.base)
desc_height(p::Partitioned) = desc_height(p.base)
desc_height(c::Carved) = desc_height(c.base)
desc_height(r::Refined) = desc_height(r.base)
desc_height(a::Assigned) = desc_height(a.base)
desc_height(sr::SubdivideRemaining) = desc_height(sr.base)
desc_height(pe::PolarEnvelope) = pe.height
desc_height(sp::SubdividedPolar) = desc_height(sp.base)
desc_height(pp::PartitionedPolar) = desc_height(pp.base)

"""
    collect_ids(desc::SpaceDesc) -> Vector{Symbol}

Recursively collect all room and zone ids from a space description tree.
"""
collect_ids(r::Room) = Symbol[r.id]
collect_ids(::Void) = Symbol[]
collect_ids(e::Envelope) = Symbol[e.id]
collect_ids(b::BesideX) = vcat(collect_ids(b.left), collect_ids(b.right))
collect_ids(b::BesideY) = vcat(collect_ids(b.front), collect_ids(b.back))
collect_ids(a::Above) = vcat(collect_ids(a.below), collect_ids(a.above))
collect_ids(r::Repeated) = collect_ids(r.unit)  # base ids before namespace scoping
collect_ids(g::GridLayout) =
  reduce(vcat, collect_ids(g.cell_fn(r, c)) for r in 1:g.rows, c in 1:g.cols; init=Symbol[])
collect_ids(s::Scaled) = collect_ids(s.base)
collect_ids(m::Mirrored) = collect_ids(m.base)
collect_ids(h::HeightOverride) = collect_ids(h.base)
collect_ids(p::PropsOverlay) = collect_ids(p.base)
collect_ids(a::Annotated) = collect_ids(a.base)
collect_ids(s::Subdivided) = vcat(collect_ids(s.base), s.ids)
collect_ids(p::Partitioned) = vcat(collect_ids(p.base), [Symbol(p.id_prefix, "_", i) for i in 1:p.count])
collect_ids(c::Carved) = vcat(collect_ids(c.base), Symbol[c.id])
collect_ids(r::Refined) = collect_ids(r.base)
collect_ids(a::Assigned) = collect_ids(a.base)
collect_ids(sr::SubdivideRemaining) = vcat(collect_ids(sr.base), [b[1] for b in sr.blocks])
collect_ids(pe::PolarEnvelope) = Symbol[pe.id]
collect_ids(sp::SubdividedPolar) = vcat(collect_ids(sp.base), sp.ids)
collect_ids(pp::PartitionedPolar) =
  vcat(collect_ids(pp.base), [Symbol(pp.id_prefix, "_", i) for i in 1:pp.count])

"""
    collect_annotations(desc::SpaceDesc) -> Vector{DesignAnnotation}

Recursively collect all annotations from a space description tree.
"""
collect_annotations(::Room) = DesignAnnotation[]
collect_annotations(::Void) = DesignAnnotation[]
collect_annotations(::Envelope) = DesignAnnotation[]
collect_annotations(b::BesideX) = vcat(collect_annotations(b.left), collect_annotations(b.right))
collect_annotations(b::BesideY) = vcat(collect_annotations(b.front), collect_annotations(b.back))
collect_annotations(a::Above) = vcat(collect_annotations(a.below), collect_annotations(a.above))
collect_annotations(r::Repeated) = collect_annotations(r.unit)
collect_annotations(g::GridLayout) = DesignAnnotation[]
collect_annotations(s::Scaled) = collect_annotations(s.base)
collect_annotations(m::Mirrored) = collect_annotations(m.base)
collect_annotations(h::HeightOverride) = collect_annotations(h.base)
collect_annotations(p::PropsOverlay) = collect_annotations(p.base)
collect_annotations(a::Annotated) = vcat(DesignAnnotation[a.annotation], collect_annotations(a.base))
collect_annotations(s::Subdivided) = collect_annotations(s.base)
collect_annotations(p::Partitioned) = collect_annotations(p.base)
collect_annotations(c::Carved) = collect_annotations(c.base)
collect_annotations(r::Refined) = collect_annotations(r.base)
collect_annotations(a::Assigned) = collect_annotations(a.base)
collect_annotations(sr::SubdivideRemaining) = collect_annotations(sr.base)
collect_annotations(::PolarEnvelope) = DesignAnnotation[]
collect_annotations(sp::SubdividedPolar) = collect_annotations(sp.base)
collect_annotations(pp::PartitionedPolar) = collect_annotations(pp.base)

# ---- Tree Update Utility ----

"""
    update_room_by_id(desc, id, f)

Walk a [`SpaceDesc`](@ref) tree and return a new tree with the `Room` or
`Carved` node whose id matches `id` rewritten by `f(node) -> node`. All
other nodes are preserved by structure. Cells inside `GridLayout` and
zone targets inside `Refined` are opaque (their function argument isn't
inspected) and are passed through unchanged.
"""
update_room_by_id(desc, id, f) = _walk_update(desc, id, f)

_walk_update(r::Room, id, f) = r.id == id ? f(r) : r
_walk_update(v::Void, _, _) = v
_walk_update(e::Envelope, id, f) = e.id == id ? f(e) : e
_walk_update(b::BesideX, id, f) =
  BesideX(_walk_update(b.left, id, f), _walk_update(b.right, id, f),
          b.shared_wall, b.align)
_walk_update(b::BesideY, id, f) =
  BesideY(_walk_update(b.front, id, f), _walk_update(b.back, id, f),
          b.shared_wall, b.align)
_walk_update(a::Above, id, f) =
  Above(_walk_update(a.below, id, f), _walk_update(a.above, id, f), a.slab_between)
_walk_update(r::Repeated, id, f) =
  Repeated(_walk_update(r.unit, id, f), r.count, r.axis, r.mirror_alternate)
_walk_update(g::GridLayout, _, _) = g
_walk_update(s::Scaled, id, f) = Scaled(_walk_update(s.base, id, f), s.sx, s.sy)
_walk_update(m::Mirrored, id, f) = Mirrored(_walk_update(m.base, id, f), m.axis)
_walk_update(h::HeightOverride, id, f) =
  HeightOverride(_walk_update(h.base, id, f), h.height)
_walk_update(p::PropsOverlay, id, f) =
  PropsOverlay(_walk_update(p.base, id, f), p.props)
_walk_update(a::Annotated, id, f) =
  Annotated(_walk_update(a.base, id, f), a.annotation)
_walk_update(s::Subdivided, id, f) =
  Subdivided(_walk_update(s.base, id, f), s.axis, s.ratios, s.ids)
_walk_update(p::Partitioned, id, f) =
  Partitioned(_walk_update(p.base, id, f), p.axis, p.count, p.id_prefix)
_walk_update(c::Carved, id, f) = c.id == id ? f(c) :
  Carved(_walk_update(c.base, id, f), c.id, c.use, c.x, c.y, c.width, c.depth)
_walk_update(r::Refined, id, f) =
  Refined(_walk_update(r.base, id, f), r.zone_id, r.transform)
_walk_update(a::Assigned, id, f) =
  Assigned(_walk_update(a.base, id, f), a.zone_id, a.use, a.props)
_walk_update(sr::SubdivideRemaining, id, f) =
  SubdivideRemaining(_walk_update(sr.base, id, f), sr.blocks)
_walk_update(pe::PolarEnvelope, id, f) = pe.id == id ? f(pe) : pe
_walk_update(sp::SubdividedPolar, id, f) =
  SubdividedPolar(_walk_update(sp.base, id, f), sp.axis, sp.ratios, sp.ids)
_walk_update(pp::PartitionedPolar, id, f) =
  PartitionedPolar(_walk_update(pp.base, id, f), pp.axis, pp.count, pp.id_prefix)
