#####################################################################
# Layout — Space-first architectural layout system
#
# Architects think in terms of spaces (rooms) and connections (doors,
# windows, arches), not individual walls. This module lets users
# describe a building as a multi-storey `Layout` whose `Storey`s carry
# spaces and connections, then automatically generates walls, doors,
# windows, and slabs via `build`.
#
# Inspired by IFC's IfcSpace / IfcRelSpaceBoundary model: `build`
# produces descriptive `BuildResult`s that persist space-to-element
# boundary relationships, enabling introspection and constraint
# validation.
#
# Architecture note — KhepriBase models spaces at three increasingly
# abstract levels:
#
#   Level 0 — BIM primitives. `wall(path, bl, tl, fam)`,
#             `slab(region, lv, fam)`, `add_door(wall, loc, fam)`, …
#             in `BIM.jl`. Imperative, direct placement.
#   Level 1 — Layout (this file). A mutable `Layout` holds one or
#             more `Storey`s, each carrying spaces defined by explicit
#             closed-path boundaries plus explicit connections.
#             `build(layout)` compiles it down to Level 0.
#   Level 2 — declarative Design (planned extraction from
#             AlgorithmicArchitecture.jl). A composable tree of
#             rooms, combinators, and transforms; `compile(design)`
#             produces a Level 1 Layout.
#
# Each level is independently usable. The analogy is Julia's Vector:
# you can mutate it with `push!` or describe it with a comprehension;
# both produce the same type.

export Space, SpaceConnection, SpaceBoundary, BuildResult, LayoutBuildResult,
       Storey, Layout,
       layout, storey, floor_plan,
       add_storey!, add_space, add_arch, add_rule,
       build,
       space_area, space_perimeter, space_origin,
       space_width, space_depth, space_height, space_name,
       spaces, find_space, storey_z, storey_index, boundary_xy,
       polar_sector_path,
       shared_boundary, exterior_edges, neighbors,
       space_boundaries, space_walls, space_doors, space_windows,
       bounding_spaces, adjacent_spaces

#=== Data Structures ===#

#=
A named bounded area on one storey (cf. IFC IfcSpace).

This is the unifying type: what the imperative API produces through
`add_space(storey, name, boundary; kind)` and what the declarative
engine (`layout(::SpaceDesc)`) produces by placing a `Room` at
computed coordinates. The previous split between `Space` and
`PlacedSpace` collapses here — `boundary` is always concrete, and
`height`/`props` carry the per-space information that the declarative
path used to keep in `PlacedSpace`.

Fields:
- `id` — canonical Symbol identifier. Strings coming from the
  imperative API are converted at the constructor.
- `kind` — use classification (`:bedroom`, `:corridor`, …), similar
  to IfcSpaceTypeEnum.
- `boundary` — closed polygon describing the floor-plan footprint.
- `height` — floor-to-ceiling clearance; per-space so that one storey
  can mix ceiling heights (storey `height` is the construction
  envelope, `space.height` is the interior ceiling).
- `props` — user-defined metadata (family tags, room numbers, etc.).

`name(space)` returns `String(space.id)` for IFC export.
=#
"A placed bounded area on one storey. Unified imperative/declarative target."
struct Space
  id::Symbol
  kind::Symbol
  boundary::ClosedPath
  height::Float64
  props::NamedTuple
  origin_z::Float64      # elevation of this space's floor (redundant
                          # with the containing Storey's level.height,
                          # carried here so code that iterates flat
                          # space lists can filter by z without a
                          # back-pointer)
end

# Convenience constructor — accepts a String name (imperative API) or
# a Symbol id (declarative API) and fills sensible defaults.
Space(id_or_name, kind, boundary; height=default_level_to_level_height(),
      props=(;), origin_z=0.0) =
  Space(id_or_name isa Symbol ? id_or_name : Symbol(id_or_name),
        kind, boundary, Float64(height), props, Float64(origin_z))

#=
Canonical computed properties on `Space`. `Space` stores only the
essentials (`id`, `kind`, `boundary`, `height`, `props`, `origin_z`)
and derives bounding-box and IFC-name views on demand. Keeping them
as properties (rather than only functions) matches the architectural
mental model — a rectangular room does have a width, depth, and
origin — and means `sp.width` reads the same whether the boundary
was supplied imperatively or produced by the declarative engine.
Non-rectangular boundaries still report bbox-sized values, so these
names stay meaningful under future relaxations.

`sp.use` is kept as an alias for `sp.kind`: the IFC-derived `kind`
is the canonical field, but `use` is the term architects reach for
when writing constraints (``sp.use == :bedroom``).
=#
function Base.getproperty(sp::Space, s::Symbol)
  s === :name      ? String(getfield(sp, :id)) :
  s === :use       ? getfield(sp, :kind) :
  s === :origin_x  ? space_origin(sp)[1] :
  s === :origin_y  ? space_origin(sp)[2] :
  s === :width     ? space_width(sp) :
  s === :depth     ? space_depth(sp) :
  getfield(sp, s)
end
# Make the computed-property names visible to `propertynames(sp)` so
# REPL tab completion advertises them alongside the real fields.
Base.propertynames(sp::Space) =
  (:id, :kind, :boundary, :height, :props, :origin_z,
   :name, :use, :origin_x, :origin_y, :width, :depth)

"IFC-style name for a space (String form of its id)."
space_name(sp::Space) = String(sp.id)

#=
`sill` is the vertical offset of the opening's bottom edge from the
floor of the hosting storey, in metres. Doors default to 0 (they
start at the floor); windows default to `default_window_sill_height()`
(0.9 m residential default). Arches are zero-sill by definition — they
elide the wall, not host a floating opening.
=#
"A connection between two spaces (door/window/arch) or between a space and the exterior."
struct SpaceConnection
  kind::Symbol                    # :door, :window, :arch
  space_a::Space
  space_b::Union{Space, Symbol}   # Space or :exterior
  family::Union{Family, Nothing}
  loc::Union{Loc, Nothing}        # World-space point on the boundary edge
  sill::Real                      # metres above floor; 0 for doors/arches
end

"IFC-style `IfcRelSpaceBoundary` record: which element bounds which space on which side, with the shared edge's endpoints."
struct SpaceBoundary
  space::Space
  element                    # Wall, Door, Window, or nothing (for arches)
  kind::Symbol               # :physical (wall), :virtual (opening/arch)
  side::Symbol               # :interior, :exterior
  related_space::Union{Space, Nothing}
  p1::Loc
  p2::Loc
end

#=
One storey — the building's horizontal slice at a single elevation.
Carries everything needed to build that slice into BIM elements:
spaces with their closed-path boundaries, interior/exterior
connections, the BIM `Level` where elements attach, storey height,
and the default wall/slab families.

`rules` lives on the containing `Layout`, not the `Storey`, because
validation constraints often span storeys (e.g. vertical alignment).
=#
"One horizontal slice of a building: spaces, connections, BIM level, height, and default families."
mutable struct Storey
  spaces::Vector{Space}
  connections::Vector{SpaceConnection}
  level::Level              # BIM level at which this storey sits
  height::Real              # storey height (floor-to-floor)
  wall_family::WallFamily   # default wall family for this storey
  slab_family::SlabFamily
  generate_slabs::Bool
end

#=
The unifying container: a building as a stack of `Storey`s plus the
validation constraints to check once it is built. A single-storey
`Layout` is what the previous `FloorPlan` was; a multi-storey
`Layout` carries vertical structure that applied imperatively
required multiple plans and hand-managed levels.
=#
#=
The `_index` field is a lazily-built `Dict{Symbol, Tuple{Storey, Space}}`
used to accelerate `find_space(l, id)` and `_storey_of(l, space)` to
O(1). It is built on first lookup and torn down by `_invalidate_index!`
on every mutation that goes through the public API (`add_storey!`,
`add_space(::Layout, …)`, …). Callers who poke at `l.storeys` or a
`Storey.spaces` vector directly should call
`KhepriBase._invalidate_index!(l)` afterwards so subsequent lookups
don't hit a stale cache.
=#
"A building as a stack of `Storey`s plus the validation `rules` and `annotations` applied to it."
mutable struct Layout
  storeys::Vector{Storey}
  rules::Vector{Constraint}
  annotations::Vector{Annotation}
  _index::Union{Nothing, Dict{Symbol, Tuple{Storey, Space}}}
  Layout(storeys, rules, annotations) =
    new(storeys, rules, annotations, nothing)
end

# Build the id→(storey, space) map on first access and cache it on
# the Layout. Subsequent lookups are O(1) until a mutation tears the
# cache down via `_invalidate_index!`.
function _ensure_index(l::Layout)
  isnothing(l._index) || return l._index
  idx = Dict{Symbol, Tuple{Storey, Space}}()
  for s in l.storeys, sp in s.spaces
    idx[sp.id] = (s, sp)
  end
  l._index = idx
  idx
end

"Drop the cached space index on a `Layout`; the next lookup rebuilds it."
_invalidate_index!(l::Layout) = (l._index = nothing; l)

#=
A `Layout`'s real surface is its `storeys`, `rules`, and
`annotations` fields; space / level / adjacency views are obtained
through the explicit accessors `spaces(l)`, `l.storeys`, and
`adjacencies(l)`. (The previous `l.spaces` / `l.levels` /
`l.adjacencies` property shims were removed once every caller
migrated to the first-class forms.)
=#

# The output of `build(storey)`. Contains the BIM elements produced by
"Output of `build(storey)`: the emitted BIM elements plus the IFC-style `SpaceBoundary` map. Supports tuple destructuring: `walls, doors, windows, slabs = build(storey)`."
struct BuildResult
  storey::Storey
  walls::Vector
  doors::Vector
  windows::Vector
  slabs::Vector
  boundaries::Vector{SpaceBoundary}
end

Base.iterate(r::BuildResult, state=1) =
  state == 1 ? (r.walls, 2) :
  state == 2 ? (r.doors, 3) :
  state == 3 ? (r.windows, 4) :
  state == 4 ? (r.slabs, 5) :
  nothing
Base.length(::BuildResult) = 4

Base.show(io::IO, r::BuildResult) =
  print(io, "BuildResult($(length(r.storey.spaces)) spaces, $(length(r.walls)) walls, ",
            "$(length(r.doors)) doors, $(length(r.windows)) windows, ",
            "$(length(r.slabs)) slabs, $(length(r.boundaries)) boundaries)")

#=== Constructors ===#

"Create an empty `Storey` with default level, height, and wall/slab families."
storey(; level = default_level(),
         height = default_level_to_level_height(),
         wall_family = default_wall_family(),
         slab_family = default_slab_family(),
         generate_slabs = true) =
  Storey(Space[], SpaceConnection[], level, height,
         wall_family, slab_family, generate_slabs)

"Wrap one or more `Storey`s in a `Layout` plus validation `rules` and `annotations`."
layout(storeys::Storey...; rules = Constraint[], annotations = Annotation[]) =
  Layout(collect(storeys), collect(rules), collect(annotations))

"Shortcut for the common one-storey case; returns a `Layout` with a single `Storey` that subsequent `add_space` / `add_door` calls route to."
floor_plan(; level = default_level(),
             height = default_level_to_level_height(),
             wall_family = default_wall_family(),
             slab_family = default_slab_family(),
             generate_slabs = true,
             rules = Constraint[],
             annotations = Annotation[]) =
  Layout([storey(; level, height, wall_family, slab_family, generate_slabs)],
         collect(rules), collect(annotations))

"Append a `Storey` to a `Layout`. Defaults the new storey's level to one above the topmost existing storey."
function add_storey!(l::Layout;
                     level = nothing,
                     height = default_level_to_level_height(),
                     wall_family = default_wall_family(),
                     slab_family = default_slab_family(),
                     generate_slabs = true)
  lv = if !isnothing(level)
    level
  elseif isempty(l.storeys)
    default_level()
  else
    let top = last(l.storeys)
      upper_level(top.level, top.height)
    end
  end
  s = storey(; level=lv, height, wall_family, slab_family, generate_slabs)
  push!(l.storeys, s)
  _invalidate_index!(l)
  s
end

#=== Mutators ===#

"""
    add_space(storey_or_layout, name, boundary; kind=:space)

Create a `Space` with the given `name`, `boundary`, and `kind`, and
add it to the target. Passing a `Storey` targets that storey directly;
passing a `Layout` routes to its last storey (the common single-storey
case) and invalidates the space-lookup cache.
"""
add_space(s::Storey, name, boundary; kind = :space) =
  let sp = Space(name, kind, boundary)
    push!(s.spaces, sp)
    sp
  end

add_space(l::Layout, name, boundary; kind = :space) =
  isempty(l.storeys) ?
    error("Layout has no storeys; call add_storey! first or build with floor_plan") :
    let sp = add_space(last(l.storeys), name, boundary; kind)
      _invalidate_index!(l)
      sp
    end

"""
    add_door(storey_or_layout, space_a, space_b; family, loc, sill=0)

Attach a `SpaceConnection` of kind `:door` between `space_a` and
`space_b` (which may be another `Space` or the symbol `:exterior`).
The `Layout` form routes to whichever storey owns `space_a`. `loc` is
required for exterior doors; interior doors default to mid-wall
placement at `build` time. Doors sit at floor level (`sill = 0`); pass
a non-zero `sill` for raised thresholds.
"""
add_door(s::Storey, space_a::Space, space_b::Union{Space, Symbol};
         family = default_door_family(), loc = nothing, sill::Real = 0) =
  let c = SpaceConnection(:door, space_a, space_b, family, loc, sill)
    push!(s.connections, c)
    c
  end

add_door(l::Layout, space_a::Space, space_b::Union{Space, Symbol}; kwargs...) =
  add_door(_storey_of(l, space_a), space_a, space_b; kwargs...)

"""
    add_window(storey_or_layout, space_a, space_b; family, loc, sill)

Analogue of [`add_door`](@ref) for windows. `sill` is the bottom-edge
elevation above the floor (metres); defaults to
[`default_window_sill_height`](@ref) (0.9 m residential sill). Pass
`sill=0` for floor-to-ceiling glazing.
"""
add_window(s::Storey, space_a::Space, space_b::Union{Space, Symbol};
           family = default_window_family(), loc = nothing,
           sill::Real = default_window_sill_height()) =
  let c = SpaceConnection(:window, space_a, space_b, family, loc, sill)
    push!(s.connections, c)
    c
  end

add_window(l::Layout, space_a::Space, space_b::Union{Space, Symbol}; kwargs...) =
  add_window(_storey_of(l, space_a), space_a, space_b; kwargs...)

"Attach an `:arch` `SpaceConnection` between two spaces: the shared wall is suppressed at `build` time so the arch becomes an opening."
add_arch(s::Storey, space_a::Space, space_b::Space) =
  let c = SpaceConnection(:arch, space_a, space_b, nothing, nothing, 0)
    push!(s.connections, c)
    c
  end

add_arch(l::Layout, space_a::Space, space_b::Space) =
  add_arch(_storey_of(l, space_a), space_a, space_b)

"Attach a validation `Constraint` to a `Layout` (constraints live on the Layout so they can span storeys)."
add_rule(l::Layout, c::Constraint) =
  let _ = push!(l.rules, c)
    c
  end

# Find the storey that contains a given Space. Used by the Layout-
# flavoured `add_door!` / `add_window!` to route to the right storey.
# Uses the same lazy index as `find_space`, so both lookups share the
# O(N) setup cost and amortise to O(1) afterwards.
function _storey_of(l::Layout, sp::Space)
  entry = get(_ensure_index(l), sp.id, nothing)
  isnothing(entry) &&
    error("Space '$(sp.name)' not found in any storey of the Layout")
  entry[1]
end

#=== Geometry Helpers ===#

#=
Boundary of a polar sector — the thick slice of an annulus between
radii `r_inner`..`r_outer` and angles `theta_start`..`theta_end`.
Used anywhere a `Space` lives on a curved floor plate: arc-shaped
buildings, radial room layouts, courtyard-around-a-centre plans.

Default behaviour (`n_arc = 0`) is **arc-native**: the returned
path is a `ClosedPathSequence` of
`[inner_radial, outer_arc, outer_radial, inner_arc_reversed]`, so
the outer and inner boundaries stay true circular arcs. Downstream,
BIM backends that natively draw curved walls (AutoCAD, Revit) emit
one curved wall per arc, and `classify_all_edges` matches arc
segments against arc segments (via `cocircular_overlap`) to find
shared boundaries without discretisation.

Passing `n_arc > 0` forces polygonal discretisation: each arc is
sampled into `n_arc` line segments, and the result is a
`ClosedPolygonalPath`. Useful for non-BIM backends that can't render
curves, or for comparing against the legacy output.

See also: `closed_polygonal_path`, `closed_path_sequence`,
`arc_path`, `division`, `vpol`.
=#
"Boundary of a polar sector between radii `(r_inner, r_outer)` and angles `(theta_start, theta_end)` around `center`. Returns a `ClosedPathSequence` of lines + arcs by default, or a `ClosedPolygonalPath` when `n_arc > 0`."
function polar_sector_path(center::Loc, r_inner::Real, r_outer::Real,
                           theta_start::Real, theta_end::Real;
                           n_arc::Integer=0)
  if n_arc > 0
    # Polygonal discretisation: outer arc forward, inner arc reverse.
    let angles = division(theta_start, theta_end, n_arc),
        outer = [center + vpol(r_outer, θ) for θ in angles],
        inner = [center + vpol(r_inner, θ) for θ in reverse(angles)]
      closed_polygonal_path(vcat(outer, inner))
    end
  else
    # Arc-native: four components as one closed path sequence.
    let θs = Float64(theta_start),
        θe = Float64(theta_end),
        ri = Float64(r_inner),
        ro = Float64(r_outer),
        p_i_s = center + vpol(ri, θs),   # inner, start angle
        p_o_s = center + vpol(ro, θs),   # outer, start angle
        p_o_e = center + vpol(ro, θe),   # outer, end angle
        p_i_e = center + vpol(ri, θe)    # inner, end angle
      closed_path_sequence(
        open_polygonal_path([p_i_s, p_o_s]),               # inner-radial start edge
        arc_path(center, ro, θs, θe - θs),                 # outer arc forward
        open_polygonal_path([p_o_e, p_i_e]),               # inner-radial end edge
        arc_path(center, ri, θe, -(θe - θs)))              # inner arc reversed
    end
  end
end

"Directed edge segments `(v_i, v_{i+1})` from a closed polygon's vertex list, wrapping the last edge back to the first vertex."
polygon_edges(vertices) =
  let n = length(vertices)
    [(vertices[i], vertices[mod1(i + 1, n)]) for i in 1:n]
  end

"Shoelace-formula polygon area using world-space xy coordinates."
polygon_area(vertices) =
  let n = length(vertices),
      ws = [in_world(v) for v in vertices]
    abs(sum(cx(ws[i]) * cy(ws[mod1(i + 1, n)]) -
            cx(ws[mod1(i + 1, n)]) * cy(ws[i])
            for i in 1:n)) / 2
  end

"Parametric overlap of two collinear directed edges `(a1→a2)` and `(b1→b2)`. Returns `(t_start, t_end)` on the `a` edge, or `nothing` if non-collinear or non-overlapping."
collinear_overlap(a1, a2, b1, b2, tol=collinearity_tolerance()) =
  let edge_len = distance(a1, a2)
    edge_len < tol ? nothing :
    !(collinear_points(a1, b1, a2, tol) &&
      collinear_points(a1, b2, a2, tol)) ? nothing :
    let d = unitized(a2 - a1),
        tb1 = dot(b1 - a1, d),
        tb2 = dot(b2 - a1, d),
        ov_start = max(0.0, min(tb1, tb2)),
        ov_end = min(edge_len, max(tb1, tb2))
      (ov_end - ov_start) > tol ? (ov_start, ov_end) : nothing
    end
  end

#=== Computed Properties ===#
#
# These helpers provide `PlacedSpace`-style field access derived from
# the unified `Space`'s boundary so legacy callers (AA's constraint
# library, generator pipeline) can keep their call shape.

"Floor area of a space (polygon area of its boundary)."
space_area(space::Space) = polygon_area(path_vertices(space.boundary))

# 2D bounding-box of a space's boundary. Non-allocating scan over the
# polygon vertices — used on every `sp.width` / `sp.depth` / `sp.origin_*`
# property access, so it stays tight.
function _space_bbox(sp::Space)
  vs = path_vertices(sp.boundary)
  isempty(vs) && return (0.0, 0.0, 0.0, 0.0)
  w = in_world(vs[1])
  x_min = cx(w); x_max = x_min
  y_min = cy(w); y_max = y_min
  for i in 2:length(vs)
    w = in_world(vs[i])
    x = cx(w); y = cy(w)
    x < x_min && (x_min = x)
    x > x_max && (x_max = x)
    y < y_min && (y_min = y)
    y > y_max && (y_max = y)
  end
  (x_min, y_min, x_max, y_max)
end

"Bottom-left corner `(x, y)` of a space's bounding box."
space_origin(sp::Space) = let (x0, y0, _, _) = _space_bbox(sp); (x0, y0) end

"Width (x-extent) of a space's bounding box."
space_width(sp::Space) = let (x0, _, x1, _) = _space_bbox(sp); x1 - x0 end

"Depth (y-extent) of a space's bounding box."
space_depth(sp::Space) = let (_, y0, _, y1) = _space_bbox(sp); y1 - y0 end

"Floor-to-ceiling height assigned to this space."
space_height(sp::Space) = sp.height

"Boundary as `Vector{NTuple{2, Float64}}` (world-space x,y)."
boundary_xy(sp::Space) =
  [(cx(in_world(v)), cy(in_world(v))) for v in path_vertices(sp.boundary)]

#=
Flat iteration over every placed space in a Layout, across all
storeys. Used by constraint checks that don't care which storey a
space lives on (e.g. `min_area` on a specific `kind`).
=#
"Flat iterator over every `Space` in every `Storey` of a `Layout`."
spaces(l::Layout) = (sp for s in l.storeys for sp in s.spaces)

"Lookup a space by its id across every storey of a Layout. Uses the lazy id → (storey, space) cache."
function find_space(l::Layout, id::Symbol)
  entry = get(_ensure_index(l), id, nothing)
  isnothing(entry) ? nothing : entry[2]
end

"z-elevation of a Storey (derived from its BIM level)."
storey_z(s::Storey) = Float64(s.level.height)

"1-based index of a Storey in its enclosing Layout."
function storey_index(l::Layout, s::Storey)
  for (i, st) in enumerate(l.storeys)
    st === s && return i
  end
  0
end

"Perimeter of a space (sum of edge lengths around its boundary)."
space_perimeter(space::Space) =
  let vs = path_vertices(space.boundary),
      n = length(vs)
    sum(distance(vs[i], vs[mod1(i + 1, n)]) for i in 1:n)
  end

#=== Edge Classification ===#
#
# Edge classification drives two things: adjacency detection (which
# pairs of spaces share a wall, and which walls face the exterior)
# and wall-graph construction (one WallSegment per classified edge,
# carrying curvature when the boundary was an arc).
#
# A space's boundary is decomposed into its structural components —
# straight line segments *and* circular arcs, preserved individually
# — and each component is classified against every other space's
# components. Line-line overlaps use `collinear_overlap`; arc-arc
# overlaps (same centre + radius) use `cocircular_overlap`. Mixed
# line/arc pairs are treated as non-shared: a tangent touch isn't a
# wall, it's a rendering coincidence.
#
# Classified segments are emitted as 6-tuples
# `(p1, p2, kind, space_a, space_b, arc_or_nothing)`, where
# `arc_or_nothing` is an `ArcPath` describing the sub-arc for arc
# segments and `nothing` for straight ones. The wall-graph builder
# downstream threads this through into `arc_segment!` / `segment!`
# so curved walls survive all the way to the BIM backend.

# Unified boundary-component type. `LineEdge` is a straight segment,
# `ArcEdge` is a sub-arc of a circle carrying the explicit `ArcPath`.
struct LineEdge
  p1::Loc
  p2::Loc
end

struct ArcEdge
  arc::ArcPath
  p1::Loc  # path_start(arc) — cached for convenience
  p2::Loc  # path_end(arc)
end

ArcEdge(arc::ArcPath) = ArcEdge(arc, path_start(arc), path_end(arc))

# Decompose a boundary path into its individual components (lines
# and arcs). Polygonal boundaries yield pure line edges; path
# sequences yield the mix that was originally authored.
boundary_components(path::ClosedPolygonalPath) =
  let vs = path.vertices, n = length(vs)
    [LineEdge(vs[i], vs[mod1(i + 1, n)]) for i in 1:n]
  end

boundary_components(path::ClosedPathSequence) =
  reduce(vcat, (_path_to_edges(p) for p in path.paths); init=Any[])

boundary_components(path::Path) =
  boundary_components(convert(ClosedPolygonalPath, path))

_path_to_edges(p::OpenPolygonalPath) =
  let vs = p.vertices
    [LineEdge(vs[i], vs[i + 1]) for i in 1:length(vs) - 1]
  end
_path_to_edges(p::ArcPath) = [ArcEdge(p)]
_path_to_edges(p::Path) =
  _path_to_edges(convert(OpenPolygonalPath, p))

# Pair-wise overlap of two boundary edges. Returns a list of
# `(kind, arc_or_nothing, p1, p2)` where `kind ∈ (:interior,)` for
# shared sub-segments and `arc_or_nothing` is the sub-arc's own
# `ArcPath` when both edges are arcs.
_edge_overlap(a::LineEdge, b::LineEdge, tol) =
  let ov = collinear_overlap(a.p1, a.p2, b.p1, b.p2, tol)
    if isnothing(ov)
      nothing
    else
      let d = unitized(a.p2 - a.p1)
        (a.p1 + d * ov[1], a.p1 + d * ov[2], nothing)
      end
    end
  end

_edge_overlap(a::ArcEdge, b::ArcEdge, tol) =
  let ov = cocircular_overlap(a.arc, b.arc, tol)
    isnothing(ov) ? nothing :
      (location_at(a.arc, ov[1] - a.arc.start_angle),
       location_at(a.arc, ov[2] - a.arc.start_angle),
       arc_path(a.arc.center, a.arc.radius, ov[1], ov[2] - ov[1]))
  end

# Mixed line/arc pairs don't share a wall (at most a tangent point).
_edge_overlap(::LineEdge, ::ArcEdge, _) = nothing
_edge_overlap(::ArcEdge, ::LineEdge, _) = nothing

# Classify one boundary edge against every edge from other spaces.
# Splits the edge into sub-segments wherever another space's edge
# overlaps it, marking overlapping sub-segments as `:interior` and
# the gaps as `:exterior`.
#
# `other_edges` is a list of `(edge, other_space)` tuples.
function classify_edge(edge, space, other_edges, tol)
  overlaps = []
  for (other_edge, other_space) in other_edges
    r = _edge_overlap(edge, other_edge, tol)
    isnothing(r) && continue
    push!(overlaps, (r..., other_space))
  end
  _emit_edge_segments(edge, overlaps, space, tol)
end

# Emit the ordered list of classified sub-segments from one edge,
# interleaving interior overlaps with the exterior stretches between
# them. Separate implementations per edge kind so the ordering uses
# the right measure (chord distance for lines, angle for arcs).
function _emit_edge_segments(edge::LineEdge, overlaps, space, tol)
  a1, a2 = edge.p1, edge.p2
  edge_len = distance(a1, a2)
  edge_len < tol && return []
  d = unitized(a2 - a1)
  # Sort overlaps by parametric start along (a1 → a2).
  scored = map(overlaps) do ov
    s_t = dot(ov[1] - a1, d)
    e_t = dot(ov[2] - a1, d)
    lo  = min(s_t, e_t)
    hi  = max(s_t, e_t)
    (lo, hi, ov[4])
  end
  sort!(scored, by = first)
  segs = []
  cursor = 0.0
  for (t_start, t_end, other_space) in scored
    if t_start > cursor + tol
      push!(segs, (a1 + d * cursor, a1 + d * t_start, :exterior, space, nothing, nothing))
    end
    push!(segs, (a1 + d * t_start, a1 + d * t_end, :interior, space, other_space, nothing))
    cursor = max(cursor, t_end)
  end
  if cursor < edge_len - tol
    push!(segs, (a1 + d * cursor, a2, :exterior, space, nothing, nothing))
  end
  isempty(segs) ? [(a1, a2, :exterior, space, nothing, nothing)] : segs
end

function _emit_edge_segments(edge::ArcEdge, overlaps, space, tol)
  arc = edge.arc
  θ_s = arc.start_angle
  θ_e = arc.start_angle + arc.amplitude
  # Parameterise along the arc by angle, oriented `start → end`.
  forward = arc.amplitude >= 0
  lo = min(θ_s, θ_e); hi = max(θ_s, θ_e)
  scored = [(min(ov[3].start_angle, ov[3].start_angle + ov[3].amplitude),
             max(ov[3].start_angle, ov[3].start_angle + ov[3].amplitude),
             ov[4])
            for ov in overlaps
            if !isnothing(ov[3])]
  sort!(scored, by = first)
  segs = []
  cursor = lo
  for (t_start, t_end, other_space) in scored
    if t_start > cursor + tol
      segs_arc = _arc_subsegment(arc, cursor, t_start, forward)
      push!(segs, (segs_arc[1], segs_arc[2], :exterior, space, nothing, segs_arc[3]))
    end
    sub = _arc_subsegment(arc, t_start, t_end, forward)
    push!(segs, (sub[1], sub[2], :interior, space, other_space, sub[3]))
    cursor = max(cursor, t_end)
  end
  if cursor < hi - tol
    sub = _arc_subsegment(arc, cursor, hi, forward)
    push!(segs, (sub[1], sub[2], :exterior, space, nothing, sub[3]))
  end
  isempty(segs) ?
    [(edge.p1, edge.p2, :exterior, space, nothing, arc)] :
    segs
end

# Construct an `ArcPath` sub-arc from `arc` between angles θ_lo and
# θ_hi (with θ_lo < θ_hi), respecting the parent arc's sweep
# direction, and return (p1, p2, sub_arc).
function _arc_subsegment(arc::ArcPath, θ_lo, θ_hi, forward)
  sa, amp = forward ? (θ_lo, θ_hi - θ_lo) : (θ_hi, -(θ_hi - θ_lo))
  sub = arc_path(arc.center, arc.radius, sa, amp)
  (path_start(sub), path_end(sub), sub)
end

# Classify every edge of every space in a storey. Interior (shared)
# sub-segments are deduplicated — one record per shared boundary —
# and the resulting list carries arc information when the boundary
# came from an arc component.
function classify_all_edges(storey, tol)
  space_edges = Dict(s => boundary_components(s.boundary) for s in storey.spaces)
  segments = []
  for space in storey.spaces
    other_edges = [(e, s)
                   for s in storey.spaces if s !== space
                   for e in space_edges[s]]
    for edge in space_edges[space]
      for seg in classify_edge(edge, space, other_edges, tol)
        _, _, kind, sp_a, sp_b, _ = seg
        if kind == :interior
          objectid(sp_a) < objectid(sp_b) && push!(segments, seg)
        else
          push!(segments, seg)
        end
      end
    end
  end
  segments
end

#=
Cocircular overlap: the angular range where two arcs on the same
circle (same centre and radius, within `tol`) share a common
sweep. Returns `(θ_start, θ_end)` in absolute angles (`θ_end >
θ_start`) of the overlap, or `nothing` if the arcs don't share a
circle or don't overlap in angle.

The returned angles are in "canonical" CCW order regardless of the
input arcs' sweep direction — the caller can build a forward sub-arc
via `arc_path(center, radius, θ_start, θ_end - θ_start)`.
=#
"Angular overlap of two co-circular `ArcPath`s, or `nothing`."
function cocircular_overlap(a::ArcPath, b::ArcPath, tol=coincidence_tolerance())
  distance(a.center, b.center) > tol && return nothing
  abs(a.radius - b.radius) > tol && return nothing
  a_lo, a_hi = minmax(a.start_angle, a.start_angle + a.amplitude)
  b_lo, b_hi = minmax(b.start_angle, b.start_angle + b.amplitude)
  # Align b's range into the same 2π-window as a (shift by integer
  # multiples of 2π). This lets us compare directly even when the
  # two arcs cross the 0/2π seam differently.
  shift = round((((a_lo + a_hi) - (b_lo + b_hi)) / 2) / (2π)) * 2π
  b_lo += shift; b_hi += shift
  lo = max(a_lo, b_lo); hi = min(a_hi, b_hi)
  (hi - lo) > tol ? (lo, hi) : nothing
end

#=== Query Helpers ===#

"Shared boundary segments between two spaces as `(Loc, Loc)` pairs in world coordinates. Empty when the spaces don't touch."
shared_boundary(space_a::Space, space_b::Space, tol=collinearity_tolerance()) =
  let edges_a = polygon_edges(path_vertices(space_a.boundary)),
      edges_b = polygon_edges(path_vertices(space_b.boundary)),
      shared = Tuple{Loc, Loc}[]
    for (a1, a2) in edges_a, (b1, b2) in edges_b
      let ov = collinear_overlap(a1, a2, b1, b2, tol)
        if !isnothing(ov)
          let d = unitized(a2 - a1)
            push!(shared, (a1 + d * ov[1], a1 + d * ov[2]))
          end
        end
      end
    end
    shared
  end

"Exterior-facing edges of a space within a storey as `(Loc, Loc)` pairs. Used for facade computations."
exterior_edges(s::Storey, space::Space, tol=collinearity_tolerance()) =
  let edges = polygon_edges(path_vertices(space.boundary)),
      other_edges = [(b1, b2, sp)
                     for sp in s.spaces if sp !== space
                     for (b1, b2) in polygon_edges(path_vertices(sp.boundary))]
    [(p1, p2)
     for (a1, a2) in edges
     for (p1, p2, kind, _, _) in classify_edge(a1, a2, space, other_edges, tol)
     if kind == :exterior]
  end

"Spaces in a storey (or layout) that share a boundary with the given space."
neighbors(s::Storey, space::Space) =
  [sp for sp in s.spaces
   if sp !== space && !isempty(shared_boundary(space, sp))]

# Convenience: resolve to the space's storey and query there.
exterior_edges(l::Layout, space::Space; kwargs...) =
  exterior_edges(_storey_of(l, space), space; kwargs...)
neighbors(l::Layout, space::Space) = neighbors(_storey_of(l, space), space)

#=== Builder ===#

"""
    build(storey_or_layout)

Compile a `Storey` (or every storey of a `Layout`) down to BIM
elements via the wall-graph chain resolver: shared edges become walls,
interior connections become doors/windows/arches, boundaries become
`SpaceBoundary` records. Returns a `BuildResult` (single storey) or
`Vector{BuildResult}` (layout).
"""
function build(s::Storey)
  tol = collinearity_tolerance()
  all_segments = classify_all_edges(s, tol)
  arch_pairs = Set(
    minmax(objectid(c.space_a), objectid(c.space_b))
    for c in s.connections
    if c.kind == :arch && c.space_b isa Space)
  segments = filter(all_segments) do seg
    _, _, kind, sp_a, sp_b, _ = seg
    !(kind == :interior && !isnothing(sp_b) &&
      minmax(objectid(sp_a), objectid(sp_b)) in arch_pairs)
  end
  wg = wall_graph(level=s.level, height=s.height)
  edge_to_seg = Int[]
  # Each classified segment either carries an `ArcPath` (arc-based
  # boundary, e.g. from a polar_sector_path) or `nothing` (straight
  # boundary). We create the corresponding wall-graph segment so the
  # resolver can emit a curved or straight Wall later.
  for (p1, p2, _, _, _, arc) in segments
    let j1 = find_or_create_junction!(wg, p1, tol),
        j2 = find_or_create_junction!(wg, p2, tol)
      if j1 == j2
        push!(edge_to_seg, 0)
      else
        push!(edge_to_seg, segment!(wg, j1, j2; family=s.wall_family, arc=arc))
      end
    end
  end
  chains = resolve(wg)
  walls = []
  seg_to_wall = Dict{Int, Int}()
  seg_offset = Dict{Int, Real}()
  seg_forward = Dict{Int, Bool}()
  for chain in chains
    let w = wall(chain.path,
                 bottom_level=wg.bottom_level, top_level=wg.top_level,
                 family=chain.family, offset=chain.offset,
                 # Face polylines carry the junction-face corners
                 # computed by the chain resolver. Passing them to
                 # the wall proxy lets `b_wall` use clean per-face
                 # polygons at every junction instead of
                 # `offset(path, ±t)`, closing the gap at non-T
                 # 3-way junctions that the old pipeline missed.
                 left_face_path=chain.left_face_path,
                 right_face_path=chain.right_face_path),
        wall_idx = length(walls) + 1,
        junctions = chain_junctions(wg, chain.source_segments),
        cumulative = 0.0
      push!(walls, w)
      for (k, sidx) in enumerate(chain.source_segments)
        seg_to_wall[sidx] = wall_idx
        seg_obj = wg.segments[sidx]
        fwd = seg_obj.junction_a == junctions[k]
        seg_forward[sidx] = fwd
        seg_offset[sidx] = cumulative
        cumulative += segment_length(wg, sidx)
      end
    end
  end
  # Junction caps: at every valence-≥3 junction, the walls' top
  # faces end at their own cap lines and leave an N-gonal gap
  # between them. Render a flat surface polygon at the ceiling
  # height to close it (and the same at floor height so that
  # top-down views without a floor slab don't show through).
  #
  # Material: use the first incident wall's side material — the
  # same material `b_wall_no_openings` uses for the wall's top
  # strips, so the cap visually continues the wall tops rather
  # than appearing as a different surface. When the incident
  # walls have different families/materials (unusual but legal),
  # this picks one consistently; a proper per-wedge tessellation
  # would assign each wedge to its two flanking walls' shared
  # material, but that's a refinement for a future pass.
  let corners = KhepriBase.all_junction_face_corners(wg),
      top_z = wg.top_level.height * KhepriBase.wall_z_fighting_factor,
      bot_z = wg.bottom_level.height
    for j_idx in 1:length(wg.junctions)
      let cap = KhepriBase.junction_cap_polygon(wg, j_idx, corners[j_idx]),
          j = wg.junctions[j_idx]
        isempty(cap) && continue
        # Pick any incident wall's family for the cap's material;
        # they all share the junction so any of them is a visually
        # continuous choice.
        cap_mat = wg.segments[first(j.segments)].family.side_material
        # Push up to the ceiling (z-fighting-scaled to match the
        # wall top) — same as `b_wall_no_openings` uses for wall
        # top strips, so the cap sits exactly on the walls' top
        # edges rather than just above them.
        surface_polygon([v + vz(top_z) for v in cap]; material=cap_mat)
        surface_polygon([v + vz(bot_z) for v in reverse(cap)]; material=cap_mat)
      end
    end
  end

  boundaries = SpaceBoundary[]
  for (i, (p1, p2, kind, sp_a, sp_b, _arc)) in enumerate(segments)
    edge_to_seg[i] == 0 && continue
    let w = walls[seg_to_wall[edge_to_seg[i]]]
      push!(boundaries, SpaceBoundary(sp_a, w, :physical,
            kind == :interior ? :interior : :exterior, sp_b, p1, p2))
      if kind == :interior && !isnothing(sp_b)
        push!(boundaries, SpaceBoundary(sp_b, w, :physical, :interior, sp_a, p1, p2))
      end
    end
  end
  for conn in s.connections
    if conn.kind == :arch && conn.space_b isa Space
      for (p1, p2) in shared_boundary(conn.space_a, conn.space_b)
        push!(boundaries, SpaceBoundary(conn.space_a, nothing, :virtual, :interior, conn.space_b, p1, p2))
        push!(boundaries, SpaceBoundary(conn.space_b, nothing, :virtual, :interior, conn.space_a, p1, p2))
      end
    end
  end
  doors = []
  windows = []
  for conn in s.connections
    conn.kind == :arch && continue
    if conn.space_b isa Space
      place_interior_connection!(conn, segments, edge_to_seg, walls,
                                 seg_to_wall, seg_offset, seg_forward, wg,
                                 doors, windows, boundaries)
    elseif !isnothing(conn.loc)
      place_exterior_connection!(conn, segments, edge_to_seg, walls,
                                 seg_to_wall, seg_offset, seg_forward, wg,
                                 doors, windows, boundaries)
    else
      error("Exterior connections require a loc parameter")
    end
  end
  slabs = s.generate_slabs ?
    [slab(sp.boundary, level=s.level, family=s.slab_family)
     for sp in s.spaces] : []
  BuildResult(s, walls, doors, windows, slabs, boundaries)
end

#=
Build a whole `Layout`. Each storey is compiled independently; the
per-storey `BuildResult`s are wrapped in a `LayoutBuildResult` so
that a caller can still destructure the whole-building elements
with the same 4-tuple shape as a single storey:

    walls, doors, windows, slabs = build(layout)

Per-storey access remains available via `result.storey_results` or
`result[i]`.
=#
build(l::Layout) = LayoutBuildResult(l, [build(s) for s in l.storeys])

# Compute the position along a merged wall path for an opening on a
# given edge. `local_x` is the distance from edge start (p1) to the
# opening's left edge; `opening_width` is needed when the segment is
# reversed in the chain.
function edge_to_wall_x(local_x, opening_width, edge_idx, edge_to_seg, seg_offset, seg_forward, wg)
  let gs = edge_to_seg[edge_idx],
      offset = seg_offset[gs],
      fwd = seg_forward[gs]
    fwd ? offset + local_x :
      let seg_len = segment_length(wg, gs)
        offset + seg_len - local_x - opening_width
      end
  end
end

#=
Return the chain-segment with the longest straight span that is
wider than `min_width` — the preferred host for an opening so its
frame stays flat. Returns `nothing` if no single segment fits.

`segs` is the chain-segment tuple list used by
`place_interior_connection!` / `place_exterior_connection!`:
`(edge_idx, t_start, t_end, p1, p2, forward?)`.
=#
function argmax_or_nothing(segs, min_width)
  best = nothing
  best_len = 0.0
  for s in segs
    len = s[3] - s[2]
    if len >= min_width && len > best_len
      best_len = len
      best = s
    end
  end
  best
end

#=
Place an interior opening (door / window) hosted on the wall that
carries the shared boundary between `conn.space_a` and `conn.space_b`.

Chain-aware: the shared boundary is often discretised into several
segments (arcs become polylines; long straight shared walls break at
transit points where another wall meets them), and the wall-graph
resolver merges those segments into one `Wall` whose path is the
full chain. The opening must be placed against the *chain's* length,
not any single segment's length, or the door can end up off the end
of the host wall.

The placement algorithm:
  1. Collect every classified segment whose `(sp_a, sp_b)` pair
     matches the connection.
  2. Project each segment onto its resolved wall's chain coordinate
     via `seg_offset` + per-segment length, giving an interval
     `[offset, offset + length]`.
  3. Merge the intervals into one or more `[t_start, t_end]` spans
     per host wall. The longest span is the "shared region" — the
     continuous stretch of wall that adjoins both spaces.
  4. If `conn.loc` is given, map it onto that span (clamped); else
     centre the opening at its midpoint.
  5. Refuse placement if the span is narrower than the opening.

`op_start` / `op_end` records are computed from a representative
segment (the one containing the opening's midpoint) so the IFC
`SpaceBoundary` metadata has world-space endpoints.
=#
function place_interior_connection!(conn, segments, edge_to_seg, walls,
                                    seg_to_wall, seg_offset, seg_forward, wg,
                                    doors, windows, boundaries)
  target = minmax(objectid(conn.space_a), objectid(conn.space_b))
  # 1. Collect matching segments, grouped by host wall.
  by_wall = Dict{Int, Vector{Tuple{Int, Real, Real, Any, Any, Bool}}}()
  for (i, seg) in enumerate(segments)
    edge_to_seg[i] == 0 && continue
    _, _, kind, sp_a, sp_b, _ = seg
    kind == :interior || continue
    isnothing(sp_b) && continue
    minmax(objectid(sp_a), objectid(sp_b)) == target || continue
    gs = edge_to_seg[i]
    wall_idx = seg_to_wall[gs]
    off = seg_offset[gs]
    ln  = segment_length(wg, gs)
    fwd = seg_forward[gs]
    t0, t1 = fwd ? (off, off + ln) : (off, off + ln)  # chain-space always forward
    push!(get!(by_wall, wall_idx, []), (i, t0, t1, seg[1], seg[2], fwd))
  end
  isempty(by_wall) &&
    error("No shared wall found between '$(conn.space_a.name)' and '$(conn.space_b.name)'")
  # 2. Pick the host wall with the longest contiguous shared span.
  best_wall = 0
  best_start = 0.0
  best_end = 0.0
  best_segs = nothing
  for (wall_idx, segs) in by_wall
    sorted = sort(segs, by = t -> t[2])
    # Merge overlapping/touching intervals into one span.
    span_lo = sorted[1][2]; span_hi = sorted[1][3]
    for k in 2:length(sorted)
      lo, hi = sorted[k][2], sorted[k][3]
      if lo <= span_hi + 1e-6
        span_hi = max(span_hi, hi)
      else
        # non-contiguous; treat each run separately — keep the largest so far.
        if (span_hi - span_lo) > (best_end - best_start)
          best_wall = wall_idx; best_start = span_lo; best_end = span_hi
          best_segs = sorted
        end
        span_lo = lo; span_hi = hi
      end
    end
    if (span_hi - span_lo) > (best_end - best_start)
      best_wall = wall_idx; best_start = span_lo; best_end = span_hi
      best_segs = sorted
    end
  end
  # 3. Place the opening on the chosen wall.
  w = walls[best_wall]
  shared_len = best_end - best_start
  opening_width = conn.family.width
  shared_len + 1e-6 < opening_width &&
    error("Shared boundary between '$(conn.space_a.name)' and '$(conn.space_b.name)' " *
          "is $(round(shared_len; digits=3))m, too short for the $(opening_width)m opening")
  wall_x = if isnothing(conn.loc)
    # Prefer a single straight sub-segment: the door frame stays
    # rectangular instead of bending across a polyline vertex. Pick
    # the widest sub-segment on the chain that can host the full
    # opening; fall back to the chain midpoint only if no single
    # segment is wide enough.
    straight = argmax_or_nothing(best_segs, opening_width)
    if !isnothing(straight)
      (s_i, s_t0, s_t1, _, _, _) = straight
      s_t0 + ((s_t1 - s_t0) - opening_width) / 2
    else
      best_start + (shared_len - opening_width) / 2
    end
  else
    # Project `conn.loc` onto the host chain via the nearest matching segment.
    nearest = argmin([distance(conn.loc, (t[4] + t[5]) / 2) for t in best_segs])
    t = best_segs[nearest]
    (p1, p2) = (t[4], t[5])
    d = unitized(p2 - p1)
    local_x = clamp(dot(conn.loc - p1, d) - opening_width / 2,
                    0.0, distance(p1, p2) - opening_width)
    t[6] ? t[2] + local_x : t[3] - local_x - opening_width
  end
  # 4. Approximate op_start/op_end from the segment containing the midpoint.
  mid_t = wall_x + opening_width / 2
  host = let h = best_segs[1]
    for t in best_segs
      if t[2] - 1e-6 <= mid_t <= t[3] + 1e-6
        h = t; break
      end
    end
    h
  end
  (p1, p2) = (host[4], host[5])
  d = unitized(p2 - p1)
  local_x = host[6] ? mid_t - host[2] - opening_width / 2 :
                       host[3] - mid_t - opening_width / 2
  op_start = p1 + d * local_x
  op_end   = p1 + d * (local_x + opening_width)
  wall_loc = xy(wall_x, conn.sill)
  if conn.kind == :door
    add_door(w, wall_loc, conn.family)
    push!(doors, w.doors[end])
    let el = w.doors[end]
      push!(boundaries, SpaceBoundary(conn.space_a, el, :virtual, :interior, conn.space_b, op_start, op_end))
      push!(boundaries, SpaceBoundary(conn.space_b, el, :virtual, :interior, conn.space_a, op_start, op_end))
    end
  else
    add_window(w, wall_loc, conn.family)
    push!(windows, w.windows[end])
    let el = w.windows[end]
      push!(boundaries, SpaceBoundary(conn.space_a, el, :virtual, :interior, conn.space_b, op_start, op_end))
      push!(boundaries, SpaceBoundary(conn.space_b, el, :virtual, :interior, conn.space_a, op_start, op_end))
    end
  end
end

#=
Place an exterior opening on whichever resolved wall hosts
`conn.space_a`'s outer face nearest to `conn.loc`.

Chain-aware (parallel to `place_interior_connection!`): exterior
facades can be discretised into many segments (arcs, stepped
facades); the resolver merges them into a single `Wall`, and the
opening must be positioned against the chain's length, not any one
segment's. The nearest-segment to `conn.loc` picks the host wall,
and the opening's `wall_x` comes from the chain offset + the clamp
of the projected distance onto the chain.
=#
function place_exterior_connection!(conn, segments, edge_to_seg, walls,
                                    seg_to_wall, seg_offset, seg_forward, wg,
                                    doors, windows, boundaries)
  # Nearest exterior segment of `conn.space_a` to `conn.loc`.
  best_idx = 0
  best_dist = Inf
  for (i, seg) in enumerate(segments)
    edge_to_seg[i] == 0 && continue
    _, _, kind, sp_a, _, _ = seg
    kind == :exterior || continue
    sp_a !== conn.space_a && continue
    p1, p2 = seg[1], seg[2]
    d = unitized(p2 - p1)
    wall_len = distance(p1, p2)
    t = clamp(dot(conn.loc - p1, d), 0.0, wall_len)
    dist = distance(conn.loc, p1 + d * t)
    if dist < best_dist
      best_dist = dist
      best_idx = i
    end
  end
  best_idx == 0 &&
    error("No exterior wall found for '$(conn.space_a.name)' near $(conn.loc)")
  w = walls[seg_to_wall[edge_to_seg[best_idx]]]
  host_wall = seg_to_wall[edge_to_seg[best_idx]]
  opening_width = conn.family.width
  # Collect every exterior segment of this space that lives on the
  # SAME resolved wall, so we can centre the opening on the chain.
  chain_segs = []
  for (i, seg) in enumerate(segments)
    edge_to_seg[i] == 0 && continue
    _, _, kind, sp_a, _, _ = seg
    kind == :exterior || continue
    sp_a !== conn.space_a && continue
    gs = edge_to_seg[i]
    seg_to_wall[gs] == host_wall || continue
    off = seg_offset[gs]
    ln  = segment_length(wg, gs)
    fwd = seg_forward[gs]
    push!(chain_segs, (i, off, off + ln, seg[1], seg[2], fwd))
  end
  sort!(chain_segs, by = t -> t[2])
  chain_start = chain_segs[1][2]
  chain_end   = chain_segs[end][3]
  chain_len   = chain_end - chain_start
  chain_len + 1e-6 < opening_width &&
    error("Exterior boundary of '$(conn.space_a.name)' on this wall is " *
          "$(round(chain_len; digits=3))m, too short for the $(opening_width)m opening")
  # Project conn.loc onto the nearest matching segment to find a
  # target chain offset.
  (p1n, p2n) = (segments[best_idx][1], segments[best_idx][2])
  d_host = unitized(p2n - p1n)
  t_near = clamp(dot(conn.loc - p1n, d_host), 0.0, distance(p1n, p2n))
  nearest_seg = first(s for s in chain_segs if s[1] == best_idx)
  target = nearest_seg[6] ? nearest_seg[2] + t_near :
                             nearest_seg[3] - t_near
  # Prefer a single straight sub-segment containing (or near) the
  # target, so the opening frame doesn't bend across a polyline
  # vertex. If the nearest segment is wide enough, anchor there;
  # otherwise fall back to clamping against the full chain.
  wall_x = if (nearest_seg[3] - nearest_seg[2]) >= opening_width
    ns_start = nearest_seg[2]
    ns_end   = nearest_seg[3]
    clamp(target - opening_width / 2, ns_start, ns_end - opening_width)
  else
    clamp(target - opening_width / 2,
          chain_start, chain_end - opening_width)
  end
  # Approximate op_start/op_end from the segment containing the midpoint.
  mid_t = wall_x + opening_width / 2
  host = chain_segs[1]
  for s in chain_segs
    if s[2] - 1e-6 <= mid_t <= s[3] + 1e-6
      host = s; break
    end
  end
  (hp1, hp2) = (host[4], host[5])
  dh = unitized(hp2 - hp1)
  local_x = host[6] ? mid_t - host[2] - opening_width / 2 :
                      host[3] - mid_t - opening_width / 2
  op_start = hp1 + dh * local_x
  op_end   = hp1 + dh * (local_x + opening_width)
  wall_loc = xy(wall_x, conn.sill)
  if conn.kind == :door
    add_door(w, wall_loc, conn.family)
    push!(doors, w.doors[end])
    let el = w.doors[end]
      push!(boundaries, SpaceBoundary(conn.space_a, el, :virtual, :exterior, nothing, op_start, op_end))
    end
  else
    add_window(w, wall_loc, conn.family)
    push!(windows, w.windows[end])
    let el = w.windows[end]
      push!(boundaries, SpaceBoundary(conn.space_a, el, :virtual, :exterior, nothing, op_start, op_end))
    end
  end
end

#=== BuildResult Queries ===#

"All `SpaceBoundary` records that mention a given `space` (cf. `IfcSpace.BoundedBy`)."
space_boundaries(result::BuildResult, space::Space) =
  [b for b in result.boundaries if b.space === space]

"Walls bounding a space within a `BuildResult` — deduplicated to one wall per shared boundary."
space_walls(result::BuildResult, space::Space) =
  unique([b.element for b in result.boundaries
          if b.space === space && b.kind == :physical])

"Doors hosted on walls that bound a space."
space_doors(result::BuildResult, space::Space) =
  unique([b.element for b in result.boundaries
          if b.space === space && !isnothing(b.element) && b.element isa Door])

"Windows hosted on walls that bound a space."
space_windows(result::BuildResult, space::Space) =
  unique([b.element for b in result.boundaries
          if b.space === space && !isnothing(b.element) && b.element isa Window])

"All spaces bounded by a given element (inverse of `space_walls`/`space_doors`/`space_windows`)."
bounding_spaces(result::BuildResult, element) =
  unique([b.space for b in result.boundaries if b.element === element])

"Spaces that share a wall or opening with the given space within a `BuildResult`."
adjacent_spaces(result::BuildResult, space::Space) =
  unique(filter(!isnothing,
    [b.related_space for b in result.boundaries if b.space === space]))

#=== Validation ===#
#
# The constraint library (`min_area`, `max_area`, `has_door`,
# `has_connection`, `must_adjoin`, `must_not_adjoin`, …), the
# algebra, and the `validate(ctx, constraints)` generic live in
# `Constraints.jl`. Here we just add the one specialisation that
# lets users call `validate(layout)` after `build`: it flattens the
# per-storey `BuildResult`s into one context and runs every
# constraint in `layout.rules` against it.

#=
Wrapper returned by `build(l::Layout)`. Holds a back-pointer to the
originating `Layout` and the per-storey `BuildResult` vector.

The iteration protocol mirrors `BuildResult` — four vectors in the
order `walls, doors, windows, slabs`, flattened across every storey
— so the idiomatic 4-tuple destructure works the same whether the
caller built a single `Storey` or a multi-storey `Layout`:

    walls, doors, windows, slabs = build(storey)   # single BuildResult
    walls, doors, windows, slabs = build(layout)   # aggregated across storeys

Per-storey results remain individually addressable via `r[i]`
(equivalent to `r.storey_results[i]`). Computed properties
(`.walls`, `.doors`, `.windows`, `.slabs`, `.boundaries`) give the
same aggregated views used for validation / realization.
=#
"Aggregated `build` output for a multi-storey `Layout`; destructures as `walls, doors, windows, slabs` across all storeys."
struct LayoutBuildResult
  layout::Layout
  storey_results::Vector{BuildResult}
end

# Aggregated element lists — what `walls, doors, windows, slabs = …` sees.
_all_walls(r::LayoutBuildResult)     = reduce(vcat, (s.walls     for s in r.storey_results); init=[])
_all_doors(r::LayoutBuildResult)     = reduce(vcat, (s.doors     for s in r.storey_results); init=[])
_all_windows(r::LayoutBuildResult)   = reduce(vcat, (s.windows   for s in r.storey_results); init=[])
_all_slabs(r::LayoutBuildResult)     = reduce(vcat, (s.slabs     for s in r.storey_results); init=[])
_all_boundaries(r::LayoutBuildResult) =
  reduce(vcat, (s.boundaries for s in r.storey_results); init=SpaceBoundary[])

# Destructuring: mirror `BuildResult.iterate` so the 4-tuple unpack
# `walls, doors, windows, slabs = build(layout)` keeps working across
# the single-storey / multi-storey divide.
Base.iterate(r::LayoutBuildResult, state=1) =
  state == 1 ? (_all_walls(r),   2) :
  state == 2 ? (_all_doors(r),   3) :
  state == 3 ? (_all_windows(r), 4) :
  state == 4 ? (_all_slabs(r),   5) :
  nothing
Base.length(::LayoutBuildResult) = 4

# Per-storey access.
Base.getindex(r::LayoutBuildResult, i::Integer) = r.storey_results[i]
Base.lastindex(r::LayoutBuildResult) = length(r.storey_results)

# Property forwarding: `.walls`, `.doors`, `.windows`, `.slabs`,
# `.boundaries` aggregate across storeys; real fields (`layout`,
# `storey_results`) pass through.
function Base.getproperty(r::LayoutBuildResult, s::Symbol)
  s === :walls      ? _all_walls(r) :
  s === :doors      ? _all_doors(r) :
  s === :windows    ? _all_windows(r) :
  s === :slabs      ? _all_slabs(r) :
  s === :boundaries ? _all_boundaries(r) :
  getfield(r, s)
end
Base.propertynames(::LayoutBuildResult) =
  (:layout, :storey_results, :walls, :doors, :windows, :slabs, :boundaries)

Base.show(io::IO, r::LayoutBuildResult) =
  print(io, "LayoutBuildResult($(length(r.storey_results)) storey(s), ",
            "$(length(_all_walls(r))) walls, $(length(_all_doors(r))) doors, ",
            "$(length(_all_windows(r))) windows, $(length(_all_slabs(r))) slabs)")

# Run every constraint against every storey result and concatenate.
# Constraints that care only about one storey will do the right thing
# because each result carries its own `BuildResult.storey`.
validate(results::Vector{BuildResult}, constraints::Vector{Constraint}) =
  let hard = Violation[], soft = Violation[], prefs = Violation[]
    for r in results, c in constraints, v in c.check(r)
      (v.severity == HARD ? hard :
       v.severity == SOFT ? soft : prefs) |> vs -> push!(vs, v)
    end
    ValidationResult(isempty(hard), hard, soft, prefs,
                     1000.0 * length(hard) + 10.0 * length(soft) + 1.0 * length(prefs))
  end

# `LayoutBuildResult` forwards to the per-storey vector so that
# validate works regardless of which build variant produced it.
validate(r::LayoutBuildResult, constraints::Vector{Constraint}) =
  validate(r.storey_results, constraints)

# Convenience: validate a freshly-built Layout against its own rules.
validate(l::Layout) = validate(build(l), l.rules)
