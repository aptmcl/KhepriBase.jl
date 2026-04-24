# Adjacencies

An *adjacency relation* records that two placed spaces share a
boundary edge — either with each other (interior adjacency) or with
the outside world (exterior adjacency). Adjacencies are the bridge
between the Level-1 `Layout` and the downstream passes that place
walls, doors, windows, and running constraint checks:

- [`build`](@ref) uses them to decide where party walls go vs. where
  a single wall faces the envelope.
- `build_walls` (in the WallGraph layer) turns each interior
  adjacency into one shared wall segment, and each exterior
  adjacency into an envelope segment.
- The [constraint library](constraints.md) relies on adjacency data
  to check `must_adjoin`, `all_reachable`, `max_dead_end`,
  `preferred_orientation`, `facade_ratio`, and more.

## The data shape

```@docs
AdjacencyRelation
```

Each `AdjacencyRelation` carries the two `Space` ids it relates
(`space_b === nothing` for exterior edges), the 2D shared edge in
world coordinates, and the z-elevation of the storey it lives on.
Spaces on different elevations never share an edge — the classifier
groups by z before comparing geometry.

## Computing adjacencies

Two entry points, one algorithm. Both flow through
`classify_all_edges` in `Spaces.jl` so the "which edges are shared
between which spaces?" math has a single authoritative
implementation.

```@docs
adjacencies
detect_adjacencies
```

Prefer `adjacencies(layout)` when you already hold a `Layout` — it
iterates storeys and stamps each `AdjacencyRelation` with the
correct `level_z`. Use `detect_adjacencies(spaces)` when you have a
loose collection (a `Vector{Space}`, or a `Dict{Symbol, Space}`)
and have not wrapped them into a `Layout` yet.

The interior and exterior edges of a four-room layout:

![adjacency example](../assets/reference/adjacencies-example.svg)

## Worked example

```julia
using KhepriBase

desc = (room(:living, :living_room, 5.0, 4.0) |
        room(:kitchen, :kitchen,    3.0, 4.0)) /
       (room(:bed,    :bedroom,     4.0, 3.0) |
        room(:bath,   :bathroom,    2.5, 3.0))

l    = layout(desc)
adjs = adjacencies(l)

# How many interior party-wall edges are there?
count(a -> !isnothing(a.space_b), adjs)

# Which rooms sit on the exterior skin?
exterior_rooms = unique(a.space_a for a in adjs if isnothing(a.space_b))
```

## See also

- [Layout Engine](layout-engine.md) — how `layout(desc)` produces the
  `Layout` that `adjacencies` consumes.
- [Spaces](../bim/spaces.md) — the narrative intro to the space-first
  layout model.
- [Wall Graph](../bim/wall_graph.md) — how adjacencies become physical
  walls once `build` runs.
- [Constraints (reference)](constraints.md) — library constraints
  that operate on adjacency data.
