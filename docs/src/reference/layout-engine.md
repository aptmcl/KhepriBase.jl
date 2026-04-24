# Layout Engine

`layout` walks a [`SpaceDesc`](@ref) tree and assigns world
coordinates to every leaf, grouping placed spaces by z into one
[`Storey`](@ref) per level. Its output is a [`Layout`](@ref) — the
same type the imperative `floor_plan` returns — which every
downstream stage (`build`, `validate`, or front-end packages like
`AlgorithmicArchitecture.generate_elements`) consumes.

```julia
using KhepriBase

desc = room(:bed, :bedroom, 4.0, 3.0) | room(:bath, :bathroom, 2.5, 3.0)
l = layout(desc)                        # origin defaults to (0, 0, 0)
sp_bed  = find_space(l, :bed)
sp_bath = find_space(l, :bath)
space_origin(sp_bed)[1]                  # 0.0
space_origin(sp_bath)[1]                 # 4.0
adjacencies(l)                           # shared walls + exterior edges
l.storeys                                # one Storey per distinct z
```

Pass `origin_x`, `origin_y`, `origin_z` keywords to place the whole
composition at a non-origin anchor.

## Where the types and accessors live

The unified Level-1 types and their accessors — [`Space`](@ref),
[`Storey`](@ref), [`Layout`](@ref), [`spaces`](@ref),
[`find_space`](@ref), [`space_origin`](@ref), [`space_width`](@ref),
[`space_depth`](@ref), [`space_area`](@ref), [`space_perimeter`](@ref),
[`boundary_xy`](@ref), [`storey_z`](@ref), [`adjacent_spaces`](@ref),
[`space_boundaries`](@ref), [`space_walls`](@ref),
[`space_doors`](@ref), [`space_windows`](@ref) — are auto-documented
under the [*Space Layout* section of API -- Infrastructure](api_other.md#Space-Layout).

The adjacency data type ([`AdjacencyRelation`](@ref)) and its
constructors ([`adjacencies`](@ref), [`detect_adjacencies`](@ref))
live on the [Adjacencies](adjacencies.md) page.

## Compiling a design

```@docs
layout(::SpaceDesc)
rectangular_boundary
```

## See also

- [Designs (Level 2)](../concepts/designs.md) — the declarative tree
  that `layout(desc)` consumes.
- [Space Descriptions](../concepts/space-descriptions.md) — how the
  tree leaves map to placed spaces.
- [Adjacencies](adjacencies.md) — what `adjacencies(l)` returns and
  how the shared-edge detection works.
- [Constraints (reference)](constraints.md) — validators that operate
  on a `Layout` or a `BuildResult`.
