# Spaces

The Spaces module provides a space-first approach to architectural layout. Instead of manually constructing individual walls, doors, and windows, you define **spaces** (rooms) as closed polygonal areas and **connections** (doors, windows, arches) between them. A single call to `build()` generates all BIM geometry automatically: shared walls between adjacent rooms are detected and built only once, exterior walls are placed around the perimeter, and openings are positioned on the correct wall segments.

After building, the result carries a descriptive boundary model that records which elements bound which spaces. This supports introspection queries ("which walls surround this room?", "which rooms does this door connect?") and validation rules ("every bedroom must be at least 9 m^2").

For a walkthrough with complete house designs, parameterized layouts, and custom validation, see the [Spaces Tutorial](../tutorials/spaces_tutorial.md).

## Types

### Space

A named bounded area on a floor plan. Each space is defined by a closed polygonal path and classified by a `kind` symbol.

```julia
struct Space
  name::String
  kind::Symbol        # :space, :room, :wc, :kitchen, :corridor, :office, :bedroom, :parking, ...
  boundary::ClosedPath
end
```

The `kind` field classifies the space's function, analogous to IFC's `IfcSpaceTypeEnum`. It is used by validation rules to apply kind-specific constraints (e.g., minimum area for `:bedroom` spaces).

Spaces are not constructed directly. Use `add_space` to create and register them on a `FloorPlan`.

### SpaceConnection

A connection between two spaces, or between a space and the exterior. Connections declare where doors, windows, or arches should be placed.

```julia
struct SpaceConnection
  kind::Symbol                    # :door, :window, :arch
  space_a::Space
  space_b::Union{Space, Symbol}   # Space or :exterior
  family::Union{Family, Nothing}
  loc::Union{Loc, Nothing}        # World-space point on the boundary edge
end
```

- `space_b` is either another `Space` (interior connection) or the symbol `:exterior`.
- `family` specifies the door or window family (dimensions, materials). It is `nothing` for arches.
- `loc` is a world-space point indicating where on the boundary edge the opening should be placed. When `nothing`, the opening is centered on the shared wall segment. Exterior connections require `loc` to identify which exterior wall receives the opening.

Connections are not constructed directly. Use `add_door`, `add_window`, or `add_arch`.

### SpaceBoundary

A relationship between a space and a bounding element, inspired by IFC's `IfcRelSpaceBoundary`. These records are produced by `build()` and enable post-build introspection.

```julia
struct SpaceBoundary
  space::Space
  element               # Wall, Door, Window, or nothing (for arches)
  kind::Symbol          # :physical (wall) or :virtual (opening/arch)
  side::Symbol          # :interior or :exterior
  related_space::Union{Space, Nothing}  # space on the other side (2nd-level boundary)
  p1::Loc               # boundary segment start
  p2::Loc               # boundary segment end
end
```

- `kind` is `:physical` for walls and `:virtual` for openings (doors, windows) and arches.
- `side` is `:interior` when the boundary is shared with another space, or `:exterior` when it faces outside.
- `related_space` records the space on the other side of the boundary (the IFC "2nd level" boundary concept). It is `nothing` for exterior boundaries.
- `element` is `nothing` for arch boundaries, where no physical element is generated.

### SpaceRule

A validation rule that can be checked against spaces after building.

```julia
struct SpaceRule
  name::String
  check       # (Space, BuildResult) -> Union{Nothing, String}
end
```

The `check` function receives a `Space` and a `BuildResult`. It returns `nothing` if the rule passes, or a message string describing the violation.

### FloorPlan

A mutable container holding all the spaces, connections, and validation rules at a given level.

```julia
mutable struct FloorPlan
  spaces::Vector{Space}
  connections::Vector{SpaceConnection}
  level::Level
  height::Real
  wall_family::WallFamily
  slab_family::SlabFamily
  generate_slabs::Bool
  rules::Vector{SpaceRule}
end
```

Create a `FloorPlan` with the `floor_plan` constructor. Spaces, connections, and rules are added incrementally with `add_space`, `add_door`, `add_window`, `add_arch`, and `add_rule`.

### BuildResult

The result of `build()`: BIM element lists plus the descriptive boundary model.

```julia
struct BuildResult
  plan::FloorPlan
  walls::Vector
  doors::Vector
  windows::Vector
  slabs::Vector
  boundaries::Vector{SpaceBoundary}
end
```

`BuildResult` supports tuple destructuring so you can extract just the element lists:

```julia
walls, doors, windows, slabs = build(plan)
```

It also has a custom `show` method that summarizes the counts:

```julia
result = build(plan)
println(result)
# BuildResult(4 spaces, 12 walls, 3 doors, 5 windows, 4 slabs, 24 boundaries)
```

## Constructor Functions

### floor_plan

Create an empty floor plan with default or custom parameters.

```julia
floor_plan(; level=default_level(),
             height=default_level_to_level_height(),
             wall_family=default_wall_family(),
             slab_family=default_slab_family(),
             generate_slabs=true,
             rules=SpaceRule[])
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `level` | `default_level()` | The building level for this floor plan |
| `height` | `default_level_to_level_height()` | Floor-to-floor height |
| `wall_family` | `default_wall_family()` | Wall family for generated walls |
| `slab_family` | `default_slab_family()` | Slab family for generated floor slabs |
| `generate_slabs` | `true` | Whether to generate floor slabs for each space |
| `rules` | `SpaceRule[]` | Validation rules to apply after building |

```julia
plan = floor_plan(
  height=2.8,
  wall_family=wall_family(thickness=0.2),
  slab_family=slab_family(thickness=0.25),
  rules=[min_area_rule(:bedroom, 9.0), has_door_rule()])
```

### add_space

Register a named space on a floor plan. Returns the new `Space`.

```julia
add_space(plan::FloorPlan, name, boundary; kind=:space)
```

The `boundary` must be a `ClosedPath`. The `kind` keyword classifies the space for validation and semantic purposes.

```julia
room = add_space(plan, "Living Room",
  closed_polygonal_path([xy(0,0), xy(6,0), xy(6,5), xy(0,5)]),
  kind=:room)

corridor = add_space(plan, "Corridor",
  rectangular_path(xy(0, 5), 10, 1.2),
  kind=:corridor)
```

### add_door

Declare a door between two spaces, or between a space and the exterior. Returns the new `SpaceConnection`.

```julia
add_door(plan::FloorPlan, space_a::Space, space_b::Union{Space, Symbol};
         family=default_door_family(), loc=nothing)
```

For interior doors (`space_b` is a `Space`), the door is placed on the shared wall segment. If `loc` is `nothing`, the door is centered. For exterior doors (`space_b` is `:exterior`), `loc` is required to identify the target wall.

```julia
# Interior door centered on the shared wall
add_door(plan, bedroom, corridor)

# Exterior door at a specific location
add_door(plan, living, :exterior, loc=xy(3, 0))
```

### add_window

Declare a window between two spaces, or between a space and the exterior. Returns the new `SpaceConnection`.

```julia
add_window(plan::FloorPlan, space_a::Space, space_b::Union{Space, Symbol};
           family=default_window_family(), loc=nothing)
```

Follows the same placement logic as `add_door`. Exterior windows always require `loc`.

```julia
# Large window on the south facade
add_window(plan, living, :exterior,
  loc=xy(2, 0),
  family=window_family(width=1.4, height=1.5))

# Window between two interior spaces
add_window(plan, office_a, office_b,
  family=window_family(width=1.0, height=0.6))
```

### add_arch

Declare an open passage (arch) between two spaces. No wall is generated on the shared boundary. Returns the new `SpaceConnection`.

```julia
add_arch(plan::FloorPlan, space_a::Space, space_b::Space)
```

Arches are always between two interior spaces (not with `:exterior`). They have no family or location parameters -- the entire shared edge becomes an open passage.

```julia
living = add_space(plan, "Living", rectangular_path(u0(), 6, 5))
dining = add_space(plan, "Dining", rectangular_path(xy(6, 0), 4, 5))
add_arch(plan, living, dining)
```

### add_rule

Attach a validation rule to a floor plan. Returns the `SpaceRule`.

```julia
add_rule(plan::FloorPlan, rule::SpaceRule)
```

Rules can be added at construction time via the `rules` keyword of `floor_plan`, or incrementally with `add_rule`.

```julia
add_rule(plan, min_area_rule(:wc, 3.0))
add_rule(plan, has_connection_rule())
```

## Building

### build

Generate all BIM elements from a floor plan. Returns a `BuildResult` containing the generated walls, doors, windows, slabs, and the complete boundary model.

```julia
build(plan::FloorPlan) -> BuildResult
```

The build process performs five steps:

1. **Edge classification**: every edge of every space is classified as `:interior` (shared with another space) or `:exterior`. Interior edges are deduplicated so that only one wall is generated per shared boundary.
2. **Wall graph construction**: classified edges are assembled into a [WallGraph](wall_graph.md), with junctions at every polygon vertex. This captures the topology of the wall network.
3. **Chain resolution**: the wall graph is resolved into chains -- maximal runs of same-family segments that can be merged into single multi-vertex wall paths. L-corners get proper miter joints; abutting walls at T-junctions are extended to meet the through-wall's face. This typically reduces the number of wall objects significantly (e.g., a 5-room house with 16 edge segments produces 5 merged walls).
4. **Opening placement**: doors and windows are placed on the merged wall paths. Positions are adjusted to account for chain offset and segment orientation within the merged path. Interior openings are centered on the shared wall unless `loc` overrides the position. Exterior openings use `loc` to find the nearest exterior wall.
5. **Slab generation**: if `generate_slabs` is `true`, a floor slab is generated for each space using its boundary path.

Throughout the process, `SpaceBoundary` records are accumulated to build the boundary model.

```julia
result = build(plan)

# Access elements directly
result.walls
result.doors
result.windows
result.slabs
result.boundaries

# Or use tuple destructuring
walls, doors, windows, slabs = build(plan)
```

## Query Functions

### Computed Properties (Pre-Build)

These functions operate on `Space` objects directly and do not require a `BuildResult`.

#### space_area

Compute the area of a space using the shoelace formula on its boundary vertices.

```julia
space_area(space::Space) -> Real
```

```julia
room = add_space(plan, "Room", rectangular_path(u0(), 5, 4))
space_area(room)  # => 20.0
```

#### space_perimeter

Compute the perimeter of a space by summing its boundary edge lengths.

```julia
space_perimeter(space::Space) -> Real
```

```julia
space_perimeter(room)  # => 18.0
```

### Geometric Topology (Pre-Build)

These functions analyze the geometric relationships between spaces and do not require a `BuildResult`.

#### shared_boundary

Find shared boundary segments between two spaces. Returns a list of `(p1, p2)` location pairs representing the shared directed edges.

```julia
shared_boundary(space_a::Space, space_b::Space, tol=collinearity_tolerance())
  -> Vector{Tuple{Loc, Loc}}
```

```julia
shared_boundary(living, kitchen)
# => [(xy(6,0), xy(6,5))]  -- a single 5m shared edge
```

#### exterior_edges

Find all exterior edges of a space within a plan (edges not shared with any other space).

```julia
exterior_edges(plan::FloorPlan, space::Space, tol=collinearity_tolerance())
  -> Vector{Tuple{Loc, Loc}}
```

```julia
exterior_edges(plan, bathroom)
# => [(xy(10,6.2), xy(10,10)), (xy(10,10), xy(8,10))]
```

#### neighbors

Find all spaces in the plan that share a boundary with the given space.

```julia
neighbors(plan::FloorPlan, space::Space) -> Vector{Space}
```

```julia
neighbors(plan, living)
# => [kitchen, corridor]
```

### Boundary Introspection (Post-Build)

These functions query the `BuildResult` boundary model produced by `build()`.

#### space_boundaries

All boundaries for a given space. This is the equivalent of IFC's `IfcSpace.BoundedBy` inverse relationship.

```julia
space_boundaries(result::BuildResult, space::Space) -> Vector{SpaceBoundary}
```

```julia
for b in space_boundaries(result, living)
  println("$(b.kind) $(b.side): $(typeof(b.element))")
end
# physical exterior: Wall
# physical interior: Wall
# virtual interior: Door
# virtual exterior: Window
```

#### space_walls

All unique walls bounding a space (physical boundaries only).

```julia
space_walls(result::BuildResult, space::Space) -> Vector
```

#### space_doors

All unique doors accessible from a space.

```julia
space_doors(result::BuildResult, space::Space) -> Vector
```

#### space_windows

All unique windows on a space.

```julia
space_windows(result::BuildResult, space::Space) -> Vector
```

#### bounding_spaces

All spaces bounded by a given element (wall, door, or window). Returns the spaces on both sides of a shared element.

```julia
bounding_spaces(result::BuildResult, element) -> Vector{Space}
```

```julia
some_wall = result.walls[1]
bounding_spaces(result, some_wall)
# => [living, kitchen]  -- for a shared wall
```

#### adjacent_spaces

All spaces adjacent to a given space through any boundary (interior walls, doors, windows, or arches).

```julia
adjacent_spaces(result::BuildResult, space::Space) -> Vector{Space}
```

```julia
adjacent_spaces(result, corridor)
# => [living, kitchen, bedroom1, bedroom2, bathroom]
```

## Validation System

The validation system allows you to define constraints on spaces and check them against the built model. Each rule is a `SpaceRule` with a name and a check function.

### Running Validation

#### validate (all registered rules)

Check all rules registered on the plan. Returns a list of violation message strings. An empty list means all rules pass.

```julia
validate(result::BuildResult) -> Vector{String}
```

#### validate (specific rules)

Check a specific list of rules without requiring them to be registered on the plan.

```julia
validate(result::BuildResult, rules) -> Vector{String}
```

```julia
result = build(plan)

# Validate all registered rules
violations = validate(result)

# Validate specific rules only
area_violations = validate(result, [min_area_rule(:wc, 3.0)])
```

Both forms iterate over every space in the plan and call each rule's `check` function. When the function returns a non-`nothing` string, it is collected as a violation.

### Predefined Rules

#### min_area_rule

Minimum area constraint. Has two variants: one for all spaces, one filtered by kind.

```julia
min_area_rule(area::Real) -> SpaceRule
min_area_rule(kind::Symbol, area::Real) -> SpaceRule
```

```julia
# Every space must be at least 4 m^2
min_area_rule(4.0)

# Bedrooms must be at least 9 m^2
min_area_rule(:bedroom, 9.0)
```

#### max_area_rule

Maximum area constraint for spaces of a given kind.

```julia
max_area_rule(kind::Symbol, area::Real) -> SpaceRule
```

```julia
# Bathrooms must not exceed 12 m^2
max_area_rule(:wc, 12.0)
```

#### has_door_rule

Every space (or every space of a given kind) must have at least one door.

```julia
has_door_rule() -> SpaceRule
has_door_rule(kind::Symbol) -> SpaceRule
```

```julia
# Every space must have a door
has_door_rule()

# Only offices must have a door
has_door_rule(:office)
```

#### has_connection_rule

Every space must have at least one connection (door, window, or arch).

```julia
has_connection_rule() -> SpaceRule
```

### Custom Rules

Create a `SpaceRule` with a name and a function `(Space, BuildResult) -> Union{Nothing, String}`:

```julia
# Every bedroom must have at least one window
bedroom_window_rule = SpaceRule(
  "Bedrooms must have a window",
  (space, result) ->
    space.kind == :bedroom && isempty(space_windows(result, space)) ?
      "$(space.name): bedroom has no window" :
      nothing)

add_rule(plan, bedroom_window_rule)
```

The check function has access to the full `BuildResult`, so it can query boundaries, elements, areas, and adjacency to express arbitrarily complex constraints.

## IFC Alignment

The Spaces module is designed with IFC (Industry Foundation Classes) concepts in mind:

- **`Space`** corresponds to `IfcSpace`. The `kind` field mirrors `IfcSpaceTypeEnum` values (though using Julia symbols rather than the IFC enumeration).
- **`SpaceBoundary`** corresponds to `IfcRelSpaceBoundary`. It records the relationship between a space and an element that bounds it, with `related_space` providing the 2nd-level boundary information (which space is on the other side).
- The **`space_boundaries`** query is analogous to the `IfcSpace.BoundedBy` inverse relationship.
- The `:physical` / `:virtual` boundary classification distinguishes solid walls from openings, paralleling IFC's `PhysicalOrVirtualEnum`.
- The `:interior` / `:exterior` side classification parallels IFC's `InternalOrExternalEnum`.

This alignment means that `BuildResult` data can be mapped directly to IFC export, preserving the semantic space-boundary-element relationships that IFC-based tools (energy analysis, facility management) depend on.

## See Also

- [Spaces Tutorial](../tutorials/spaces_tutorial.md) -- a guided walkthrough covering house design, parameterized layouts (office grids, radial plans), and custom validation rules.
- [Wall Graph](wall_graph.md) -- the junction-aware wall network layer used internally by `build()`. Can also be used directly for precise wall layout control.
- [Wall Graph Tutorial](../tutorials/wall_graph_tutorial.md) -- a guided walkthrough of direct wall graph construction.
- [Vertical Elements](vertical_elements.md) -- reference for `wall`, `door`, and `window`, which are the BIM primitives that `build()` generates.
- [Horizontal Elements](horizontal_elements.md) -- reference for `slab`, generated for each space when `generate_slabs` is `true`.
