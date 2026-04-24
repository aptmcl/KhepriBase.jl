# Constraints

Function and type reference for KhepriBase's constraint system. See
[Constraints (concept)](../concepts/constraints.md) for the narrative
treatment — when to use which severity, how context polymorphism
works, and how the fixer loop plugs in.

Every library constructor below returns a [`Constraint`](@ref) whose
`check(ctx)` runs against either a [`Layout`](@ref) (multi-storey,
declarative) or a `BuildResult` (single-storey, imperative). A
handful of constraints (`has_door`, `has_connection`) need
door/connection data only present on a `BuildResult`; the rest are
context-polymorphic.

## Types

```@docs
ConstraintSeverity
HARD
SOFT
PREFERENCE
ConstraintCategory
DIMENSIONAL
ADJACENCY
AREA_PROPORTION
CIRCULATION
ENVIRONMENTAL
Constraint
Violation
ConstraintSet
ValidationResult
ConstraintFixer
```

## Validation

```@docs
validate
report
```

## Algebra

Constraints compose algebraically; every combinator below returns a
new `Constraint`.

```@docs
combine
either
when
with_severity
merge_constraints
```

## Constraint Library

### Dimensional

```@docs
min_area
max_area
min_dimension
max_aspect_ratio
floor_height_range
min_corridor_width
area_ratio
```

### Adjacency

```@docs
must_adjoin
must_not_adjoin
vertical_alignment
```

### Circulation

```@docs
all_reachable
max_dead_end
min_egress_routes
has_door
has_connection
```

### Environmental

```@docs
preferred_orientation
facade_ratio
min_exterior_exposure
```

## Fixer Loop

[`ConstraintFixer`](@ref) pairs a constraint-name substring with a
rewrite function; [`apply_fixers`](@ref) iterates build → validate →
first matching fix until the design is hard-clean, no fixer matches,
or `max_iters` is reached.

```@docs
apply_fixers
```

## See also

- [Constraints (concept)](../concepts/constraints.md) — narrative,
  examples, validation workflow, severity conventions.
- [Layout Engine](layout-engine.md) — produces the `Layout` that
  constraints consume.
- [Adjacencies](adjacencies.md) — the data adjacency-category
  constraints inspect.
