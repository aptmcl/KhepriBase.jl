# Design Types

Every design is a tree of immutable [`SpaceDesc`](@ref) values.
Leaves (`Room`, `Void`, `Envelope`, `PolarEnvelope`) carry the
physical dimensions; composite, transform, and subdivision nodes
wrap children to describe how they relate. Prefer the constructor
functions ([`room`](@ref), [`beside_x`](@ref), [`subdivide_x`](@ref),
…) rather than calling the struct constructors directly — the
functions enforce invariants and provide identity-elision for
`Void`.

## The type hierarchy at a glance

Nodes fall into four roles. Each role is handled by a distinct
branch in the `layout(desc)` compiler:

| Role          | Types | What the compiler does |
|---------------|-------|------------------------|
| Leaf          | `Room`, `Void`, `Envelope`, `PolarEnvelope` | Place (or skip) a concrete `Space` at the current cursor. |
| Composition   | `BesideX`, `BesideY`, `Above`, `Repeated`, `GridLayout` | Recurse into children, advancing the cursor between them. |
| Transform     | `Scaled`, `Mirrored`, `HeightOverride`, `PropsOverlay`, `Annotated` | Lay out the child subtree, then rewrite each placed space's geometry or metadata. |
| Subdivision   | `Subdivided`, `Partitioned`, `Carved`, `Refined`, `Assigned`, `SubdivideRemaining`, `SubdividedPolar`, `PartitionedPolar` | Start from a zone or envelope and carve named sub-zones inside it. |

## Leaves

Leaves are the terminals. Two of them produce a `Space` in the
compiled `Layout`; `Void` does not.

- `Room` — a named room with `use`, `(width, depth, height)` and
  free-form `props`. This is the common case.
- `Envelope` — an anonymous-ish shell that composite nodes carve
  into named zones. Conceptually a "room-shaped slot" whose `use`
  defaults to `:envelope` unless overridden by subsequent
  subdivision.
- `Void` — reserves width/depth as a spacer but produces no `Space`.
  A zero-sized `Void` is treated as the identity for `beside_x`,
  `beside_y`, and `above`, so it can be spliced into generated trees
  without affecting layout.
- `PolarEnvelope` — polar-coordinate envelope (a ring sector). Used
  as the root of a polar subtree; polar subdivision nodes
  (`SubdividedPolar`, `PartitionedPolar`) split it into rings or
  wedges.

## Composition

Composition nodes place two or more children next to each other.
Their cursor-handling logic is what the infix operators `|`, `/`,
and `^` expand to.

- `BesideX` — two children laid out along x (`a | b`).
- `BesideY` — two children laid out along y (`a / b`).
- `Above` — two children stacked along z (`b ^ a`, with `a` on top).
  `slab_between` controls whether the compiler emits a dividing
  slab.
- `Repeated` — one unit replicated `count` times along an axis, with
  optional `mirror_alternate` for pinwheel patterns.
- `GridLayout` — a 2D grid built by calling `cell_fn(row, col)` for
  each cell; the row/col widths come from the maximum extent of the
  cell outputs.

## Transforms

Transforms have one child. The compiler first lays the child out in
a temporary map, then rewrites each placed `Space` according to the
transform's rule before splicing the result back into the main
layout.

- `Scaled` — multiply each placed space's origin and dimensions by
  `(sx, sy)`.
- `Mirrored` — flip each space across the subtree's x- or y-axis.
- `HeightOverride` — replace the per-space `height` on every space
  laid out at the current level.
- `PropsOverlay` — merge a `NamedTuple` into each placed space's
  `props` (used by `with_props`, `tag_wall_family`,
  `tag_slab_family`).
- `Annotated` — attach an `Annotation` (see
  [Annotations](design-annotations.md)) to the subtree. Transparent
  to layout; the annotation is collected by
  `collect_annotations(desc)` for downstream consumption.

## Subdivision

Subdivision nodes invert the composition direction: instead of
"combine small things into a big thing", they describe "start with
a shell and name the pieces".

- `Subdivided` — split along an axis into ratio-weighted zones with
  explicit ids. Rectangular case of `subdivide_x` / `subdivide_y`.
- `Partitioned` — split along an axis into `count` equal cells
  sharing an id prefix. The "just divide this into five" form.
- `Carved` — punch a named rectangle at an absolute `(x, y, w, d)`
  inside the zone.
- `Refined` — replace a named zone with the output of a
  `transform(zone_env)` function. The zone's envelope (either
  rectangular or polar) is fed in so the transform can build a
  nested sub-layout inside it.
- `Assigned` — stamp a `use` (and optional `props`) onto a named
  zone without changing its geometry. Typically chained after a
  `split_x` to label the produced zones.
- `SubdivideRemaining` — given an envelope with a single `carve`
  inside it, name the four remaining L-shaped blocks (north / south
  / east / west of the carve) in one move. Used for
  room-plus-corridor patterns.
- `SubdividedPolar`, `PartitionedPolar` — polar analogues of
  `Subdivided` / `Partitioned`, splitting a `PolarEnvelope` into
  radial rings or angular wedges.

## Supertype

```@docs
SpaceDesc
```

## Leaf types

```@docs
Room
Void
Envelope
PolarEnvelope
```

## Composition types

```@docs
BesideX
BesideY
Above
Repeated
GridLayout
```

## Transform types

```@docs
Scaled
Mirrored
HeightOverride
PropsOverlay
Annotated
```

## Subdivision types

```@docs
Subdivided
Partitioned
Carved
Refined
Assigned
SubdivideRemaining
SubdividedPolar
PartitionedPolar
```

## See also

- [Designs (Level 2)](../concepts/designs.md) — narrative intro
- [Leaf Constructors](design-leaves.md) — the preferred constructors
- [Combinators](design-combinators.md) — construction of composition
  and transform nodes via function calls and infix operators
- [Subdivision](design-subdivision.md) — construction of the
  subdivision nodes
- [Annotations](design-annotations.md) — the `Annotation` subtree
  attached by `Annotated`
- [Layout Engine](layout-engine.md) — what `layout(desc)` does with
  each type above
