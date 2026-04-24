# Combinators

Pure functions that compose [`SpaceDesc`](@ref) values into larger
trees. The core vocabulary is tiny — `beside_x`, `beside_y`,
`above`, `repeat_unit`, `grid` — with transforms (`scale`,
`mirror_*`, `with_height`) layered on top. Infix operators give the
same behaviour in a compact syntax.

```julia
# Two rooms side by side on the x axis (|), stacked in depth on y (/),
# then a second storey above the first (^).
ground = room(:living, :living_room, 5.0, 4.0) | room(:kitchen, :kitchen, 3.0, 4.0)
second = room(:bed1, :bedroom, 4.0, 3.0) | room(:bed2, :bedroom, 4.0, 3.0)
house = second ^ ground                           # axis precedence: ^ > / > |

# A 3-wide column of identical studios:
studios = repeat_unit(room(:u, :studio, 6.0, 5.0), 3; axis=:x)

# A 4×2 array of 5×5 m clinics:
ward = grid((r, c) -> room(Symbol("c_$(r)_$(c)"), :clinic, 5.0, 5.0), 2, 4)
```

A zero-sized `void()` is the identity for all three composition
operators, so it can be spliced into a composition without effect
(see [Leaf Constructors](design-leaves.md)).

## Spatial Composition

```@docs
beside
beside_x
beside_y
above
```

## Infix Operators

```@docs
Base.:(|)(::SpaceDesc, ::SpaceDesc)
Base.:(/)(::SpaceDesc, ::SpaceDesc)
Base.:(^)(::SpaceDesc, ::SpaceDesc)
```

## Repetition and Grid

```@docs
repeat_unit
grid(::Any, ::Any, ::Any)
```

## Transforms

```@docs
scale(::SpaceDesc, ::Any)
mirror_x
mirror_y
with_height
with_props
tag_wall_family
tag_slab_family
```
