# Leaf Constructors

Leaf nodes are the terminal elements of a [`SpaceDesc`](@ref) tree.
They represent concrete spaces (or absence of space) that composite
combinators compose into larger designs. Every non-trivial design
starts with these.

```julia
using KhepriBase

living = room(:living, :living_room, 5.0, 4.0)                 # 5×4 m living room
bedroom = room(:bed, :bedroom, 4.0, 3.0; height=3.0,           # with custom height
               props=(orientation=:south,))                    # and free-form props
spacer = void(1.5, 0.0)                                        # 1.5 m horizontal gap
shell  = envelope(20.0, 12.0, 3.0; id=:floor_1)                # 20×12 m shell, 3 m high
```

A zero-sized `void()` is the identity for `beside_x`, `beside_y`,
and `above`, so it can be dropped into a composition without
affecting the result. A non-zero `void(w, d)` reserves space as a
spacer.

```@docs
room
void
envelope
polar_envelope
```
