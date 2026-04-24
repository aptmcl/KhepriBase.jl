# Tree Queries

Pure functions for inspecting a [`SpaceDesc`](@ref) tree before
laying it out. They return the bounding dimensions of the composed
tree and the set of ids and annotations it carries — useful when
writing constraints, custom combinators, or debugging a design.

```julia
house = (room(:living, :living_room, 5.0, 4.0) |
         room(:kitchen, :kitchen, 3.0, 4.0)) /
        (room(:bed, :bedroom, 4.0, 3.0) |
         room(:bath, :bathroom, 2.5, 3.0))

desc_width(house)   # 8.0
desc_depth(house)   # 7.0
desc_height(house)  # 2.8 (default room height)
collect_ids(house)  # [:living, :kitchen, :bed, :bath]
```

```@docs
desc_width
desc_depth
desc_height
collect_ids
collect_annotations
update_room_by_id
```
