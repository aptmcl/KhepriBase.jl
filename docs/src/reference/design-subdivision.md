# Subdivision

Top-down operations that start with a large shell and carve it into
named zones. Mixes freely with the bottom-up combinators: any
subdivided node can be replaced later via [`refine`](@ref) or
labelled via [`assign`](@ref).

- `subdivide_x`, `subdivide_y` take proportional ratios summing to 1.
- `split_x`, `split_y` take absolute positions (split lines).
- `partition_x`, `partition_y` divide into equal parts.
- `carve` places a named rectangle at an absolute position within a
  zone.
- `refine` replaces a named zone with the output of a transformation.
- `assign` / `assign_all` stamp `use` and `props` onto named zones.

```julia
# Split a 20×12 shell into entry (6 m) + middle (10 m) + service (4 m) zones.
envelope(20.0, 12.0, 3.0) |>
  d -> split_x(d, [6.0, 16.0], [:entry, :middle, :service]) |>
  assign(:entry, :entrance) |>
  assign(:middle, :open_office) |>
  assign(:service, :storage)
```

```@docs
subdivide_x
subdivide_y
split_x
split_y
partition_x
partition_y
carve
refine
assign
assign_all
subdivide_remaining
subdivide_radial
subdivide_angular
partition_angular
partition_radial
```
