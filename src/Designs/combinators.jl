# ---- Spatial Combinators ----
# Pure functions that compose SpaceDesc values into new SpaceDesc values.
# A zero-sized `void()` is treated as the identity for `beside_x`, `beside_y`,
# and `above`, so the combinators form a near-monoid under each axis.

_is_identity_void(x) = x isa Void && x.width == 0.0 && x.depth == 0.0

# ---- Horizontal adjacency (along x) ----

"""
    beside_x(a, b; shared_wall=true, align=:start)

Place two spaces side by side along the x-axis. A zero-sized `void()` on either
side is elided, so `beside_x(void(), a) === a`.
"""
beside_x(a, b; shared_wall=true, align=:start) =
  _is_identity_void(a) ? b :
  _is_identity_void(b) ? a :
  BesideX(a, b, shared_wall, align)

# ---- Depth adjacency (along y) ----

"""
    beside_y(a, b; shared_wall=true, align=:start)

Place two spaces side by side along the y-axis (depth). A zero-sized `void()`
on either side is elided, so `beside_y(void(), a) === a`.
"""
beside_y(a, b; shared_wall=true, align=:start) =
  _is_identity_void(a) ? b :
  _is_identity_void(b) ? a :
  BesideY(a, b, shared_wall, align)

# ---- General beside (2 args) ----

"""
    beside(a, b; axis=:x, shared_wall=true, align=:start)
    beside(a, b, rest...; axis=:x, shared_wall=true, align=:start)

Place two or more spaces adjacent along the given `axis` (`:x` or `:y`).
The variadic form folds left over all arguments.
"""
beside(a, b; axis=:x, shared_wall=true, align=:start) =
  axis == :x ? beside_x(a, b; shared_wall, align) :
               beside_y(a, b; shared_wall, align)

# ---- General beside (variadic) ----

beside(a, b, rest...; axis=:x, shared_wall=true, align=:start) =
  reduce((x, y) -> beside(x, y; axis, shared_wall, align), (a, b, rest...))

# ---- Vertical stacking (2 args) ----

"""
    above(a, b; slab_between=true)
    above(a, b, rest...; slab_between=true)

Stack two or more spaces vertically. `a` sits on top of `b`. A zero-sized
`void()` on either side is elided, so `above(void(), a) === a`.
The variadic form folds left over all arguments.
"""
above(a, b; slab_between=true) =
  _is_identity_void(a) ? b :
  _is_identity_void(b) ? a :
  Above(a, b, slab_between)

# ---- Vertical stacking (variadic) ----

above(a, b, rest...; slab_between=true) =
  reduce((x, y) -> above(x, y; slab_between), (a, b, rest...))

# ---- Repetition ----

"""
    repeat_unit(unit, n; axis=:x, mirror_alternate=false)

Repeat `unit` `n` times along `axis`. When `mirror_alternate` is true,
even-indexed copies are mirrored.
"""
function repeat_unit(unit, n; axis=:x, mirror_alternate=false)
  n < 1 && error("repeat_unit: count must be >= 1, got $n")
  n == 1 && return unit
  Repeated(unit, n, axis, mirror_alternate)
end

# ---- Grid ----

"""
    grid(cell_fn, rows, cols)

Create a grid layout. `cell_fn(row, col)` is called for each cell
and must return a `SpaceDesc`.
"""
grid(cell_fn, rows, cols) = GridLayout(cell_fn, rows, cols)

# ---- Transformations ----

"""
    scale(s, sx, sy=sx)

Scale a space description by `sx` along x and `sy` along y.
"""
scale(s::SpaceDesc, sx, sy=sx) = Scaled(s, Float64(sx), Float64(sy))

"""
    mirror_x(s)

Mirror a space description about the x-axis.
"""
mirror_x(s) = Mirrored(s, :x)

"""
    mirror_y(s)

Mirror a space description about the y-axis.
"""
mirror_y(s) = Mirrored(s, :y)

"""
    with_height(s, h)

Override the floor-to-ceiling height of a space to `h`.
"""
with_height(s, h) = HeightOverride(s, Float64(h))

"""
    with_props(s, props)

Return a space description that merges the given `props` `NamedTuple` into
every placed space under `s` at layout time. Existing per-space props take
precedence over the overlay.
"""
with_props(s, props) = PropsOverlay(s, NamedTuple(props))

"""
    tag_wall_family(s, family_name)

Attach `family_name` as a `wall_family` prop on every placed space under `s`.
At generation time, walls between spaces that agree on the family name use
the corresponding `WallFamilyDef` (from AA's element-rule layer) is
looked up in `rules.families.walls`, overriding the rule system.
Named `tag_` rather than `with_` to avoid a collision with Khepri's
`with_wall_family` context manager.
"""
tag_wall_family(s, family_name) = with_props(s, (wall_family=Symbol(family_name),))

"""
    tag_slab_family(s, family_name)

Attach `family_name` as a `slab_family` prop on every placed space under `s`.
At generation time, each floor slab looks the name up in
`rules.families.slabs` and uses it in place of the default `SlabSpec`.
Named `tag_` rather than `with_` to avoid a collision with Khepri.
"""
tag_slab_family(s, family_name) = with_props(s, (slab_family=Symbol(family_name),))

# ---- Annotation combinators ----

"""
    connect(desc, from, to; kind=:door, width=nothing, height=nothing)

Annotate a connection (e.g. door or opening) between zones `from` and `to`.
"""
connect(desc, from, to; kind=:door, width=nothing, height=nothing) =
  Annotated(desc, ConnectAnnotation(from, to, kind,
    isnothing(width) ? nothing : Float64(width),
    isnothing(height) ? nothing : Float64(height)))

"""
    connect_exterior(desc, space_id; kind=:window, face=:auto, count=nothing, width=nothing, height=nothing)

Annotate a connection from zone `space_id` to the building exterior
(e.g. a window or vent).
"""
connect_exterior(desc, space_id; kind=:window, face=:auto, count=nothing,
                 width=nothing, height=nothing) =
  Annotated(desc, ConnectExteriorAnnotation(space_id, kind, face,
    isnothing(count) ? nothing : Int(count),
    isnothing(width) ? nothing : Float64(width),
    isnothing(height) ? nothing : Float64(height)))

"""
    disconnect(desc, from, to)

Remove any connection between zones `from` and `to`.
"""
disconnect(desc, from, to) = Annotated(desc, DisconnectAnnotation(from, to))

"""
    no_windows(desc, space_id)

Suppress automatic window generation for zone `space_id`.
"""
no_windows(desc, space_id) = Annotated(desc, NoWindowsAnnotation(space_id))

# ---- Subdivision combinators ----

"""
    subdivide_x(desc, ratios, ids)

Subdivide a space along x into strips with the given `ratios` (must sum to 1)
and label each strip with the corresponding entry from `ids`.
"""
function subdivide_x(desc, ratios, ids)
  rv = Float64.(collect(ratios))
  iv = Symbol.(collect(ids))
  length(rv) == length(iv) || error("subdivide_x: ratios and ids must have same length")
  abs(sum(rv) - 1.0) < 1e-10 || error("subdivide_x: ratios must sum to 1.0, got $(sum(rv))")
  Subdivided(desc, :x, rv, iv)
end

"""
    subdivide_y(desc, ratios, ids)

Subdivide a space along y into strips with the given `ratios` (must sum to 1)
and label each strip with the corresponding entry from `ids`.
"""
function subdivide_y(desc, ratios, ids)
  rv = Float64.(collect(ratios))
  iv = Symbol.(collect(ids))
  length(rv) == length(iv) || error("subdivide_y: ratios and ids must have same length")
  abs(sum(rv) - 1.0) < 1e-10 || error("subdivide_y: ratios must sum to 1.0, got $(sum(rv))")
  Subdivided(desc, :y, rv, iv)
end

"""
    split_x(desc, positions, ids)

Split a space along x at the given absolute `positions`, producing
`length(positions) + 1` zones labeled by `ids`. Positions must be
strictly ascending and lie inside `(0, desc_width(desc))`.

    split_x(envelope(10, 5, 3), [3.0, 8.0], [:a, :b, :c])

yields zones of width 3, 5, and 2 metres.
"""
split_x(desc, positions, ids) = _split(desc, :x, positions, ids)

"""
    split_y(desc, positions, ids)

Split a space along y at the given absolute `positions`, producing
`length(positions) + 1` zones labeled by `ids`.
"""
split_y(desc, positions, ids) = _split(desc, :y, positions, ids)

function _split(desc, axis, positions, ids)
  pv = Float64.(collect(positions))
  iv = Symbol.(collect(ids))
  length(iv) == length(pv) + 1 ||
    error("split_$axis: expected $(length(pv) + 1) ids for $(length(pv)) split positions, got $(length(iv))")
  total = axis == :x ? desc_width(desc) : desc_depth(desc)
  total > 0 || error("split_$axis: base has zero extent along $axis")
  issorted(pv) || error("split_$axis: positions must be ascending")
  (isempty(pv) || (0 < pv[1] && pv[end] < total)) ||
    error("split_$axis: positions must lie strictly within (0, $total)")
  widths = isempty(pv) ? Float64[total] : [pv[1]; diff(pv); total - pv[end]]
  Subdivided(desc, axis, widths ./ total, iv)
end

"""
    partition_x(desc, n, id_prefix)

Partition a space into `n` equal strips along x, labeling them
`id_prefix_1`, `id_prefix_2`, etc.
"""
partition_x(desc, n, id_prefix) = Partitioned(desc, :x, n, id_prefix)

"""
    partition_y(desc, n, id_prefix)

Partition a space into `n` equal strips along y, labeling them
`id_prefix_1`, `id_prefix_2`, etc.
"""
partition_y(desc, n, id_prefix) = Partitioned(desc, :y, n, id_prefix)

"""
    carve(desc, id, use; x, y, width, depth)

Carve out a rectangular sub-zone at position (`x`, `y`) with the given
`width` and `depth`, labeling it `id` with usage `use`.
"""
carve(desc, id, use; x, y, width, depth) =
  Carved(desc, id, use, Float64(x), Float64(y), Float64(width), Float64(depth))

"""
    refine(desc, zone_id, transform)
    refine(zone_id, transform)

Apply `transform` (a function `SpaceDesc -> SpaceDesc`) to the sub-zone
identified by `zone_id`. The two-argument form returns a curried closure
suitable for piping.
"""
refine(desc, zone_id, transform) = Refined(desc, zone_id, transform)

refine(zone_id, transform) = desc -> refine(desc, zone_id, transform)

"""
    assign(desc, zone_id, use; props=(;))
    assign(zone_id, use; props=(;))

Assign a programmatic `use` (e.g. `:bedroom`) and optional `props` to zone
`zone_id`. The two-argument form returns a curried closure suitable for piping.
"""
assign(desc, zone_id, use; props=(;)) = Assigned(desc, zone_id, use, props)

assign(zone_id, use; props=(;)) = desc -> assign(desc, zone_id, use; props)

"""
    assign_all(desc, id_prefix, use; props=(;))
    assign_all(id_prefix, use; props=(;))

Assign `use` and `props` to every zone whose id starts with `id_prefix_`.
The two-argument form returns a curried closure suitable for piping.
"""
function assign_all(desc, id_prefix, use; props=(;))
  prefix_str = String(id_prefix) * "_"
  ids = filter(collect_ids(desc)) do id
    s = String(id)
    startswith(s, prefix_str) && all(isdigit, s[length(prefix_str)+1:end])
  end
  result = desc
  for id in ids
    result = assign(result, id, use; props)
  end
  result
end

assign_all(id_prefix, use; props=(;)) = desc -> assign_all(desc, id_prefix, use; props)

"""
    subdivide_remaining(desc, blocks)
    subdivide_remaining(blocks)

Given a base `desc` containing a single central carved hole, add perimeter
blocks around that hole. `blocks` is a vector of `(block_id, position)`
pairs where `position` is `:north`, `:south`, `:east`, or `:west`.

```julia
envelope(50.0, 50.0, 3.0) |>
  d -> carve(d, :courtyard, :garden; x=12, y=12, width=26, depth=26) |>
  subdivide_remaining([
    (:north_block, :north),
    (:south_block, :south),
    (:east_block,  :east),
    (:west_block,  :west)])
```

The two-argument form returns a curried closure suitable for piping.
"""
subdivide_remaining(desc::SpaceDesc, blocks) =
  SubdivideRemaining(desc, [(Symbol(b[1]), Symbol(b[2])) for b in blocks])

subdivide_remaining(blocks) = desc -> subdivide_remaining(desc, blocks)
