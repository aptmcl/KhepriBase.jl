# Utilities

This page documents the utility functions exported by KhepriBase, defined
primarily in `Utils.jl`.

## Division and Mapping

`division` and `map_division` subdivide numeric ranges (or arbitrary objects) into
evenly spaced samples.  They are the idiomatic way to generate parameter sweeps,
grids, and discretizations in Khepri.

### `division`

```julia
division(t0, t1, n::Int, include_last::Bool=true)
division((t0, t1)::Tuple, n::Int, include_last::Bool=true)
division(obj, n::Int)
```

Returns a `Vector` of `n+1` evenly spaced values from `t0` to `t1` (inclusive).
When `include_last` is `false`, the endpoint `t1` is excluded and only `n`
values are returned.

The tuple variant unpacks the endpoints from a 2-tuple.  The single-object
variant calls `map_division(identity, obj, n)`, which allows any type that
implements `map_division` (e.g., paths and shapes) to be subdivided.

### `map_division`

```julia
map_division(f, t0, t1, n::Int, include_last::Bool=true)
```

Like `division`, but applies `f` to each sample point and returns the mapped
results.

#### 2D grid variants

```julia
map_division(f, u0, u1, nu, include_last_u, v0, v1, nv)
map_division(f, u0, u1, nu, v0, v1, nv, include_last_v=true)
map_division(f, u0, u1, nu, include_last_u, v0, v1, nv, include_last_v)
```

These produce a nested `Vector{Vector{...}}` by sweeping two parameters.  `f`
receives two arguments `(u, v)`.  The `include_last_*` flags independently
control whether the last sample in each direction is included.

**Example -- sampling a surface:**

```julia
map_division(0, 1, 10, 0, 2pi, 32) do u, v
  xyz(u * cos(v), u * sin(v), 0)
end
```

## Grasshopper-Compatibility Functions

These functions mirror common Grasshopper components to ease porting of visual
scripts to Julia.

| Function | Signature | Description |
|----------|-----------|-------------|
| `series` | `series(start, step, count)` | Generate `count` values starting at `start` with the given `step`. |
| `crossref` | `crossref(as, bs)` | Cartesian product of two arrays as a matrix of `(a, b)` tuples. |
| `crossref_holistic` | `crossref_holistic(arr1, arr2)` | Flattened cross-reference returning two aligned arrays. |
| `remap` | `remap(value, (min_in, max_in), (min_out, max_out))` | Linearly remap `value` from one range to another. |
| `cull` | `cull(template, as)` | Keep elements of `as` where the cycling boolean `template` is `true`. |
| `map_longest` | `map_longest(f, args...)` | Map `f` over parallel arrays, extending shorter ones by repeating their last element. |
| `list_item` | `list_item(L, i)` | Wrap-around index into `L`. Accepts a single index or an array of indices. |
| `cull_pattern` | `cull_pattern(L, P)` | Filter `L` by cycling boolean pattern `P`. |
| `shift_list` | `shift_list(L, s)` | Circular-shift `L` by `s` positions (positive = shift left). |
| `cull_index` | `cull_index(L, I)` | Remove elements at wrap-around indices `I` from `L`. |
| `repeat_data` | `repeat_data(L, n)` | Repeat elements of `L` cyclically to produce `n` elements. |
| `duplicate_data` | `duplicate_data(L, n)` | Duplicate each element of `L` `n` times, in order. |
| `random_values` | `random_values(domain, n, seed)` | Generate `n` random values in `[domain[1], domain[2]]` using `MersenneTwister(seed)`. |
| `grid_rectangular` | `grid_rectangular(p, xn, yn, xs=1, ys=1)` | Grid of locations centered at `p`, extending `xn`/`yn` cells in each direction with spacing `xs`/`ys`. |

## Random Number Functions

KhepriBase provides its own deterministic pseudo-random number generator for
reproducible designs.  It uses a linear congruential generator seeded by the
`random_seed` parameter (default `12345`).

| Function | Signature | Description |
|----------|-----------|-------------|
| `set_random_seed` | `set_random_seed(v::Int)` | Set the global random seed. |
| `random` | `random(x::Int)` | Random integer in `[0, x)`. |
| `random` | `random(x::Real)` | Random real in `[0, x)`. |
| `random_range` | `random_range(x0, x1)` | Random value in `[x0, x1)`. Returns `x0` when `x0 == x1`. |

The `random_seed` itself is a `Parameter{Int}` and can be used with the
`with(random_seed, value) do ... end` pattern.

## Color Constructors

Colors in KhepriBase are backed by the `ColorTypes` package.

| Name | Type | Description |
|------|------|-------------|
| `rgb` | Alias for `RGB` | Construct an RGB color: `rgb(r, g, b)` with channels in `[0, 1]`. |
| `rgba` | Alias for `RGBA` | Construct an RGBA color: `rgba(r, g, b, a)`. |
| `rgb_radiance` | `rgb_radiance(c::RGB) -> Real` | Perceptual luminance: `0.265*red(c) + 0.67*green(c) + 0.065*blue(c)`. |

Hex and named-color support is available through `ColorTypes` itself (e.g.,
`parse(Colorant, "#ff0000")`), but is not re-exported by KhepriBase.

## Sun Position

```julia
sun_pos(year, month, day, hour, minute, Lstm, latitude, longitude)
  -> (altitude, azimuth)

sun_pos(date::DateTime, timezone, latitude, longitude)
  -> (altitude, azimuth)
```

Computes the sun's altitude and azimuth angles (in degrees) for a given date,
time, and geographic location.  Based on Paul Schlyter's "How to compute
planetary positions" algorithm.

- `Lstm` / `timezone` -- standard meridian of the local time zone (degrees).
- `latitude`, `longitude` -- geographic coordinates (degrees).
- Returns a tuple `(altitude, azimuth)` in degrees.

A warning is logged when `|longitude - Lstm| > 30`.

The `DateTime` convenience form extracts year, month, day, hour, and minute
automatically.

## Immutable List Type

KhepriBase provides a functional, singly-linked `List{T}` type (`Nil` / `Cons`)
for use in recursive algorithms.

| Function | Signature | Description |
|----------|-----------|-------------|
| `nil` | constant | The empty list (`Nil{Union{}}`). |
| `list` | `list(elts...)` | Construct a list from elements. Also accepts a `Generator`. |
| `cons` | `cons(head, tail)` | Prepend an element. |
| `head` | `head(lst::Cons)` | First element. |
| `tail` | `tail(lst::Cons)` | Rest of list. |

Standard `Base` iteration, `length`, `map`, `filter`, `getindex`, `isempty`,
`cat`, and `show` are implemented.

## Miscellaneous

| Function | Signature | Description |
|----------|-----------|-------------|
| `path_replace_suffix` | `path_replace_suffix(path, suffix)` | Replace the file extension in a path string. |
| `reverse_dict` | `reverse_dict(dict)` | Invert a dictionary, collecting keys that map to the same value. |
| `required` | `required()` | Sentinel that throws an error; used as a default for mandatory parameters. |
| `PNGFile`, `PDFFile`, `DVIFile` | Wrappers | Wrap a file path for integration with the Julia display system (`Base.show` for `MIME"image/png"` and `MIME"image/svg+xml"`). |
