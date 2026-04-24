# Shapes and Geometry

This page documents all shape constructors (organized by dimensionality), CSG
operations, high-level modeling functions, and geometric utilities.

Shapes in Khepri are lazy proxies.  They are only realized in a backend when
their reference is needed.  Each shape carries a `material` parameter that
defaults to a dimension-appropriate material (point, curve, surface, or solid).

## Shape0D -- Points and Text

| Shape | Constructor Signature | Description |
|-------|----------------------|-------------|
| `point` | `point(position::Loc=u0(); material)` | A single point. |
| `text` | `text(str::String="", corner::Loc=u0(), height::Real=1; material)` | A text annotation placed at `corner` with the given `height`. |
| `text_centered` | `text_centered(str="", center=u0(), height=1)` | Convenience: centers the text horizontally and vertically around `center`. |
| `block` | `block(name::String="Block", shapes::Shapes=Shape[])` | A named collection of shapes (block definition). |
| `block_instance` | `block_instance(block::Block, loc::Loc=u0(), scale::Real=1.0)` | An instance of a previously defined block. |

## Shape1D -- Curves

| Shape | Constructor Signature | Description |
|-------|----------------------|-------------|
| `line` | `line(vertices::Locs=[u0(), ux()]; material)` | Open polyline through `vertices`. Also: `line(v0, v1, vs...)`. |
| `closed_line` | `closed_line(vertices::Locs=[u0(), ux(), uy()]; material)` | Closed polyline. Also: `closed_line(v0, v1, vs...)`. |
| `polygon` | `polygon(vertices::Locs=[u0(), ux(), uy()]; material)` | Alias for closed polyline. Also: `polygon(v0, v1, vs...)`. |
| `spline` | `spline(points::Locs=[u0(), ux(), uy()], v0=false, v1=false; material)` | Open spline through `points`. Optional start/end tangent vectors `v0`, `v1` (`Vec` or `false`). Also: `spline(v0, v1, vs...)`. |
| `closed_spline` | `closed_spline(points::Locs=[u0(), ux(), uy()]; material)` | Closed spline. Also: `closed_spline(v0, v1, vs...)`. |
| `circle` | `circle(center::Loc=u0(), radius::Real=1; material)` | Circle. |
| `arc` | `arc(center::Loc=u0(), radius::Real=1, start_angle::Real=0, amplitude::Real=pi; material)` | Circular arc. |
| `ellipse` | `ellipse(center::Loc=u0(), radius_x::Real=1, radius_y::Real=1; material)` | Ellipse. |
| `elliptic_arc` | `elliptic_arc(center::Loc=u0(), radius_x::Real=1, radius_y::Real=1, start_angle::Real=0, amplitude::Real=pi; material)` | Elliptic arc. |
| `regular_polygon` | `regular_polygon(edges::Integer=3, center::Loc=u0(), radius::Real=1, angle::Real=0, inscribed::Bool=true; material)` | Regular polygon. |
| `rectangle` | `rectangle(corner::Loc=u0(), dx::Real=1, dy::Real=1; material)` | Axis-aligned rectangle. Also: `rectangle(p::Loc, q::Loc)` (two-corner form). |

## Shape2D -- Surfaces

| Shape | Constructor Signature | Description |
|-------|----------------------|-------------|
| `surface_circle` | `surface_circle(center::Loc=u0(), radius::Real=1; material)` | Filled circle. |
| `surface_ring` | `surface_ring(center::Loc=u0(), inner_radius::Real=0.5, outer_radius::Real=1; material)` | Annular ring (flat washer shape). |
| `surface_arc` | `surface_arc(center::Loc=u0(), radius::Real=1, start_angle::Real=0, amplitude::Real=pi; material)` | Filled arc sector. |
| `surface_ellipse` | `surface_ellipse(center::Loc=u0(), radius_x::Real=1, radius_y::Real=1; material)` | Filled ellipse. |
| `surface_elliptic_arc` | `surface_elliptic_arc(center::Loc=u0(), radius_x::Real=1, radius_y::Real=1, start_angle::Real=0, amplitude::Real=pi; material)` | Filled elliptic arc sector. |
| `surface_polygon` | `surface_polygon(vertices::Locs=[u0(), ux(), uy()]; material)` | Filled polygon. Also: `surface_polygon(v0, v1, vs...)`. |
| `surface_regular_polygon` | `surface_regular_polygon(edges::Integer=3, center::Loc=u0(), radius::Real=1, angle::Real=0, inscribed::Bool=true; material)` | Filled regular polygon. |
| `surface_rectangle` | `surface_rectangle(corner::Loc=u0(), dx::Real=1, dy::Real=1; material)` | Filled rectangle. Also: `surface_rectangle(p, q)` (two-corner form). |
| `surface` | `surface(frontier::Shapes1D=[circle()]; material)` | Surface bounded by one or more closed curves. Also: `surface(c0, cs...)`. |
| `surface_grid` | `surface_grid(points::Matrix{<:Loc}, closed_u=false, closed_v=false, smooth_u=true, smooth_v=true; material)` | Parametric surface from a grid of points. Also accepts `Vector{Vector{Loc}}`. |
| `surface_mesh` | `surface_mesh(vertices::Locs, faces::Vector{Vector{Int}}; material)` | Triangulated or polygonal mesh. Face indices are 1-based. |

## Shape3D -- Solids

| Shape | Constructor Signature | Description |
|-------|----------------------|-------------|
| `sphere` | `sphere(center::Loc=u0(), radius::Real=1; material)` | Sphere. |
| `box` | `box(c::Loc=u0(), dx::Real=1, dy::Real=dx, dz::Real=dy; material)` | Axis-aligned box. Also: `box(c0, c1)` (two-corner form, auto-corrects negative dimensions). |
| `cylinder` | `cylinder(cb::Loc=u0(), r::Real=1, h::Real=1; material)` | Cylinder. Also: `cylinder(cb, r, ct)` (center-to-center form). |
| `cone` | `cone(cb::Loc=u0(), r::Real=1, h::Real=1; material)` | Cone. Also: `cone(cb, r, ct)`. |
| `cone_frustum` | `cone_frustum(cb::Loc=u0(), rb::Real=1, h::Real=1, rt::Real=1; material)` | Truncated cone. Also: `cone_frustum(cb, rb, ct, rt)`. |
| `torus` | `torus(center::Loc=u0(), re::Real=1, ri::Real=0.5; material)` | Torus with major radius `re` and minor radius `ri`. |
| `cuboid` | `cuboid(b0::Loc=u0(), b1, b2, b3, t0, t1, t2, t3; material)` | General hexahedron from 8 corner locations (bottom b0-b3, top t0-t3). |
| `regular_pyramid` | `regular_pyramid(edges::Integer=3, cb::Loc=u0(), rb::Real=1, angle::Real=0, h::Real=1, inscribed::Bool=true; material)` | Regular pyramid. Also: `regular_pyramid(edges, cb, rb, angle, ct, inscribed)`. |
| `regular_pyramid_frustum` | `regular_pyramid_frustum(edges::Integer=4, cb::Loc=u0(), rb::Real=1, angle::Real=0, h::Real=1, rt::Real=1, inscribed::Bool=true; material)` | Truncated regular pyramid. Also accepts a top-center location. |
| `pyramid` | `pyramid(bs::Locs=[ux(), uy(), uxy()], t::Loc=uz(); material)` | General pyramid from a base polygon and apex. |
| `pyramid_frustum` | `pyramid_frustum(bs::Locs, ts::Locs; material)` | General truncated pyramid from base and top polygons. |
| `regular_prism` | `regular_prism(edges::Integer=3, cb::Loc=u0(), r::Real=1, angle::Real=0, h::Real=1, inscribed::Bool=true; material)` | Regular prism. Also: `regular_prism(edges, cb, r, angle, ct, inscribed)`. |
| `prism` | `prism(bs::Locs=[ux(), uy(), uxy()], v::Vec=vz(1); material)` | General prism from base polygon and extrusion vector. Also: `prism(bs, h::Real)`. |
| `right_cuboid` | `right_cuboid(cb::Loc=u0(), width::Real=1, height::Real=1, h::Real=1; material)` | Centered cuboid. Also: `right_cuboid(cb, width, height, ct, angle=0)`. |
| `isosurface` | `isosurface(frep::Function=loc->sph_rho(loc), bounding_box::Locs=[xyz(-1,-1,-1), xyz(1,1,1)]; material)` | Implicit surface defined by `frep(loc) = 0`. |
| `thicken` | `thicken(shape::Shape=surface_circle(), thickness::Real=1)` | Thicken a surface into a solid. |
| `unknown` | `unknown(baseref::Any)` | Opaque wrapper for shapes imported from backends. |

## Annotation Shapes

Annotations are overlays that do not participate in boolean operations.

| Shape | Constructor Signature | Description |
|-------|----------------------|-------------|
| `dimension` | `dimension(from::Loc, to::Loc, text, size=1, offset=0.1; material)` | Linear dimension line. |
| `arc_dimension` | `arc_dimension(center, radius, start_angle, amplitude, radius_text, amplitude_text, size=1, offset=0.1; material)` | Arc dimension. |
| `labels` | `labels(p::Loc, data::Vector{NamedTuple{(:txt,:mat,:scale)}}; material)` | Multi-entry label at a point. Use `label(p, txt, mat, scale)` for incremental construction. |

## CSG Operations

Boolean operations work on shapes of matching dimensionality (2D or 3D).

### `union`

```julia
union(shape, mask)        -> United shape (2D: united_surfaces, 3D: united_solids)
union(shape, shapes...)   -> Left fold of union
union(shapes::Shapes)     -> Splat form
union(shape, empty_shape()) -> shape  (identity)
```

### `subtraction`

```julia
subtraction(shape, mask)        -> Subtracted shape
subtraction(shape, shapes...)   -> Left fold of subtraction
subtraction(shapes::Shapes)     -> Splat form
subtraction(shape, empty_shape()) -> shape  (identity)
```

### `intersection`

```julia
intersection(shape, mask)        -> Intersected shape
intersection(shape, shapes...)   -> Left fold of intersection
intersection(shapes::Shapes)     -> Splat form
intersection(shape, universal_shape()) -> shape  (identity)
```

### `slice`

```julia
slice(shape::Shape3D, p::Loc=u0(), n::Vec=vz(1); material)
```

Cuts a solid with a half-space defined by point `p` and normal `n`, keeping the
side in the direction of `n`.

## High-Level Modeling

These functions select the appropriate dimensionality variant automatically
based on the profile type.

### `extrusion`

```julia
extrusion(profile, h::Real, args...; kargs...)
extrusion(profile, v::Vec, args...; kargs...)
```

Extrudes a profile along a vector.

| Profile type | Result |
|---|---|
| `PointPath` / point shape | `extruded_point` (Shape1D) |
| `Path` / curve shape | `extruded_curve` (Shape2D) |
| `Region` / surface shape | `extruded_surface` (Shape3D) |

### `sweep`

```julia
sweep(path, profile, rotation=0, scale=1; material)
```

Sweeps a profile along a path.

| Profile type | Result |
|---|---|
| `PointPath` / point | `swept_point` (Shape1D) |
| `Path` / curve | `swept_curve` (Shape2D) |
| `Region` / surface | `swept_surface` (Shape3D) |

`rotation` (radians) twists the profile over the sweep length.  `scale`
linearly scales the profile from 1 at the start to `scale` at the end.

### `revolve`

```julia
revolve(profile, p::Loc=u0(), n::Vec=vz(1), start_angle::Real=0, amplitude::Real=2pi; material)
```

Revolves a profile around an axis defined by point `p` and direction `n`.

| Profile type | Result |
|---|---|
| point | `revolved_point` (Shape1D) |
| curve | `revolved_curve` (Shape2D) |
| surface | `revolved_surface` (Shape3D) |

### `loft`

```julia
loft(profiles::Shapes, rails=Shape[], ruled=false, closed=false; material)
loft_ruled(profiles)
```

Lofts through a sequence of cross-section profiles.

| Profile types | Result |
|---|---|
| all points | `loft_points` (Shape1D) |
| all curves | `loft_curves` (Shape2D) |
| all surfaces | `loft_surfaces` (Shape3D) |
| one curve + one point | `loft_curve_point` (Shape2D) |
| one surface + one point | `loft_surface_point` (Shape3D) |

When `ruled=true`, straight lines connect the profiles instead of smooth
interpolation.  When `closed=true`, the loft wraps around to connect the last
profile back to the first.

## Transforms

| Function | Signature | Description |
|----------|-----------|-------------|
| `move` | `move(shape, v::Vec)` | Translate shape by vector `v`. |
| `scale` | `scale(shape, s::Real, p::Loc=u0())` | Scale shape by factor `s` around point `p`. |
| `rotate` | `rotate(shape, angle::Real, p::Loc=u0(), v::Vec=vz(1))` | Rotate shape by `angle` (radians) around axis through `p` with direction `v`. |
| `mirror` | `mirror(shape, p::Loc=u0(), n::Vec=vx())` | Mirror shape across a plane through `p` with normal `n`. |
| `transform` | `transform(shape, xform::Loc)` | Apply a coordinate-system transformation. |
| `union_mirror` | `union_mirror(shape, p, v)` | Union of a shape with its mirror. |

## Bounding Box

```julia
bounding_box(shape::Shape) -> [min_loc, max_loc]
bounding_box(shapes::Shapes=Shape[]) -> [min_loc, max_loc]
```

Returns the axis-aligned bounding box as a two-element vector of locations.
Delegates to `backend_bounding_box` on the shape's backend.  Returns
`[u0(), u0()]` for an empty collection.

## Geometric Utilities (Geometry.jl)

### Area functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `triangle_area` | `triangle_area(a, b, c)` | Area of a triangle from its three side lengths (Heron's formula). |
| `circle_area` | `circle_area(r)` | Area of a circle. |
| `annulus_area` | `annulus_area(r_outer, r_inner)` | Area of an annular ring. |

### Polygon offset

```julia
offset_vertices(ps::Locs, d::Real, closed::Bool) -> Locs
```

Computes a parallel polygonal line offset by distance `d` from the original
vertices `ps`.  If `closed` is `true`, the polygon is treated as closed.

### Intersections

| Function | Signature | Description |
|----------|-----------|-------------|
| `segments_intersection` | `segments_intersection(p0, p1, p2, p3)` | Intersection point of segments `p0-p1` and `p2-p3`, or `nothing` if they do not intersect within their extents. |
| `lines_intersection` | `lines_intersection(p0, p1, p2, p3)` | Intersection point of infinite lines through `p0-p1` and `p2-p3`, or `nothing` if parallel. |
| `nearest_point_from_lines` | `nearest_point_from_lines(l0p0, l0p1, l1p0, l1p1)` | Midpoint of the shortest segment connecting two skew lines in 3D. |

### Collinearity

```julia
collinear_points(p0, pm, p1, tol=collinearity_tolerance()) -> Bool
```

Returns `true` when the triangle `p0-pm-p1` has area less than `tol`,
indicating that the three points are approximately collinear.  The default
tolerance is controlled by the `collinearity_tolerance` parameter (default
`1e-2` m²). See [Geometric tolerances](../concepts/parameters.md#geometric-tolerances)
for the full family of named tolerances.

### Circle from three points

```julia
circle_from_three_points(p0::Loc, p1::Loc, p2::Loc) -> (center, radius)
```

Computes the circumscribed circle of three 3D points.  Returns a tuple of the
center location and radius.

### Quad grids

```julia
quad_grid(quad_fn, points::Matrix, closed_u::Bool, closed_v::Bool)
```

Iterates over a matrix of points, calling `quad_fn(p00, p10, p11, p01)` for
each grid cell.  Handles closure in both u and v directions.

```julia
quad_grid_indexes(nu, nv, closed_u::Bool, closed_v::Bool) -> Vector{Vector{Int}}
```

Returns triangle-pair index lists (0-based) for a grid of size `nu x nv`,
suitable for constructing mesh face arrays.
