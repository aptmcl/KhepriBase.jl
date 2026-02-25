# Shapes & Geometry

Shapes are the central building blocks in Khepri. Every geometric object -- a sphere,
a line, a surface -- is represented as a lightweight Julia struct called a **shape proxy**.
Proxies store parameters but do not create geometry until they are **realized** in a
backend.

## Shape Proxies

When you write:

```julia
s = sphere(xyz(0, 0, 0), 5)
```

Khepri creates a `Sphere` struct holding the center and radius. No geometry exists yet.
Only when the shape is sent to a backend (explicitly or via eager realization) does
Khepri call `b_sphere(backend, center, radius, material_ref)` to produce actual
geometry in the target application.

This lazy, two-phase approach gives several benefits:

- The same proxy can be realized in multiple backends simultaneously.
- Parameters are inspectable and modifiable before realization.
- Backend-specific details remain isolated in `b_*` implementations.

## The `@defshape` Macro

All standard shapes are defined with the `@defshape` macro, which is a thin wrapper
around `@defproxy`. For each shape definition, `@defshape` generates:

| Generated name | Purpose | Example for `sphere` |
|----------------|---------|----------------------|
| `sphere(...)` | Constructor function | `sphere(xyz(0,0,0), 5)` |
| `Sphere` | Struct type | Used for dispatch |
| `is_sphere(v)` | Predicate | `is_sphere(s) # true` |
| `sphere_center(s)` | Field selector | `sphere_center(s) # xyz(0,0,0)` |
| `sphere_radius(s)` | Field selector | `sphere_radius(s) # 5` |
| `realize(b, s)` | Backend realization | Calls `b_sphere(b, ...)` |
| `material` field | Automatic | Default depends on dimensionality |

The `material` field is appended automatically based on the shape's dimensionality
(see [Default Materials](#default-materials-by-dimensionality) below).

## Shape Dimensionality

Shapes are parameterized by their topological dimension through the abstract type
`Shape{D}`, with convenience aliases for each dimension:

```
Proxy (abstract)
├── Shape{D} (abstract, parameterized by dimension D)
│   ├── Shape0D  (= Shape{0}) — points, text, blocks
│   ├── Shape1D  (= Shape{1}) — lines, circles, arcs, splines, polygons, rectangles, ellipses
│   ├── Shape2D  (= Shape{2}) — surfaces: circle, polygon, rectangle, grid, mesh
│   ├── Shape3D  (= Shape{3}) — solids: sphere, box, cylinder, cone, torus, cuboid, prism, pyramid
│   │   └── BIMShape         — walls, slabs, beams, columns (see BIM docs)
│   └── (dimension predicates: is_curve, is_surface, is_solid)
├── Annotation (abstract)    — dimensions, labels
└── UniqueProxy (abstract)   — layers, materials, families
```

The predicates `is_curve(s)`, `is_surface(s)`, and `is_solid(s)` return `true` for
`Shape1D`, `Shape2D`, and `Shape3D` respectively.

## Shape Catalog

### Shape0D -- Points and Text

```julia
# A point at a location (default: origin)
point(xyz(3, 4, 0))

# Text placed at a corner with a given height
text("Hello", xyz(0, 0, 0), 2.0)
```

| Constructor | Parameters | Defaults |
|-------------|-----------|----------|
| `point(position)` | `position::Loc` | `u0()` |
| `text(str, corner, height)` | `str::String`, `corner::Loc`, `height::Real` | `""`, `u0()`, `1` |

### Shape1D -- Curves

Shape1D constructors produce open or closed curves. Many accept either explicit
parameter lists or variadic location arguments.

```julia
# Open polyline through three points
line(xyz(0,0,0), xyz(5,0,0), xyz(5,5,0))

# Closed polygon
polygon(xyz(0,0,0), xyz(5,0,0), xyz(2.5,5,0))

# Circle with center and radius
circle(xyz(0,0,0), 3)

# Circular arc: center, radius, start angle, amplitude
arc(xyz(0,0,0), 5, 0, pi/2)

# Smooth spline through points
spline(xyz(0,0,0), xyz(2,3,0), xyz(5,1,0), xyz(7,4,0))

# Rectangle from corner with width and height
rectangle(xyz(0,0,0), 10, 5)

# Regular polygon (e.g., hexagon)
regular_polygon(6, xyz(0,0,0), 4)
```

| Constructor | Key Parameters | Defaults |
|-------------|---------------|----------|
| `line(vertices)` | `vertices::Locs` | `[u0(), ux()]` |
| `closed_line(vertices)` | `vertices::Locs` | `[u0(), ux(), uy()]` |
| `polygon(vertices)` | `vertices::Locs` | `[u0(), ux(), uy()]` |
| `spline(points, v0, v1)` | `points::Locs`, `v0`, `v1` | tangent: `false` |
| `closed_spline(points)` | `points::Locs` | `[u0(), ux(), uy()]` |
| `circle(center, radius)` | `center::Loc`, `radius::Real` | `u0()`, `1` |
| `arc(center, radius, start_angle, amplitude)` | all `Real` except center | `u0()`, `1`, `0`, `pi` |
| `ellipse(center, radius_x, radius_y)` | `center::Loc`, radii `Real` | `u0()`, `1`, `1` |
| `elliptic_arc(center, radius_x, radius_y, start_angle, amplitude)` | as above | `u0()`, `1`, `1`, `0`, `pi` |
| `regular_polygon(edges, center, radius, angle, inscribed)` | `edges::Integer` | `3`, `u0()`, `1`, `0`, `true` |
| `rectangle(corner, dx, dy)` | `corner::Loc`, `dx`, `dy` | `u0()`, `1`, `1` |

Locations are created with coordinate constructors described in
[Coordinates & Vectors](../getting_started/coordinates.md).

### Shape2D -- Surfaces

Surface shapes are filled regions. Their names mirror the corresponding 1D shapes
with a `surface_` prefix.

```julia
# Filled circle
surface_circle(xyz(0,0,0), 3)

# Annular ring (inner radius 1, outer radius 3)
surface_ring(xyz(0,0,0), 1, 3)

# Filled polygon (variadic or vector form)
surface_polygon(xyz(0,0,0), xyz(5,0,0), xyz(2.5,5,0))

# Filled rectangle
surface_rectangle(xyz(0,0,0), 10, 5)

# Surface from one or more boundary curves
surface([circle(xyz(0,0,0), 5), circle(xyz(0,0,0), 2)])  # ring with hole

# Grid surface from a matrix of points
pts = [xyz(u, v, sin(u)*cos(v)) for u in 0:0.5:pi, v in 0:0.5:2pi]
surface_grid(pts, false, true)  # closed in v direction

# Mesh from vertices and face index lists (0-based indices)
surface_mesh([xyz(0,0,0), xyz(1,0,0), xyz(0,1,0)], [[0,1,2]])
```

| Constructor | Key Parameters | Defaults |
|-------------|---------------|----------|
| `surface_circle(center, radius)` | `center::Loc`, `radius::Real` | `u0()`, `1` |
| `surface_ring(center, inner_radius, outer_radius)` | `center::Loc`, radii `Real` | `u0()`, `0.5`, `1` |
| `surface_arc(center, radius, start_angle, amplitude)` | as above | `u0()`, `1`, `0`, `pi` |
| `surface_ellipse(center, radius_x, radius_y)` | `center::Loc`, radii | `u0()`, `1`, `1` |
| `surface_elliptic_arc(center, radius_x, radius_y, start_angle, amplitude)` | as above | `u0()`, `1`, `1`, `0`, `pi` |
| `surface_polygon(vertices)` | `vertices::Locs` | `[u0(), ux(), uy()]` |
| `surface_regular_polygon(edges, center, radius, angle, inscribed)` | `edges::Integer` | `3`, `u0()`, `1`, `0`, `true` |
| `surface_rectangle(corner, dx, dy)` | `corner::Loc`, `dx`, `dy` | `u0()`, `1`, `1` |
| `surface(frontier)` | `frontier::Shapes1D` | `[circle()]` |
| `surface_grid(points, closed_u, closed_v, smooth_u, smooth_v)` | `points::Matrix{Loc}`, booleans | `false`, `false`, `true`, `true` |
| `surface_mesh(vertices, faces)` | `vertices::Locs`, `faces::Vector{Vector{Int}}` | `[u0(), ux(), uy()]`, `[[0,1,2]]` |

### Shape3D -- Solids

Solid shapes produce closed volumes. Several constructors accept either a height
value or a top location, using `position_and_height` internally to compute the
oriented center and scalar height.

```julia
# Sphere
sphere(xyz(0, 0, 0), 5)

# Axis-aligned box from corner with dimensions
box(xyz(0, 0, 0), 10, 5, 3)

# Box from two opposite corners
box(xyz(0, 0, 0), xyz(10, 5, 3))

# Cylinder from base center, radius, and height
cylinder(xyz(0, 0, 0), 2, 8)

# Cylinder from base center to top center
cylinder(xyz(0, 0, 0), 2, xyz(0, 0, 8))

# Cone and cone frustum
cone(xyz(0, 0, 0), 3, 6)
cone_frustum(xyz(0, 0, 0), 3, 6, 1)

# Torus (center, major radius, minor radius)
torus(xyz(0, 0, 0), 5, 1)

# Prism: extrude a polygon along a vector
prism([xyz(0,0,0), xyz(3,0,0), xyz(1.5,2,0)], vz(4))

# Regular prism (e.g., hexagonal prism)
regular_prism(6, xyz(0,0,0), 2, 0, 5)

# Pyramid from base vertices to apex
pyramid([xyz(0,0,0), xyz(4,0,0), xyz(4,4,0), xyz(0,4,0)], xyz(2,2,6))

# Cuboid from eight corner points
cuboid(xyz(0,0,0), xyz(1,0,0), xyz(1,1,0), xyz(0,1,0),
       xyz(0,0,1), xyz(1,0,1), xyz(1,1,1), xyz(0,1,1))

# Isosurface from an implicit function and bounding box
isosurface(p -> sph_rho(p) - 3, [xyz(-4,-4,-4), xyz(4,4,4)])
```

| Constructor | Key Parameters | Defaults |
|-------------|---------------|----------|
| `sphere(center, radius)` | `center::Loc`, `radius::Real` | `u0()`, `1` |
| `box(c, dx, dy, dz)` | `c::Loc`, dimensions `Real` | `u0()`, `1`, `dx`, `dy` |
| `cylinder(cb, r, h)` | `cb::Loc`, `r::Real`, `h::Real` | `u0()`, `1`, `1` |
| `cone(cb, r, h)` | `cb::Loc`, `r::Real`, `h::Real` | `u0()`, `1`, `1` |
| `cone_frustum(cb, rb, h, rt)` | base/top radii, height | `u0()`, `1`, `1`, `1` |
| `torus(center, re, ri)` | major/minor radii | `u0()`, `1`, `0.5` |
| `cuboid(b0, b1, b2, b3, t0, t1, t2, t3)` | eight `Loc` corners | unit cube |
| `prism(bs, v)` | `bs::Locs`, `v::Vec` | base vertices, `vz(1)` |
| `regular_prism(edges, cb, r, angle, h, inscribed)` | `edges::Integer` | `3`, `u0()`, `1`, `0`, `1`, `true` |
| `regular_pyramid(edges, cb, rb, angle, h, inscribed)` | `edges::Integer` | `3`, `u0()`, `1`, `0`, `1`, `true` |
| `regular_pyramid_frustum(edges, cb, rb, angle, h, rt, inscribed)` | as above + `rt` | `4`, ..., `1` |
| `pyramid(bs, t)` | `bs::Locs`, `t::Loc` | base vertices, `uz()` |
| `pyramid_frustum(bs, ts)` | bottom/top `Locs` | base/top vertices |
| `right_cuboid(cb, width, height, h)` | `cb::Loc`, dimensions | `u0()`, `1`, `1`, `1` |
| `isosurface(frep, bounding_box)` | `frep::Function`, `bounding_box::Locs` | `sph_rho`, unit cube |

## Default Materials by Dimensionality

The `@defshape` macro automatically appends a `material` field to every shape. The
default value depends on the shape's dimensionality:

| Dimension | Default Material Parameter | Predefined Value |
|-----------|---------------------------|------------------|
| `Shape0D` | `default_point_material()` | `material_point` |
| `Shape1D` | `default_curve_material()` | `material_curve` |
| `Shape2D` | `default_surface_material()` | `material_surface` |
| `Shape3D` | `default_material()` | `material_basic` |
| `Annotation` | `default_annotation_material()` | blue-tinted material |

You can override the default on any individual shape:

```julia
red = standard_material(base_color=rgba(1, 0, 0, 1))
sphere(xyz(0, 0, 0), 5, material=red)
```

Or change the default for all subsequent shapes of a given dimension:

```julia
with(default_material, my_custom_material) do
  # all Shape3D created here use my_custom_material
  box(xyz(0, 0, 0), 1, 1, 1)
  sphere(xyz(5, 0, 0), 2)
end
```

See [Levels & Families](levels_and_families.md) for more on the material system.

## CSG Operations

Khepri supports Constructive Solid Geometry through `union`, `subtraction`, and
`intersection`. These operations work on both 2D surfaces and 3D solids, producing
the appropriate compound shape type.

```julia
# Boolean union of two solids
s = union(sphere(xyz(0, 0, 0), 3), box(xyz(1, 1, 1), 4, 4, 4))

# Subtract a cylinder from a box
s = subtraction(box(xyz(0, 0, 0), 10, 10, 10),
                cylinder(xyz(5, 5, 0), 3, 10))

# Intersection of two spheres
s = intersection(sphere(xyz(0, 0, 0), 5), sphere(xyz(3, 0, 0), 5))

# Chained subtraction (removes multiple shapes from the first)
s = subtraction(box(xyz(0,0,0), 10, 10, 10),
                cylinder(xyz(5, 5, 0), 2, 10),
                sphere(xyz(5, 5, 5), 3))

# 2D boolean operations work the same way
s = subtraction(surface_circle(u0(), 5), surface_rectangle(xy(-1, -1), 2, 2))
```

| Operation | 2D Result Type | 3D Result Type |
|-----------|---------------|---------------|
| `union(a, b)` | `UnitedSurfaces` | `UnitedSolids` |
| `subtraction(a, b)` | `SubtractedSurfaces` | `SubtractedSolids` |
| `intersection(a, b)` | `IntersectedSurfaces` | `IntersectedSolids` |

The `slice(shape, p, n)` operation cuts a solid with a plane defined by point `p`
and normal vector `n`:

```julia
# Cut a sphere with the XY plane
half = slice(sphere(xyz(0, 0, 0), 5), u0(), vz(1))
```

## High-Level Modeling Operations

These operations create shapes by transforming profiles along paths or axes.

### Extrusion

Extrude a profile along a vector or by a height. The result dimension depends on
the profile: a point produces a curve, a curve produces a surface, and a surface
produces a solid.

```julia
# Extrude a circular path upward by 10 units
extrusion(circular_path(u0(), 3), 10)

# Extrude a surface region along a vector
extrusion(region(rectangular_path(u0(), 5, 3)), vxyz(0, 0, 8))
```

### Sweep

Sweep a profile along a path. Like extrusion, the result dimension depends on the
profile.

```julia
# Sweep a small circle along a larger circular path (creates a torus-like shape)
sweep(circular_path(u0(), 10), circular_path(u0(), 1))
```

### Revolve

Revolve a profile around an axis. Parameters `p` and `n` define the axis of
revolution (point and direction), and `start_angle`/`amplitude` control the sweep
range.

```julia
# Revolve a rectangle to create a cylinder-like shape
revolve(line(xyz(3, 0, 0), xyz(3, 0, 5)),
        p=u0(), n=vz(1), start_angle=0, amplitude=2pi)
```

### Loft

Loft between multiple cross-section profiles. Profiles must all be the same
dimension (all points, all curves, or all surfaces). Options include `rails` for
guide curves, `ruled` for straight interpolation, and `closed` for closing the
loft back to the first profile.

```julia
# Loft between two circles at different heights
loft([circle(xyz(0, 0, 0), 5), circle(xyz(0, 0, 10), 3)])

# Ruled loft (straight edges between profiles)
loft([circle(xyz(0, 0, 0), 5), circle(xyz(0, 0, 10), 3)], ruled=true)

# Loft from a curve to a point (cone-like shape)
loft([circle(xyz(0, 0, 0), 5), point(xyz(0, 0, 10))])
```

## Shape Management

### Collecting Shapes

The `collecting_shapes` function captures all shapes created within its block,
returning them as a vector:

```julia
shapes = collecting_shapes() do
  sphere(xyz(0, 0, 0), 1)
  box(xyz(3, 0, 0), 2, 2, 2)
  cylinder(xyz(6, 0, 0), 1, 4)
end
# shapes is a Vector{Shape} with three elements
```

### Deleting Shapes

```julia
s = sphere(xyz(0, 0, 0), 5)

# Delete a single shape
delete_shape(s)

# Delete multiple shapes
delete_shapes([s1, s2, s3])

# Delete all shapes (but keep materials, layers, etc.)
delete_all_shapes()

# Delete all annotations only
delete_all_annotations()

# Delete everything: shapes, materials, layers
delete_all()
```

## Transformations

Shapes can be moved, rotated, scaled, mirrored, and arbitrarily transformed. These
produce new proxy shapes that wrap the original.

```julia
s = sphere(xyz(0, 0, 0), 3)

# Translate by a vector
move(s, vxyz(10, 0, 0))

# Rotate around an axis
rotate(s, pi/4, u0(), vz(1))

# Scale relative to a point
scale(s, 2.0, u0())

# Mirror across a plane (point + normal)
mirror(s, u0(), vx(1))

# Union of a shape with its mirror
union_mirror(s, u0(), vx(1))
```

## See Also

- [Coordinates & Vectors](../getting_started/coordinates.md) -- location and vector constructors used in shape parameters
- [Paths](../getting_started/paths.md) -- paths used as profiles in extrusions, sweeps, and surface boundaries
- [Levels & Families](levels_and_families.md) -- BIM-level abstractions built on top of shapes
- [Shapes & Geometry Reference](../reference/shapes_geometry.md) -- complete API reference tables
