# Geometry Operations

Geometry operations compute new geometric values from existing geometric
values. They are different from shape CSG: `intersection`, `union`, and
`subtraction` create lazy shape proxies that are realized in a backend, while
`intersections`, `section`, `project`, and `closest_points` return explicit
KhepriBase geometry and metadata immediately.

Use geometry operations when the result has to drive later modeling decisions:
where two walls meet, where a curve pierces a surface, which curve is produced
by a section plane, or where to split a path before creating BIM elements.

## Intersections

The main entry point is `intersections`:

```julia
a = line_path(xy(0, 0), xy(4, 0))
b = line_path(xy(2, -1), xy(2, 1))

r = intersections(a, b)
pts = intersection_points(r)
```

`r` is an `IntersectionSet`. It is iterable and contains zero or more
intersection elements. For this example, `pts` contains one location:
`xy(2, 0)`.

Intersections can be more than points. Overlapping segments produce a
`CurveIntersection`:

```julia
a = line_path(xy(0, 0), xy(4, 0))
b = line_path(xy(1, 0), xy(3, 0))

r = intersections(a, b)
curves = intersection_curves(r)
```

The first curve is the overlapping line segment from `xy(1, 0)` to `xy(3, 0)`.
This is why `intersections` returns a set of classified elements instead of a
single point or `nothing`.

## Result Elements

The result model is intentionally heterogeneous:

| Type | Meaning |
|------|---------|
| `IntersectionSet` | Full result of an intersection or section operation. |
| `PointIntersection` | A point result, with optional parameters on both operands. |
| `CurveIntersection` | A curve result, such as an overlap or surface section curve. |
| `RegionIntersection` | A filled region result. |
| `MultiRegion` | Several independent planar regions. |
| `GeometryCollection` | Heterogeneous result container for mixed-dimensional results. |
| `InfiniteLine` | Unbounded line result, for example from two intersecting planes. |

Convenience selectors extract the most common parts:

```julia
intersection_points(r)
intersection_curves(r)
intersection_regions(r)
```

Each intersection element has a `kind`, such as `:transversal`, `:tangent`,
`:endpoint`, `:overlap`, or `:section`. Elements also carry a `parameters`
named tuple when the operation can compute useful operand parameters.

## Sections

`section` is the curve-valued part of `intersections`. It is useful when the
operation is expected to produce a curve rather than isolated points:

```julia
xy_plane = plane_surface(u0())
vertical = plane_surface(loc_from_o_rot_x(u0(), pi/2))

r = section(xy_plane, vertical)
line = r[1].curve
```

The section of two non-parallel planes is an `InfiniteLine`, because the result
is not a finite path.

## Splitting

`split` uses intersection results to cut geometry into pieces. The local
implementation currently supports splitting `LinePath` values:

```julia
path = line_path(xy(0, 0), xy(4, 0))
cutter = line_path(xy(1, -1), xy(1, 1))

pieces = split(path, cutter)
```

`pieces` is a `PathSet` containing the two line segments before and after the
cut point. Backends may extend `split` for richer curve, surface, and Brep
types.

## Projection and Closest Points

Projection returns a `ProjectionResult`:

```julia
plane = plane_surface(u0())
p = xyz(1, 2, 3)

res = project(p, plane)
res.geometry   # xyz(1, 2, 0)
res.distance   # 3.0
```

`closest_points` returns the closest locations on two operands:

```julia
line = line_path(xy(0, 0), xy(4, 0))
res = closest_points(xy(1, 2), line)

res.first       # original point
res.second      # closest point on the line
res.distance    # shortest distance
```

The ordinary `distance(a, b)` function is extended for geometry operands and
uses `closest_points` when appropriate.

## Boolean Geometry

`boolean(op, a, b)` is the result-valued counterpart to lazy shape CSG. It is
intended for operations that should return explicit geometry, such as `Region`,
`MultiRegion`, or `GeometryCollection`:

```julia
boolean(:intersection, region_a, region_b)
boolean(:difference, region_a, region_b)
```

Only operand combinations with a local implementation or a backend override are
supported. This is deliberate: a boolean result may have multiple disconnected
components or holes, so callers should not expect a single `Path`.

## Local Coverage

KhepriBase includes a small analytic kernel for common exact/toleranced cases:

| Operation | Local support |
|-----------|---------------|
| line/line | point intersections and overlapping segments |
| line/circle, line/arc | point intersections and tangencies |
| circle/circle, arc/arc | point intersections, tangencies, and same-circle overlaps |
| composite paths | decomposes into primitive path pieces |
| line/plane | point intersection or overlapping line |
| circle/plane, arc/plane | point intersections or coplanar curve overlap |
| path/plane | decomposes paths into primitive pieces when possible |
| line/trimmed plane | point intersections filtered by the trim boundary |
| plane/plane | unbounded section line |
| region/region | planar polygon intersection, including concave contours and disconnected results |
| split | `LinePath` split at point intersections |
| project | point or line segment to `PlaneSurface` |
| closest points | point/point, point/line, line/line |

Other combinations throw `UnsupportedGeometryOperation` unless the selected
backend provides a native implementation.

## Backends and Methods

All operations accept the same option pattern:

```julia
intersections(a, b; tolerance=1e-8)
intersections(a, b; method=:local)
intersections(a, b; method=:backend, backend=top_backend())
```

The default `method=:auto` asks the selected backend first when it advertises
support, then falls back to KhepriBase's local analytic implementation.

Backend authors implement geometry operations with hooks such as
`b_intersections`, `b_section`, `b_project_geometry`, and `b_closest_points`.
See [Backend Operations Matrix](../reference/backend_operations.md#geometry-computation-hooks)
for the backend side of the contract.

The Rhino and FreeCAD backends can delegate curve/curve, curve/finite-surface,
and finite-surface/finite-surface intersections to their native kernels and map
straight sampled section curves back to `LinePath`. AutoCAD currently maps
curve/curve and curve/finite-surface point intersections through `IntersectWith`.

## Tolerances

Geometric operations are toleranced. The default tolerance is
`coincidence_tolerance()`, and overlapping results also use
`overlap_tolerance`:

```julia
intersections(a, b; tolerance=1e-7, overlap_tolerance=1e-6)
```

Use tighter tolerances for exact synthetic geometry and looser tolerances for
surveyed or imported data. See [Parameters](../concepts/parameters.md) for the
named tolerance parameters used across KhepriBase.
