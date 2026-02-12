# Paths

Paths are one-dimensional geometric objects that describe curves and outlines. They serve
as the backbone for wall centerlines, slab boundaries, extrusion profiles, sweep
trajectories, and visible strokes or fills. The path system lives in
`KhepriBase/src/Paths.jl` and builds on the [Coordinates](coordinates.md) system.

## Type Hierarchy

All paths descend from the abstract type `Path`, which branches into two families:

```
Path (abstract)
├── OpenPath (abstract) -- has distinct start and end
│   ├── ArcPath
│   ├── OpenPolygonalPath
│   ├── OpenSplinePath
│   └── OpenPathSequence
├── ClosedPath (abstract) -- forms a loop
│   ├── CircularPath
│   ├── EllipticPath
│   ├── RectangularPath
│   ├── ClosedPolygonalPath
│   ├── ClosedSplinePath
│   └── ClosedPathSequence
├── EmptyPath, PointPath
├── PathOps -- line/arc operations from a start location
├── PathSet -- collection of independent paths
└── Mesh -- vertices + face indices
```

The `OpenPath`/`ClosedPath` distinction matters for operations like `fill` (which
requires a closed path) and `map_division` (which excludes the endpoint for closed
paths to avoid duplication).

## Constructors

All constructors use lowercase names and return immutable path structs.

### Arcs, Circles, and Ellipses

```julia
arc_path(u0(), 5, 0, pi/2)      # center, radius, start angle, amplitude (OpenPath)
circular_path(xyz(0, 0, 0), 3)  # center, radius -- full circle (ClosedPath)
elliptic_path(u0(), 4, 2)       # center, semi-axis r1, semi-axis r2 (ClosedPath)
```

`arc_path` has domain `(0, amplitude)`. `circular_path` and `elliptic_path` have
domain `(0, 2pi)`.

### Rectangles

```julia
rectangular_path(xyz(0, 0, 0), 10, 5)  # corner, dx, dy
centered_rectangular_path(u0(), 10, 5)  # center, dx, dy (offsets corner by -dx/2, -dy/2)
```

### Polygonal Paths

```julia
open_polygonal_path([xyz(0,0,0), xyz(5,0,0), xyz(5,5,0)])    # open polyline
closed_polygonal_path([xyz(0,0,0), xyz(5,0,0), xyz(5,5,0)])  # closed polygon
polygonal_path([xyz(0,0,0), xyz(5,0,0), xyz(5,5,0), xyz(0,0,0)])  # auto-detect
```

`closed_polygonal_path` requires that first and last vertices are not coincident.
`polygonal_path` strips the duplicate endpoint automatically when detected.

### Spline Paths

```julia
open_spline_path([xyz(0,0,0), xyz(3,2,0), xyz(6,0,0), xyz(9,2,0)])
open_spline_path([xyz(0,0,0), xyz(3,2,0), xyz(6,0,0)], vx(), vx())  # with tangents
closed_spline_path([xyz(0,0,0), xyz(3,2,0), xyz(6,0,0), xyz(9,2,0)])
spline_path([xyz(0,0,0), xyz(3,2,0), xyz(6,0,0)])                    # auto-detect
```

Splines use parametric cubic interpolation (Dierckx) with domain `(0, 1)`.
`location_at` returns an oriented frame whose tangent and normal come from the spline
derivatives.

### Path Sequences and Sets

```julia
# Connected sequence (end of each sub-path meets start of next)
open_path_sequence(arc_path(u0(), 5, 0, pi/2),
  open_polygonal_path([xyz(0, 5, 0), xyz(-3, 5, 0)]))
closed_path_sequence(arc_path(u0(), 5, 0, pi),
  open_polygonal_path([xyz(-5, 0, 0), xyz(5, 0, 0)]))
path_sequence(...)  # auto-detects open vs closed

path_set(circular_path(u0(), 3), circular_path(xyz(10, 0, 0), 2))  # independent paths
open_path_ops(u0(), LineOp(vx(5)), ArcOp(2, 0, pi/2), LineOp(vy(3)))  # from operations
```

## Querying Paths

### Length, Domain, and Endpoints

```julia
path = open_polygonal_path([u0(), xyz(3, 0, 0), xyz(3, 4, 0)])
path_length(path)    # => 7.0
path_domain(path)    # => (0, 7.0)
path_start(path)     # => u0()
path_end(path)       # => xyz(3, 4, 0)
```

The domain varies by type: splines use `(0, 1)`, arcs use `(0, amplitude)`, circles
use `(0, 2pi)`, and polygonal/rectangular paths use `(0, path_length(path))`. For closed
paths, `path_end` returns the same location as `path_start`.

### Location at Parameter or Length

```julia
location_at(circular_path(u0(), 5), pi/2)                           # at parameter
location_at_length(open_polygonal_path([u0(), xyz(10, 0, 0)]), 3.0)  # at arc-length
```

Both return a full location with an oriented coordinate system. Paths also support
indexing as shorthand: `path[3.0]` is equivalent to `location_at_length(path, 3.0)`,
`path[begin]` starts at 0, and `path[end]` evaluates at `path_length(path)`.

### Vertices and Frames

```julia
path_vertices(path)   # control points (converts to polygonal approximation if needed)
path_frames(path)     # oriented frames at each vertex, useful for sweeps and placement
```

## Subpaths, Joining, and Transformations

```julia
subpath(path, 2, 8)              # extract between arc-lengths 2 and 8
join_paths(p1, p2)               # concatenate two open paths end-to-end

translate(path, vz(3))           # shift by vector
scale(path, 2.0)                 # scale around origin
scale(path, 0.5, xyz(5, 0, 0))  # scale around a point
reverse(path)                    # reverse traversal direction
```

All transformations return a new path; the original is not modified. `join_paths` avoids
duplicating the shared vertex when the endpoint of the first path coincides with the
start of the second.

## Stroke and Fill

`stroke` renders a path as a visible curve; `fill` fills a closed path as a surface.
Both accept optional `material` and `backend` keyword arguments.

```julia
stroke(open_polygonal_path([u0(), xyz(5, 3, 0), xyz(10, 0, 0)]))
fill(circular_path(u0(), 5), material=my_material)
```

These dispatch to `b_stroke` and `b_fill` in the active backend. See
[Shapes](../concepts/shapes.md) for details on the material system.

## Regions

A `Region` is a closed path with optional holes -- the standard input for slabs,
surfaces, and extrusions.

```julia
outer = rectangular_path(u0(), 20, 15)
hole1 = circular_path(xyz(5, 5, 0), 2)
hole2 = circular_path(xyz(15, 10, 0), 1.5)
r = region(outer, hole1, hole2)

outer_path(r)    # => the rectangular outer boundary
inner_paths(r)   # => [hole1, hole2]
```

Regions are used extensively in BIM operations (see
[Vertical Elements](../bim/vertical_elements.md) for walls and
[Horizontal Elements](../bim/horizontal_elements.md) for slabs).

## Profiles

Profiles are predefined centered paths used as cross-sections for beams and columns.

```julia
rectangular_profile(0.3, 0.5)                                       # centered rectangle
circular_profile(0.15)                                               # centered circle
i_profile(0.2, 0.3, web_thickness=0.01, flange_thickness=0.015)     # I-beam section
plus_profile(0.2, 0.2, thickness=0.04)                               # plus/cross section
top_aligned_rectangular_profile(0.3, 0.5)     # top edge at origin
bottom_aligned_rectangular_profile(0.3, 0.5)  # bottom edge at origin
```

## Using Paths with map_division

`map_division` evaluates a function at evenly spaced parameter values along a path.

```julia
# Place spheres around a circle
map_division(circular_path(u0(), 10), 12) do loc
  sphere(loc, 0.5)
end

# Place columns along a wall path
wall_path = open_polygonal_path([xyz(0,0,0), xyz(20,0,0), xyz(20,15,0)])
map_division(wall_path, 8) do frame
  cylinder(frame, 0.3, 4)
end
```

For open paths, `map_division` includes both endpoints. For closed paths, the last point
is excluded (since it would duplicate the first). Each returned location carries a full
coordinate frame, so objects placed at those locations inherit the path's orientation.

## Predicates

```julia
is_closed_path(circular_path(u0(), 5))              # => true
is_closed_path(open_polygonal_path([u0(), x()]))     # => false
is_smooth_path(circular_path(u0(), 5))               # => true (differentiable)
is_smooth_path(open_polygonal_path([u0(), x()]))     # => false (has corners)
```

Smooth paths (arcs, circles, splines) are continuously differentiable. Polygonal and
rectangular paths are not smooth because they have corners.

## Next Steps

- [Coordinates](coordinates.md) -- Locations, vectors, and coordinate systems used to define path vertices.
- [Shapes](../concepts/shapes.md) -- Geometric primitives that consume paths (extrusions, sweeps, surfaces).
- [Vertical Elements](../bim/vertical_elements.md) -- Walls defined by paths.
- [Horizontal Elements](../bim/horizontal_elements.md) -- Slabs and floors defined by regions.
