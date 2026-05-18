# Geometry Operations Tutorial

This tutorial shows how to use explicit geometry operations to drive later
modeling decisions. The examples use KhepriBase geometry values first; when a
backend is active, the same results can be drawn with `stroke`, `point`, and
other shape constructors.

## Finding a Construction Point

Start with two construction lines:

```julia
axis_x = line_path(xy(0, 0), xy(6, 0))
axis_y = line_path(xy(3, -2), xy(3, 2))

hits = intersections(axis_x, axis_y)
p = only(intersection_points(hits))
```

`p` is the geometric crossing point. It can be used directly in later
construction:

```julia
column_radius = 0.25
column_center = p

circle(column_center, column_radius)
```

The important distinction is that the point was computed from geometry, not
copied from an already-known coordinate. This lets the model remain robust when
the construction lines move.

## Splitting a Path at a Cutter

Intersections become more useful when they feed another operation. Here a beam
axis is split where it crosses a grid line:

```julia
beam_axis = line_path(xy(0, 0), xy(8, 0))
grid_line = line_path(xy(2.5, -1), xy(2.5, 1))

beam_pieces = split(beam_axis, grid_line)
```

`beam_pieces` is a `PathSet`. The first path ends at the cut point and the
second path starts at that same point:

```julia
first_piece = beam_pieces.paths[1]
second_piece = beam_pieces.paths[2]

path_end(first_piece)
path_start(second_piece)
```

For backend visualization:

```julia
stroke(first_piece)
stroke(second_piece)
point(path_start(second_piece))
```

This pattern is useful for placing joints, structural members, control points,
or annotations at computed locations.

When only one side is needed, trim the same axis instead of manually indexing
the split result:

```julia
left_piece = only(trim(beam_axis, grid_line; keep=:start).paths)
right_piece = only(trim(beam_axis, grid_line; keep=:end).paths)
```

## Intersecting Paths with a Boundary

Closed paths decompose into primitive pieces during intersection. A line across
a rectangular boundary returns both crossing points:

```julia
boundary = rectangular_path(xy(0, 0), 10, 6)
axis = line_path(xy(-2, 3), xy(12, 3))

pts = sort(intersection_points(intersections(boundary, axis)); by=p -> p.x)
entry = pts[1]
exit = pts[2]
```

The computed entry and exit points can define a path clipped to the boundary:

```julia
inside_axis = line_path(entry, exit)
stroke(inside_axis)
```

This is the same kind of operation needed for slabs, rooms, facade grids, and
wall-layout helpers: compute where a guiding curve meets a boundary, then use
the result as authored geometry for the next step.

## Intersecting Filled Regions

For simple planar footprints, result-valued booleans return geometry that can
drive later modeling. This example computes the overlap between two rectangular
zones:

```julia
room = region(rectangular_path(xy(0, 0), 6, 4))
setback = region(rectangular_path(xy(2, 1), 5, 5))

usable = boolean(:intersection, room, setback)
```

`usable` is a `Region` when the overlap is connected. A disjoint result would
be a `MultiRegion`. That distinction matters for modeling operations such as a
slab contour, a ceiling panel layout, or a room area calculation.

The local implementation is not limited to rectangles. Concave polygonal
regions are decomposed internally and stitched back into boundary loops:

```julia
room = region(closed_polygonal_path([
  xy(0, 0), xy(4, 0), xy(4, 1), xy(1, 1), xy(1, 4), xy(0, 4)
]))
zone = region(rectangular_path(xy(0.5, 0.5), 2.5, 2.5))

overlap = boolean(:intersection, room, zone)
```

Regions with holes keep that topology when the hole is fully contained in the
overlap:

```julia
room = region(rectangular_path(xy(0, 0), 6, 6),
              rectangular_path(xy(2, 2), 2, 2))
zone = region(rectangular_path(xy(1, 1), 4, 4))

usable_ring = boolean(:intersection, room, zone)
```

`usable_ring` is a `Region` whose outer path is the clipped square and whose
inner paths still describe the void.

## Circle Chords and Tangencies

Circle intersections are classified. A secant line produces two point
intersections:

```julia
c = circular_path(xy(0, 0), 3)
secant = line_path(xy(-4, 1), xy(4, 1))

pts = sort(intersection_points(intersections(secant, c)); by=p -> p.x)
chord = line_path(pts[1], pts[2])
stroke(chord)
```

A tangent line produces one point with `kind == :tangent`:

```julia
tangent = line_path(xy(-4, 3), xy(4, 3))
r = intersections(tangent, c)

hit = only(r)
hit.kind     # :tangent
hit.point    # xy(0, 3)
```

Code that depends on topological meaning should inspect `kind`, not only the
number of returned points.

## Plane Sections

Sections are the curve part of an intersection. Two non-parallel planes produce
an unbounded line:

```julia
floor_plane = plane_surface(u0())
wall_plane = plane_surface(loc_from_o_rot_x(u0(), pi/2))

r = section(floor_plane, wall_plane)
axis = only(intersection_curves(r))
```

`axis` is an `InfiniteLine`. That is intentional: the true section of two
planes is not a finite path. Use a finite path only when the model supplies
limits, such as a wall length, slab boundary, or trimmed surface.

## Projection to a Work Plane

Projection is useful when imported or generated points are not exactly on the
working plane:

```julia
work_plane = plane_surface(u0())
survey_point = xyz(4, 2, 0.013)

projection = project(survey_point, work_plane)
clean_point = projection.geometry
offset_error = projection.distance
```

The model can decide what to do with the result:

```julia
if offset_error < 0.02
  point(clean_point)
else
  error("Survey point is too far from the work plane")
end
```

The same pattern works for line segments:

```julia
skew_line = line_path(xyz(0, 0, 0.02), xyz(4, 0, -0.01))
projected = project(skew_line, work_plane).geometry
stroke(projected)
```

Closed paths can be projected too. Smooth curves are sampled when the local
kernel cannot preserve their exact analytic type:

```julia
tilted_outline = rectangular_path(xyz(0, 0, 0.15), 4, 2)
flat_outline = project(tilted_outline, work_plane).geometry
```

## Closest Points and Distances

`closest_points` returns both closest locations and the distance between them:

```julia
path = line_path(xy(0, 0), xy(5, 0))
fixture = xy(3, 1.2)

res = closest_points(fixture, path)
attachment_point = res.second
clearance = res.distance
```

The shorter form uses the same machinery:

```julia
clearance = distance(fixture, path)
```

This is useful for snapping, clearance checks, and placing dependent objects on
nearby guide geometry.

The same pattern is useful in higher-level modeling code. WallGraph, for
example, computes miter and arc-wall junction points from path intersections
instead of keeping a separate hand-written solver for every junction type. That
keeps wall contours tied to the same geometric operations used elsewhere in the
model.

## Classification Checks

Classification is useful before committing to a modeling operation:

```julia
footprint = region(rectangular_path(xy(0, 0), 8, 6))
candidate = xy(3, 2)

if contains_geometry(footprint, candidate)
  point(candidate)
end
```

Use the symbolic result when boundary cases matter:

```julia
classify_geometry(footprint, xy(0, 3))  # :boundary
```

## Backend-Native Queries

When a backend is active, `method=:auto` asks that backend first for operations
it advertises. Use `backend_geometry_mapping` to inspect the current backend's
direct mappings:

```julia
report = backend_geometry_mapping(top_backend())
report.operations.project_point_surface
report.operations.closest_points_path_path
report.operations.classify_region_point
```

For operations where the native kernel matters, request it explicitly:

```julia
b = top_backend()
face = region(rectangular_path(u0(), 5, 3))
p = xyz(1, 1, 2)

projection = project(p, face; method=:backend, backend=b)
closest = closest_points(p, face; method=:backend, backend=b)
where = classify_geometry(face, xy(1, 1); method=:backend, backend=b)
```

Rhino and FreeCAD map these calls to their native curve and surface kernels for
the supported operand pairs. If the backend does not advertise the requested
combination, KhepriBase throws `UnsupportedGeometryOperation` for
`method=:backend` or falls back to the local kernel for `method=:auto`.

Backend-native work also matters in the other direction. Shape queries such as
`all_shapes`, layer searches, and selection import backend references through
`get_shape` and related mapping hooks. Exact concepts such as lines, circles,
arcs, ellipses, and splines should be reconstructed as Khepri geometry whenever
the backend can report enough data; otherwise the import path must say that the
result is approximate or opaque.

## Choosing Local or Backend Computation

By default, KhepriBase uses `method=:auto`: it asks the selected backend when
that backend advertises a native implementation, otherwise it tries the local
analytic kernel.

```julia
intersections(a, b; method=:auto)
```

Use `method=:local` when reproducibility matters more than backend-specific
behavior:

```julia
intersections(a, b; method=:local)
```

Use `method=:backend` when the operation must be computed by the backend's
native geometry kernel:

```julia
intersections(a, b; method=:backend, backend=top_backend())
```

If neither KhepriBase nor the backend supports the operand combination, the
operation throws `UnsupportedGeometryOperation`. That failure is preferable to
silently returning incomplete geometry.
