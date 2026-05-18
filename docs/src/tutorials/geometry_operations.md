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
