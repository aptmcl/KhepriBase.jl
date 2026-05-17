# Spline Curves and Surfaces

Khepri distinguishes interpolation splines from control-point splines. That
distinction matters for backend fidelity: an interpolation spline says "pass
through these points", while Bezier, B-spline, and NURBS geometry says "use this
control polygon, degree, knots, and optional weights".

## Curves

Use `spline_path` for fit-point/interpolating splines:

```julia
path = spline_path([u0(), x(2), xy(3, 1), xy(4, 0)])
stroke(path)
```

Use `bezier_path` when the points are Bezier control points:

```julia
path = bezier_path([
  u0(),
  xy(1, 0),
  xy(1, 1),
  xy(2, 1),
])
stroke(path)
```

Use `bspline_path` for a non-rational B-spline with explicit degree and knots:

```julia
path = bspline_path([
  u0(),
  xy(1, 1),
  xy(2, -0.5),
  xy(3, 0.5),
]; degree=3)
stroke(path)
```

Use `nurbs_path` when weights matter:

```julia
path = nurbs_path([
  u0(),
  xy(1, 1),
  xy(2, -0.5),
  xy(3, 0.5),
]; degree=3, weights=[1.0, 0.5, 1.4, 1.0])
stroke(path)
```

## Surfaces

Tensor-product surfaces use a matrix of control points. Rows are the `u`
direction and columns are the `v` direction.

```julia
pts = [xyz(i, j, 0.2sin(i + j)) for i in 0:3, j in 0:3]

bezier_surface(pts) |> fill
bspline_surface(pts; degree_u=3, degree_v=3) |> fill
nurbs_surface(
  pts;
  degree_u=3,
  degree_v=3,
  weights=[1.0 + 0.15*((i + j) % 2) for i in axes(pts, 1), j in axes(pts, 2)],
) |> fill
```

## Exactness

Backends that implement the exact hooks receive the control-point data directly.
Backends without those hooks still work, but KhepriBase samples curves to
polylines and surfaces to meshes.

```julia
supports_exact_nurbs_curves(top_backend())
supports_exact_nurbs_surfaces(top_backend())
```

Currently, Rhino, AutoCAD, and FreeCAD declare direct Bezier, B-spline, and NURBS
curve/surface support. Non-planar trimmed surfaces still fall back to sampled
geometry unless a backend adds a native trim implementation.
