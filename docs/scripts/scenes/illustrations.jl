#=
Parameter-explainer illustrations.

Each scene shows the geometry produced by a Khepri shape
constructor with every parameter named on the figure itself -
`circle(p, r)` becomes a circle with a centred dot labelled `p`
and a radius marker labelled `r`.

Implemented entirely with KhepriBase's standard annotation
primitives - `label`, `radius_illustration`, `vector_illustration`,
`angle_illustration`, `arc_illustration`.  KhepriSVG implements
the corresponding `b_*` operations; KhepriBase routes annotations
through `save_shape!` so they participate in the regular render
pass.

For 3D shapes we draw the shape's most informative 2D silhouette
(a circle for `sphere`, an iso outline for `box`, a front view
with elliptical caps for `cylinder`, ...) and annotate the defining
parameters in the same style.  The doc page text makes the
3D nature explicit.
=#

# Annotation palette: warm orange that reads well on dark backgrounds.
const _ill_ann_mat = material(
  layer("doc-illustration-annotations", true,
        rgba(255/255, 168/255, 96/255, 1.0)))

# Run a build under the doc-illustration annotation material so every
# label / radius_illustration / ... inherits the same warm orange.
_with_ann(f) = with(default_annotation_material, _ill_ann_mat) do
  f()
end

# Iso-projection helper.  A 3D (x, y, z) becomes a 2D point in the
# SVG viewport.  Used for box-style silhouettes.
_iso(x, y, z) = xy(x - 0.5y, z + 0.32y)

# ==================================================================
# 2D shapes
# ==================================================================

register_scene(
  id = "illustration_circle",
  section = "reference",
  filename = "illustrations-circle.svg",
  backend = :svg,
  build = () -> _with_ann(() -> begin
    c = xy(0, 0); r = 2.0
    circle(c, r)
    label(c, "p")
    radius_illustration(c, r, "r")
  end),
)

register_scene(
  id = "illustration_arc",
  section = "reference",
  filename = "illustrations-arc.svg",
  backend = :svg,
  build = () -> _with_ann(() -> begin
    c = xy(0, 0); r = 2.0
    α0 = π/6; Δα = 2π/3
    arc(c, r, α0, Δα)
    label(c, "c")
    arc_illustration(c, r, α0, Δα, "r", "alpha", "delta_alpha")
  end),
)

register_scene(
  id = "illustration_rectangle",
  section = "reference",
  filename = "illustrations-rectangle.svg",
  backend = :svg,
  build = () -> _with_ann(() -> begin
    corner = xy(0, 0); dx = 4.0; dy = 2.5
    rectangle(corner, dx, dy)
    label(corner, "corner")
    vector_illustration(corner, 0,    dx, "dx")
    vector_illustration(corner, π/2,  dy, "dy")
  end),
)

register_scene(
  id = "illustration_regular_polygon",
  section = "reference",
  filename = "illustrations-regular_polygon.svg",
  backend = :svg,
  build = () -> _with_ann(() -> begin
    c = xy(0, 0); r = 2.0; ϕ = π/8; n = 6
    regular_polygon(n, c, r, ϕ, true)
    label(c, "center")
    angle_illustration(c, r, 0.0, ϕ, "radius", "0", "angle")
  end),
)

register_scene(
  id = "illustration_polygon",
  section = "reference",
  filename = "illustrations-polygon.svg",
  backend = :svg,
  build = () -> _with_ann(() -> begin
    pts = [xy(0, 0), xy(3, 0), xy(4, 2), xy(2, 3), xy(-0.5, 1.8)]
    polygon(pts)
    for (i, p) in enumerate(pts)
      label(p, "p$(i)")
    end
  end),
)

register_scene(
  id = "illustration_line",
  section = "reference",
  filename = "illustrations-line.svg",
  backend = :svg,
  build = () -> _with_ann(() -> begin
    pts = [xy(0, 0), xy(3, 1.5), xy(5.5, 0.5), xy(7, 2)]
    line(pts...)
    for (i, p) in enumerate(pts)
      label(p, "p$(i)")
    end
  end),
)

register_scene(
  id = "illustration_spline",
  section = "reference",
  filename = "illustrations-spline.svg",
  backend = :svg,
  build = () -> _with_ann(() -> begin
    pts = [xy(0, 0), xy(2, 2), xy(4, -0.5), xy(6, 1.5), xy(8, 0)]
    spline(pts)
    for (i, p) in enumerate(pts)
      label(p, "p$(i)")
    end
  end),
)

register_scene(
  id = "illustration_ellipse",
  section = "reference",
  filename = "illustrations-ellipse.svg",
  backend = :svg,
  build = () -> _with_ann(() -> begin
    c = xy(0, 0); rx = 3.0; ry = 1.7
    ellipse(c, rx, ry)
    label(c, "center")
    vector_illustration(c, 0,   rx, "radius_x")
    vector_illustration(c, π/2, ry, "radius_y")
  end),
)

register_scene(
  id = "illustration_surface_circle",
  section = "reference",
  filename = "illustrations-surface_circle.svg",
  backend = :svg,
  build = () -> _with_ann(() -> begin
    c = xy(0, 0); r = 2.0
    surface_circle(c, r,
      material=KhepriSVG.svg_option(
        "fill:rgba(255,168,96,0.18);stroke:none"))
    circle(c, r)
    label(c, "p")
    radius_illustration(c, r, "r")
  end),
)

register_scene(
  id = "illustration_surface_rectangle",
  section = "reference",
  filename = "illustrations-surface_rectangle.svg",
  backend = :svg,
  build = () -> _with_ann(() -> begin
    corner = xy(0, 0); dx = 4.0; dy = 2.5
    surface_rectangle(corner, dx, dy,
      material=KhepriSVG.svg_option(
        "fill:rgba(255,168,96,0.18);stroke:none"))
    rectangle(corner, dx, dy)
    label(corner, "corner")
    vector_illustration(corner, 0,    dx, "dx")
    vector_illustration(corner, π/2,  dy, "dy")
  end),
)

# ==================================================================
# 3D shapes - projected silhouettes with annotated parameters
# ==================================================================

register_scene(
  id = "illustration_sphere",
  section = "reference",
  filename = "illustrations-sphere.svg",
  backend = :svg,
  build = () -> _with_ann(() -> begin
    c = xy(0, 0); r = 2.0
    circle(c, r)
    label(c, "p")
    radius_illustration(c, r, "r")
  end),
)

register_scene(
  id = "illustration_box",
  section = "reference",
  filename = "illustrations-box.svg",
  backend = :svg,
  build = () -> _with_ann(() -> begin
    dx, dy, dz = 4.0, 3.0, 2.5
    polygon([_iso(0, 0, 0), _iso(dx, 0, 0),
             _iso(dx, 0, dz), _iso(0, 0, dz)])     # front face
    polygon([_iso(0, 0, dz), _iso(dx, 0, dz),
             _iso(dx, dy, dz), _iso(0, dy, dz)])   # top face
    polygon([_iso(dx, 0, 0), _iso(dx, dy, 0),
             _iso(dx, dy, dz), _iso(dx, 0, dz)])   # right face
    label(_iso(0, 0, 0), "corner")
    vector_illustration(_iso(0, 0, 0), 0,            dx, "dx")
    vector_illustration(_iso(0, 0, 0), π/2,          dz, "dz")
    vector_illustration(_iso(dx, 0, 0),
                        atan(0.32, -0.5),
                        dy * sqrt(0.32^2 + 0.5^2), "dy")
  end),
)

register_scene(
  id = "illustration_box_corners",
  section = "reference",
  filename = "illustrations-box_corners.svg",
  backend = :svg,
  build = () -> _with_ann(() -> begin
    dx, dy, dz = 4.0, 3.0, 2.5
    polygon([_iso(0, 0, 0), _iso(dx, 0, 0),
             _iso(dx, 0, dz), _iso(0, 0, dz)])
    polygon([_iso(0, 0, dz), _iso(dx, 0, dz),
             _iso(dx, dy, dz), _iso(0, dy, dz)])
    polygon([_iso(dx, 0, 0), _iso(dx, dy, 0),
             _iso(dx, dy, dz), _iso(dx, 0, dz)])
    label(_iso(0,  0,  0),  "p1")
    label(_iso(dx, dy, dz), "p2")
  end),
)

register_scene(
  id = "illustration_cylinder",
  section = "reference",
  filename = "illustrations-cylinder.svg",
  backend = :svg,
  build = () -> _with_ann(() -> begin
    r, h = 1.8, 4.0
    base = xy(0, 0)
    rectangle(base + vxy(-r, 0), 2r, h)
    arc(base + vxy(0, h), r, 0, π)        # top rim front
    arc(base, r, 0, π)                    # bottom rim front
    label(base, "p")
    vector_illustration(base, 0,   r, "r")
    vector_illustration(base, π/2, h, "h")
  end),
)

register_scene(
  id = "illustration_cone",
  section = "reference",
  filename = "illustrations-cone.svg",
  backend = :svg,
  build = () -> _with_ann(() -> begin
    r, h = 2.0, 4.0
    base = xy(0, 0)
    apex = xy(0, h)
    line(base + vxy(-r, 0), apex, base + vxy(r, 0))
    arc(base, r, 0, π)
    label(base, "p")
    vector_illustration(base, 0,   r, "r")
    vector_illustration(base, π/2, h, "h")
  end),
)

register_scene(
  id = "illustration_regular_pyramid",
  section = "reference",
  filename = "illustrations-regular_pyramid.svg",
  backend = :svg,
  build = () -> _with_ann(() -> begin
    rb, h = 2.0, 4.0
    base = xy(0, 0)
    apex = xy(0, h)
    p_left  = base + vxy(-rb, -0.4)
    p_right = base + vxy( rb, -0.4)
    p_back  = base + vxy(0,    0.4)
    line(p_left, apex, p_right)
    line(p_left, p_back, p_right)
    line(p_back, apex)
    label(base, "p")
    vector_illustration(base, 0,   rb, "rb")
    vector_illustration(base, π/2, h,  "h")
  end),
)

register_scene(
  id = "illustration_torus",
  section = "reference",
  filename = "illustrations-torus.svg",
  backend = :svg,
  build = () -> _with_ann(() -> begin
    R, r = 2.5, 0.6
    base = xy(0, 0)
    circle(base, R + r)
    circle(base, R - r)
    circle(base + vxy(R, 0),  r)
    circle(base + vxy(-R, 0), r)
    label(base, "p")
    vector_illustration(base, 0, R, "R")
    vector_illustration(base + vxy(R, 0), 0, r, "r")
  end),
)
