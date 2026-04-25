#=
Parameter-explainer illustrations.

Each scene shows the geometry produced by a Khepri shape
constructor with every parameter named on the figure itself —
`circle(p, r)` becomes a circle with a centred dot labelled `p`
and a radius marker labelled `r`.  KhepriBase ships annotation
primitives (`label`, `radius_illustration`, …), but the SVG
backend funnels them into a manual-commit transaction that
`render_view` never commits, so they get dropped.  We sidestep
that by composing the dimension marks from ordinary
`circle`/`line`/`text` primitives — they realise normally.

For 3D shapes we draw the shape's most informative 2D silhouette
(a circle for `sphere`, an iso outline for `box`, a front view
with elliptical caps for `cylinder`, …) and annotate the defining
parameters in the same style.  The doc page text makes the
3D nature explicit.
=#

# Annotation palette, light enough to read on dark backgrounds.
const _ill_dim_mat   = KhepriSVG.svg_option(
  "fill:none;stroke:rgb(255,168,96);stroke-width:0.04")
const _ill_label_mat = KhepriSVG.svg_option(
  "fill:rgb(255,168,96);stroke:none")
const _ill_dot_mat   = KhepriSVG.svg_option(
  "fill:rgb(255,168,96);stroke:none")

# Tiny dot for naming a location.
_ill_dot(p; r=0.06) = surface_circle(p, r, material=_ill_dot_mat)

# Place a label at `p`, offset by `(dx, dy)` in world units, with
# height `h`.
_ill_label(str, p; dx=0.18, dy=0.18, h=0.30) =
  text(str, p + vxy(dx, dy), h, material=_ill_label_mat)

# Draw a dimension segment from `a` to `b` with a label at the
# midpoint.  Adds short tick marks at each end.
_ill_dim(a, b, str; tick=0.12, h=0.30, label_offset=0.20) =
  let dir = b - a,
      dlen = pol_rho(dir),
      θ    = pol_phi(dir),
      perp = vpol(tick, θ + π/2)
    line([a, b], material=_ill_dim_mat)
    line([a + perp, a - perp], material=_ill_dim_mat)
    line([b + perp, b - perp], material=_ill_dim_mat)
    let mid    = a + 0.5 * dir,
        normal = vpol(label_offset, θ + π/2)
      text(str, mid + normal, h, material=_ill_label_mat)
    end
  end

# Angle-illustration helper.  Draw an arc inside a circle and
# label it.
_ill_angle(c, r, α0, Δα, str; h=0.28) =
  let mid_θ = α0 + Δα/2,
      label_p = c + vpol(0.55r, mid_θ)
    arc(c, 0.4r, α0, Δα, material=_ill_dim_mat)
    line([c, c + vpol(r, α0)], material=_ill_dim_mat)
    line([c, c + vpol(r, α0 + Δα)], material=_ill_dim_mat)
    text(str, label_p, h, material=_ill_label_mat)
  end

# ==================================================================
# 2D shapes
# ==================================================================

register_scene(
  id = "illustration_circle",
  section = "reference",
  filename = "illustrations-circle.svg",
  backend = :svg,
  build = () -> begin
    c = xy(0, 0)
    r = 2.0
    circle(c, r)
    _ill_dot(c)
    _ill_label("p", c; dx=0.15, dy=0.20)
    _ill_dim(c, c + vxy(r, 0), "r")
  end,
)

register_scene(
  id = "illustration_arc",
  section = "reference",
  filename = "illustrations-arc.svg",
  backend = :svg,
  build = () -> begin
    c  = xy(0, 0)
    r  = 2.0
    α0 = π/6
    Δα = 2π/3
    arc(c, r, α0, Δα)
    _ill_dot(c)
    _ill_label("c", c; dx=0.15, dy=0.20)
    _ill_dim(c, c + vpol(r, α0), "r")
    _ill_angle(c, r, 0, α0, "α")
    _ill_angle(c, r, α0, Δα, "Δα")
  end,
)

register_scene(
  id = "illustration_rectangle",
  section = "reference",
  filename = "illustrations-rectangle.svg",
  backend = :svg,
  build = () -> begin
    corner = xy(0, 0)
    dx, dy = 4.0, 2.5
    rectangle(corner, dx, dy)
    _ill_dot(corner)
    _ill_label("corner", corner; dx=-1.4, dy=-0.4)
    _ill_dim(corner + vxy(0, -0.5),
             corner + vxy(dx, -0.5), "dx")
    _ill_dim(corner + vxy(-0.5, 0),
             corner + vxy(-0.5, dy), "dy"; label_offset=-0.4)
  end,
)

register_scene(
  id = "illustration_regular_polygon",
  section = "reference",
  filename = "illustrations-regular_polygon.svg",
  backend = :svg,
  build = () -> begin
    c = xy(0, 0)
    r = 2.0
    ϕ = π/8
    n = 6
    regular_polygon(n, c, r, ϕ, true)
    _ill_dot(c)
    _ill_label("center", c; dx=-1.4, dy=0.20)
    _ill_dim(c, c + vpol(r, ϕ), "radius")
    _ill_angle(c, r, 0, ϕ, "ϕ")
    text("n = $(n)", c + vxy(-2.5, -1.4), 0.30,
         material=_ill_label_mat)
  end,
)

register_scene(
  id = "illustration_polygon",
  section = "reference",
  filename = "illustrations-polygon.svg",
  backend = :svg,
  build = () -> begin
    pts = [xy(0, 0), xy(3, 0), xy(4, 2), xy(2, 3), xy(-0.5, 1.8)]
    polygon(pts)
    for (i, p) in enumerate(pts)
      _ill_dot(p)
      _ill_label("p$(i)", p; dx=0.15, dy=0.20)
    end
  end,
)

register_scene(
  id = "illustration_line",
  section = "reference",
  filename = "illustrations-line.svg",
  backend = :svg,
  build = () -> begin
    pts = [xy(0, 0), xy(3, 1.5), xy(5.5, 0.5), xy(7, 2)]
    line(pts...)
    for (i, p) in enumerate(pts)
      _ill_dot(p)
      _ill_label("p$(i)", p; dx=0.15, dy=0.25)
    end
  end,
)

register_scene(
  id = "illustration_spline",
  section = "reference",
  filename = "illustrations-spline.svg",
  backend = :svg,
  build = () -> begin
    pts = [xy(0, 0), xy(2, 2), xy(4, -0.5), xy(6, 1.5), xy(8, 0)]
    spline(pts)
    for (i, p) in enumerate(pts)
      _ill_dot(p)
      _ill_label("p$(i)", p; dx=0.15, dy=0.25)
    end
  end,
)

register_scene(
  id = "illustration_ellipse",
  section = "reference",
  filename = "illustrations-ellipse.svg",
  backend = :svg,
  build = () -> begin
    c = xy(0, 0)
    rx, ry = 3.0, 1.7
    ellipse(c, rx, ry)
    _ill_dot(c)
    _ill_label("center", c; dx=-1.4, dy=0.20)
    _ill_dim(c, c + vxy(rx, 0), "radius_x")
    _ill_dim(c, c + vxy(0, ry), "radius_y")
  end,
)

register_scene(
  id = "illustration_surface_circle",
  section = "reference",
  filename = "illustrations-surface_circle.svg",
  backend = :svg,
  build = () -> begin
    c = xy(0, 0)
    r = 2.0
    surface_circle(c, r,
      material=KhepriSVG.svg_option(
        "fill:rgba(255,168,96,0.12);stroke:none"))
    circle(c, r)
    _ill_dot(c)
    _ill_label("p", c; dx=0.15, dy=0.20)
    _ill_dim(c, c + vxy(r, 0), "r")
  end,
)

register_scene(
  id = "illustration_surface_rectangle",
  section = "reference",
  filename = "illustrations-surface_rectangle.svg",
  backend = :svg,
  build = () -> begin
    corner = xy(0, 0)
    dx, dy = 4.0, 2.5
    surface_rectangle(corner, dx, dy,
      material=KhepriSVG.svg_option(
        "fill:rgba(255,168,96,0.12);stroke:none"))
    rectangle(corner, dx, dy)
    _ill_dot(corner)
    _ill_label("corner", corner; dx=-1.4, dy=-0.4)
    _ill_dim(corner + vxy(0, -0.5),
             corner + vxy(dx, -0.5), "dx")
    _ill_dim(corner + vxy(-0.5, 0),
             corner + vxy(-0.5, dy), "dy"; label_offset=-0.4)
  end,
)

# ==================================================================
# 3D shapes — projected silhouettes with annotated parameters
# ==================================================================

register_scene(
  id = "illustration_sphere",
  section = "reference",
  filename = "illustrations-sphere.svg",
  backend = :svg,
  build = () -> begin
    c = xy(0, 0)
    r = 2.0
    # Silhouette + an inner "equator" ellipse for 3D feel.
    circle(c, r)
    arc(c, r, 0, π)        # upper hemisphere
    # An equator ellipse is a horizontal squashed arc; approximate
    # with two arcs.
    let ry = r * 0.32
      arc(c + vxy(0, 0), r, π, π)  # lower silhouette already in circle
      # Equator dashed line: draw two narrow arcs
      line([c + vxy(-r, 0), c + vxy(r, 0)],
           material=_ill_dim_mat)
    end
    _ill_dot(c)
    _ill_label("p", c; dx=0.15, dy=0.25)
    _ill_dim(c, c + vxy(r, 0), "r")
  end,
)

# Iso projection helper for cuboids.  Returns SVG (x', y') given
# 3D (x, y, z).  Uses a 30° iso angle: y rotates 30° back-up-left.
const _ISO_DX = -0.5
const _ISO_DY = 0.32

_iso(x, y, z) = xy(x + _ISO_DX * y, z + _ISO_DY * y)

register_scene(
  id = "illustration_box",
  section = "reference",
  filename = "illustrations-box.svg",
  backend = :svg,
  build = () -> begin
    dx, dy, dz = 4.0, 3.0, 2.5
    # Three visible faces of an iso box anchored at the origin.
    polygon([_iso(0, 0, 0), _iso(dx, 0, 0),
             _iso(dx, 0, dz), _iso(0, 0, dz)])           # front
    polygon([_iso(0, 0, dz), _iso(dx, 0, dz),
             _iso(dx, dy, dz), _iso(0, dy, dz)])         # top
    polygon([_iso(dx, 0, 0), _iso(dx, dy, 0),
             _iso(dx, dy, dz), _iso(dx, 0, dz)])         # right
    # Hidden edges (back) drawn dashed-style via faint material
    line([_iso(0, 0, 0), _iso(0, dy, 0)], material=_ill_dim_mat)
    line([_iso(0, dy, 0), _iso(dx, dy, 0)], material=_ill_dim_mat)
    line([_iso(0, dy, 0), _iso(0, dy, dz)], material=_ill_dim_mat)
    # Anchor + dimensions
    _ill_dot(_iso(0, 0, 0))
    _ill_label("corner", _iso(0, 0, 0); dx=-1.6, dy=-0.4)
    _ill_dim(_iso(0, 0, -0.4), _iso(dx, 0, -0.4), "dx")
    _ill_dim(_iso(-0.5, 0, 0), _iso(-0.5, 0, dz), "dz";
             label_offset=-0.45)
    _ill_dim(_iso(dx + 0.4, 0, 0), _iso(dx + 0.4, dy, 0), "dy")
  end,
)

register_scene(
  id = "illustration_box_corners",
  section = "reference",
  filename = "illustrations-box_corners.svg",
  backend = :svg,
  build = () -> begin
    dx, dy, dz = 4.0, 3.0, 2.5
    polygon([_iso(0, 0, 0), _iso(dx, 0, 0),
             _iso(dx, 0, dz), _iso(0, 0, dz)])
    polygon([_iso(0, 0, dz), _iso(dx, 0, dz),
             _iso(dx, dy, dz), _iso(0, dy, dz)])
    polygon([_iso(dx, 0, 0), _iso(dx, dy, 0),
             _iso(dx, dy, dz), _iso(dx, 0, dz)])
    line([_iso(0, 0, 0), _iso(0, dy, 0)], material=_ill_dim_mat)
    line([_iso(0, dy, 0), _iso(dx, dy, 0)], material=_ill_dim_mat)
    line([_iso(0, dy, 0), _iso(0, dy, dz)], material=_ill_dim_mat)
    _ill_dot(_iso(0, 0, 0))
    _ill_label("p1", _iso(0, 0, 0); dx=-0.8, dy=-0.4)
    _ill_dot(_iso(dx, dy, dz))
    _ill_label("p2", _iso(dx, dy, dz); dx=0.20, dy=0.20)
  end,
)

register_scene(
  id = "illustration_cylinder",
  section = "reference",
  filename = "illustrations-cylinder.svg",
  backend = :svg,
  build = () -> begin
    r, h = 1.8, 4.0
    base = xy(0, 0)
    top  = xy(0, h)
    # Body silhouette
    line(base + vxy(-r, 0), base + vxy(-r, h))
    line(base + vxy( r, 0), base + vxy( r, h))
    # Top elliptical rim (full ellipse approximated as flattened arc)
    arc(top,  r, 0,   π)
    arc(top,  r, π,   π)        # back of top rim
    # Bottom rim — front half solid, back half faint
    arc(base, r, 0,   π)
    line([base + vxy(-r, 0), base + vxy(r, 0)],
         material=_ill_dim_mat)
    _ill_dot(base)
    _ill_label("p", base; dx=0.15, dy=0.25)
    _ill_dim(base, base + vxy(r, 0), "r")
    _ill_dim(base + vxy(r + 0.8, 0),
             base + vxy(r + 0.8, h), "h")
  end,
)

register_scene(
  id = "illustration_cone",
  section = "reference",
  filename = "illustrations-cone.svg",
  backend = :svg,
  build = () -> begin
    r, h = 2.0, 4.0
    base = xy(0, 0)
    apex = xy(0, h)
    line(base + vxy(-r, 0), apex)
    line(apex, base + vxy(r, 0))
    arc(base, r, 0, π)            # front of base rim
    line([base + vxy(-r, 0), base + vxy(r, 0)],
         material=_ill_dim_mat)   # back of base, faint
    _ill_dot(base)
    _ill_label("p", base; dx=0.15, dy=0.25)
    _ill_dim(base, base + vxy(r, 0), "r")
    _ill_dim(base + vxy(r + 0.8, 0),
             base + vxy(r + 0.8, h), "h")
  end,
)

register_scene(
  id = "illustration_regular_pyramid",
  section = "reference",
  filename = "illustrations-regular_pyramid.svg",
  backend = :svg,
  build = () -> begin
    rb, h = 2.0, 4.0
    base = xy(0, 0)
    apex = xy(0, h)
    # 4-edged base seen from a slight elevation
    p_left  = base + vxy(-rb, -0.4)
    p_right = base + vxy( rb, -0.4)
    p_back  = base + vxy(0,    0.4)
    line(p_left, apex)
    line(p_right, apex)
    line(p_back,  apex)
    line(p_left, p_back)
    line(p_back, p_right)
    line([p_left, p_right], material=_ill_dim_mat)  # hidden front edge
    _ill_dot(base)
    _ill_label("p", base; dx=-0.5, dy=-0.4)
    _ill_dim(base, base + vxy(rb, -0.4), "rb")
    _ill_dim(base + vxy(rb + 0.8, -0.4),
             base + vxy(rb + 0.8, h), "h")
  end,
)

register_scene(
  id = "illustration_torus",
  section = "reference",
  filename = "illustrations-torus.svg",
  backend = :svg,
  build = () -> begin
    R, r = 2.5, 0.6
    base = xy(0, 0)
    circle(base, R + r)
    circle(base, R - r)
    # Tube cross-section circles on left/right show the minor radius
    circle(base + vxy(R, 0), r)
    circle(base + vxy(-R, 0), r)
    _ill_dot(base)
    _ill_label("p", base; dx=0.15, dy=0.25)
    _ill_dim(base, base + vxy(R, 0), "R")
    _ill_dim(base + vxy(R, 0), base + vxy(R + r, 0), "r")
  end,
)
