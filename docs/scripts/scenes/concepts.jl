#=
Scenes for docs/src/concepts/*.md — mostly schematic 2D line drawings
rendered via KhepriSVG.

Convention for floor-plan schematics:
- Rooms are surface_rectangles with a light fill and a thin dark
  outline (see `_room_material` / `_label_material` below).
- Labels are black text sized to the rectangle.
- Dimensions are suggestive, not real — images communicate topology,
  not measurements.
=#

# ---- Styling: SVG-specific materials matching the schematic palette ----

#=
SVG styling.  The SVG backend's `svg_style_attr` blanks out `stroke` on
filled shapes, so a single `surface_rectangle(..., material=...)` can't
carry both a fill and an outline — we stack a filled shape and a
separate outline primitive on top to get the schematic "boxed room"
look with a faint panel fill.
=#

#=
Light palette tuned for the doc site's dark background.  Strokes
and text are off-white; room fills are a faint white tint so the
schematic structure shows through without dominating the figure.
Highlight colours stay vivid so they pop on dark.
=#

const _fill_material = KhepriSVG.svg_option(
  "fill:rgba(255,255,255,0.10);stroke:none")
const _warm_fill     = KhepriSVG.svg_option(
  "fill:rgba(255,180,160,0.40);stroke:none")
const _highlight_material = KhepriSVG.svg_option(
  "fill:rgb(120,200,140);stroke:none")
const _stroke_material = KhepriSVG.svg_option(
  "fill:none;stroke:rgb(245,245,247);stroke-width:0.05")
const _faint_stroke    = KhepriSVG.svg_option(
  "fill:none;stroke:rgba(245,245,247,0.55);stroke-width:0.025")
const _label_material = KhepriSVG.svg_option(
  "fill:rgb(245,245,247);stroke:none")

_room_rect(x, y, w, d; fill=_fill_material, stroke=_stroke_material) = begin
  corners = [xy(x, y), xy(x+w, y), xy(x+w, y+d), xy(x, y+d)]
  surface_polygon(corners, material=fill)
  polygon([corners..., corners[1]], material=stroke)
end

_labelled_rect(x, y, w, d, label; text_size=0.32,
               fill=_fill_material, stroke=_stroke_material) = begin
  _room_rect(x, y, w, d; fill=fill, stroke=stroke)
  text(label,
       xy(x + w/2 - length(label) * text_size * 0.3,
          y + d/2 - text_size/2),
       text_size,
       material=_label_material)
end

_diag_line(p1, p2) = line(p1, p2, material=_stroke_material)
_faint_line(p1, p2) = line(p1, p2, material=_faint_stroke)
_label(str, p; h=0.35) = text(str, p, h, material=_label_material)

# ==================================================================
# Levels of abstraction
# ==================================================================

# One diagram showing the three stacked rectangles representing Level 0/1/2
register_scene(
  id     = "concepts_levels_of_abstraction",
  section = "concepts",
  filename = "levels_of_abstraction.svg",
  backend = :svg,
  build = () -> begin
    # Level 0 — BIM primitives (low)
    surface_rectangle(xy(0, 0), 8, 1.5)
    text("Level 0 — BIM primitives  (wall, slab, door, …)",
         xy(0.3, 0.5), 0.35)
    # Level 1 — Layout (middle)
    surface_rectangle(xy(0, 2.5), 8, 1.5)
    text("Level 1 — Layout  (Space, Storey, Layout)",
         xy(0.3, 3.0), 0.35)
    # Level 2 — Design (top)
    surface_rectangle(xy(0, 5.0), 8, 1.5)
    text("Level 2 — Design  (SpaceDesc tree)",
         xy(0.3, 5.5), 0.35)
    # Downward arrows between layers
    line(xy(4, 5.0), xy(4, 4.1))
    text("layout(desc)", xy(4.15, 4.5), 0.3)
    line(xy(4, 2.5), xy(4, 1.6))
    text("build(layout)", xy(4.15, 2.0), 0.3)
  end,
)

# ==================================================================
# Composition operators
# ==================================================================

register_scene(
  id = "concepts_composition_beside_x",
  section = "concepts",
  filename = "composition-beside_x.svg",
  backend = :svg,
  build = () -> begin
    _labelled_rect(0, 0, 4, 3, "a")
    _labelled_rect(4, 0, 4, 3, "b")
    text("a | b  —  beside_x", xy(1, -1), 0.5)
  end,
)

register_scene(
  id = "concepts_composition_beside_y",
  section = "concepts",
  filename = "composition-beside_y.svg",
  backend = :svg,
  build = () -> begin
    _labelled_rect(0, 0, 4, 3, "a")
    _labelled_rect(0, 3, 4, 3, "b")
    text("a / b  —  beside_y", xy(0, -1), 0.5)
  end,
)

# "above" shown as isometric: draw both floors as parallelograms stacked
register_scene(
  id = "concepts_composition_above",
  section = "concepts",
  filename = "composition-above.svg",
  backend = :svg,
  build = () -> begin
    # Lower floor (projected parallelogram)
    surface_polygon([xy(0, 0), xy(4, 0), xy(4.6, 0.8), xy(0.6, 0.8)])
    text("a (ground)", xy(1.5, 0.3), 0.35)
    # Upper floor
    surface_polygon([xy(0, 1.6), xy(4, 1.6), xy(4.6, 2.4), xy(0.6, 2.4)])
    text("b (first)", xy(1.5, 1.9), 0.35)
    # Connecting edges
    line(xy(0, 0), xy(0, 1.6))
    line(xy(4, 0), xy(4, 1.6))
    line(xy(4.6, 0.8), xy(4.6, 2.4))
    line(xy(0.6, 0.8), xy(0.6, 2.4))
    text("b ^ a  —  above", xy(0, -0.8), 0.5)
  end,
)

register_scene(
  id = "concepts_composition_mixed",
  section = "concepts",
  filename = "composition-mixed.svg",
  backend = :svg,
  build = () -> begin
    # Ground floor: living | kitchen, with bed / bath behind
    _labelled_rect(0, 0, 5, 4, "living")
    _labelled_rect(5, 0, 3, 4, "kitchen")
    _labelled_rect(0, 4, 4, 3, "bed")
    _labelled_rect(4, 4, 4, 3, "bath")
    text("((living | kitchen) / (bed | bath))",
         xy(0, -1), 0.45)
  end,
)

# ==================================================================
# Subdivision vocabulary
# ==================================================================

register_scene(
  id = "concepts_subdivision_subdivide_x",
  section = "concepts",
  filename = "subdivision-subdivide_x.svg",
  backend = :svg,
  build = () -> begin
    # envelope 10×6 split into 0.3 / 0.5 / 0.2 zones
    surface_rectangle(xy(0, 0), 10, 6)
    line(xy(3, 0), xy(3, 6))
    line(xy(8, 0), xy(8, 6))
    text(":entry",  xy(1.0, 3),   0.35)
    text(":middle", xy(4.7, 3),   0.35)
    text(":svc",    xy(8.7, 3),   0.35)
    text("subdivide_x(env, [0.3, 0.5, 0.2], [:entry, :middle, :svc])",
         xy(0, -1), 0.35)
  end,
)

register_scene(
  id = "concepts_subdivision_split_x",
  section = "concepts",
  filename = "subdivision-split_x.svg",
  backend = :svg,
  build = () -> begin
    surface_rectangle(xy(0, 0), 12, 6)
    line(xy(4, 0), xy(4, 6))
    line(xy(9, 0), xy(9, 6))
    text("0 … 4",     xy(1.0,  3),   0.35)
    text("4 … 9",     xy(5.8,  3),   0.35)
    text("9 … 12",    xy(10.0, 3),   0.35)
    # Dimension arrows
    line(xy(0, -0.3), xy(4, -0.3))
    line(xy(4, -0.3), xy(9, -0.3))
    line(xy(9, -0.3), xy(12, -0.3))
    text("split_x(env, [4.0, 9.0], [:a, :b, :c])",
         xy(0, -1), 0.35)
  end,
)

register_scene(
  id = "concepts_subdivision_partition_x",
  section = "concepts",
  filename = "subdivision-partition_x.svg",
  backend = :svg,
  build = () -> begin
    surface_rectangle(xy(0, 0), 10, 6)
    for i in 1:4
      line(xy(2 * i, 0), xy(2 * i, 6))
    end
    for i in 1:5
      text(":zone_$i", xy(2 * (i - 1) + 0.3, 3), 0.3)
    end
    text("partition_x(env, 5, :zone)", xy(0, -1), 0.35)
  end,
)

register_scene(
  id = "concepts_subdivision_carve",
  section = "concepts",
  filename = "subdivision-carve.svg",
  backend = :svg,
  build = () -> begin
    surface_rectangle(xy(0, 0), 10, 6)
    line(xy(3, 2), xy(6, 2))
    line(xy(6, 2), xy(6, 4))
    line(xy(6, 4), xy(3, 4))
    line(xy(3, 4), xy(3, 2))
    text(":core",  xy(3.5, 3),   0.35)
    text("envelope(10, 6, 3)", xy(5.5, 0.3), 0.3)
    text("carve(env, :core, :stair; x=3, y=2, width=3, depth=2)",
         xy(0, -1), 0.35)
  end,
)

register_scene(
  id = "concepts_subdivision_refine",
  section = "concepts",
  filename = "subdivision-refine.svg",
  backend = :svg,
  build = () -> begin
    # Left: before (named zone :entry in a split envelope)
    surface_rectangle(xy(0, 0), 4, 4)
    line(xy(1.5, 0), xy(1.5, 4))
    text(":entry",  xy(0.2, 2), 0.3)
    text(":office", xy(2.0, 2), 0.3)
    text("before", xy(1.2, -0.7), 0.35)
    # Arrow
    line(xy(4.5, 2), xy(5.5, 2))
    # Right: :entry refined into a welcome + reception pair
    surface_rectangle(xy(6, 0), 4, 4)
    line(xy(7.5, 0), xy(7.5, 4))
    line(xy(6, 2), xy(7.5, 2))
    text(":welcome",   xy(6.1, 3),   0.22)
    text(":reception", xy(6.1, 1),   0.22)
    text(":office",    xy(8.0, 2),   0.3)
    text("after", xy(7.5, -0.7), 0.35)
    text("refine(desc, :entry, env -> split_y(env, [2], [:welcome, :reception]))",
         xy(0, -1.8), 0.3)
  end,
)

register_scene(
  id = "concepts_subdivision_subdivide_remaining",
  section = "concepts",
  filename = "subdivision-subdivide_remaining.svg",
  backend = :svg,
  build = () -> begin
    # outer envelope, inner carve, 4 named blocks
    surface_rectangle(xy(0, 0), 10, 6)
    line(xy(2, 1), xy(6, 1))
    line(xy(6, 1), xy(6, 4))
    line(xy(6, 4), xy(2, 4))
    line(xy(2, 4), xy(2, 1))
    # The 4 L-blocks (north/south/east/west)
    text(":north",  xy(3.8, 4.9), 0.32)
    text(":south",  xy(3.8, 0.4), 0.32)
    text(":east",   xy(7.7, 2.4), 0.32)
    text(":west",   xy(0.4, 2.4), 0.32)
    text(":atrium", xy(3.3, 2.4), 0.32)
    text("subdivide_remaining(env_with_carved_atrium, [:north, :south, :east, :west])",
         xy(0, -1), 0.3)
  end,
)

# ==================================================================
# Leaf types
# ==================================================================

register_scene(
  id = "concepts_leaf_room",
  section = "concepts",
  filename = "leaf-room.svg",
  backend = :svg,
  build = () -> begin
    _labelled_rect(0, 0, 5, 4, ":living_room")
    text("room(:id, :use, width, depth)",
         xy(0, -0.9), 0.35)
  end,
)

register_scene(
  id = "concepts_leaf_void",
  section = "concepts",
  filename = "leaf-void.svg",
  backend = :svg,
  build = () -> begin
    _labelled_rect(0, 0, 3, 3, "room a")
    # Dashed boundary for Void
    for x in 0:0.4:1.5
      line(xy(3 + x, 0), xy(3 + x + 0.2, 0))
      line(xy(3 + x, 3), xy(3 + x + 0.2, 3))
    end
    for y in 0:0.4:3
      line(xy(3, y), xy(3, y + 0.2))
      line(xy(4.5, y), xy(4.5, y + 0.2))
    end
    text("void(1.5, 0)", xy(3.2, 1.5), 0.25)
    _labelled_rect(4.5, 0, 3, 3, "room b")
    text("A Void reserves space without producing a Space.",
         xy(0, -0.9), 0.3)
  end,
)

register_scene(
  id = "concepts_leaf_envelope",
  section = "concepts",
  filename = "leaf-envelope.svg",
  backend = :svg,
  build = () -> begin
    surface_rectangle(xy(0, 0), 8, 5)
    text(":envelope", xy(3.2, 2.3), 0.5)
    text("envelope(8, 5, 3)  —  a shell for top-down subdivision",
         xy(0, -0.9), 0.35)
  end,
)

register_scene(
  id = "concepts_leaf_polar_envelope",
  section = "concepts",
  filename = "leaf-polar_envelope.svg",
  backend = :svg,
  build = () -> begin
    # Ring sector from angles 20° to 110°, inner radius 2, outer 5
    surface_arc(xy(0, 0), 5, deg2rad(20), deg2rad(90))
    surface_circle(xy(0, 0), 2)
    text("polar_envelope", xy(1.5, 3), 0.3)
    text("(center, r_inner, r_outer, θ_start, θ_end, height)",
         xy(-3, -1), 0.25)
  end,
)

# ==================================================================
# Adjacency diagram
# ==================================================================

register_scene(
  id = "concepts_adjacency_basic",
  section = "concepts",
  filename = "adjacency-basic.svg",
  backend = :svg,
  build = () -> begin
    # 2x2 house floorplan
    _labelled_rect(0, 0, 5, 4, ":living")
    _labelled_rect(5, 0, 3, 4, ":kitchen")
    _labelled_rect(0, 4, 4, 3, ":bed")
    _labelled_rect(4, 4, 4, 3, ":bath")

    # Adjacency markers: thick segments where rooms share walls
    # (we'll use surface_rectangle as a highlight)
    surface_rectangle(xy(5 - 0.08, 0.1), 0.16, 3.8)    # living-kitchen (interior)
    surface_rectangle(xy(0.1, 4 - 0.08), 3.8, 0.16)    # living-bed (interior)
    surface_rectangle(xy(4 - 0.08, 4.1), 0.16, 2.8)    # bed-bath (interior)
    surface_rectangle(xy(5.1, 4 - 0.08), 2.8, 0.16)    # kitchen-bath (interior)

    text("Interior adjacencies (shared walls) highlighted.",
         xy(0, -0.9), 0.3)
  end,
)

# ==================================================================
# Constraint violations
# ==================================================================

register_scene(
  id = "concepts_constraint_min_area",
  section = "concepts",
  filename = "constraint-min_area_violation.svg",
  backend = :svg,
  build = () -> begin
    _labelled_rect(0, 0, 5, 4, ":bedroom  20m²  ok")
    _labelled_rect(5, 0, 3, 2, "")
    text(":bedroom  6m²  ✗", xy(5.1, 1), 0.3)
    # Red-ish cross-hatch on the violating room
    for x in 5.1:0.3:7.9
      line(xy(x, 0.1), xy(x + 0.2, 0.3))
    end
    text("min_area(:bedroom, 9.0) violated.", xy(0, -0.9), 0.35)
  end,
)

register_scene(
  id = "concepts_constraint_must_adjoin",
  section = "concepts",
  filename = "constraint-must_adjoin_failure.svg",
  backend = :svg,
  build = () -> begin
    _labelled_rect(0, 0, 3, 3, ":bath")
    _labelled_rect(3, 0, 5, 3, ":living")
    _labelled_rect(0, 3, 8, 3, ":bed  (isolated)")
    # Mark bath-bed pair with a strike
    line(xy(1.5, 1.5), xy(4, 4.5))
    line(xy(4, 1.5), xy(1.5, 4.5))
    text("must_adjoin(:bath, :bed) violated — no shared wall.",
         xy(0, -0.9), 0.3)
  end,
)

# ==================================================================
# Designs overview tree
# ==================================================================

register_scene(
  id = "concepts_designs_tree",
  section = "concepts",
  filename = "designs-tree.svg",
  backend = :svg,
  build = () -> begin
    # Render a simplified SpaceDesc tree diagram.
    # Root node at (4, 0)
    text("BesideY", xy(3.5, 0), 0.35)
    # Two children
    line(xy(4, 0), xy(2, -2))
    line(xy(4, 0), xy(6, -2))
    text("BesideX", xy(1.5, -2), 0.3)
    text("BesideX", xy(5.5, -2), 0.3)
    # Grand-children
    line(xy(2, -2), xy(1, -4));  line(xy(2, -2), xy(3, -4))
    line(xy(6, -2), xy(5, -4));  line(xy(6, -2), xy(7, -4))
    text("Room :living", xy(0.2, -4), 0.22)
    text("Room :kitchen", xy(2.5, -4), 0.22)
    text("Room :bed", xy(4.5, -4), 0.22)
    text("Room :bath", xy(6.5, -4), 0.22)
  end,
)

# ==================================================================
# Family variants (same family, different parameters)
# ==================================================================

register_scene(
  id = "concepts_family_variants",
  section = "concepts",
  filename = "family-variants.svg",
  backend = :svg,
  build = () -> begin
    # Three walls of same family, different lengths/widths
    _labelled_rect(0, 0, 4, 0.2, "")
    text("wall(p1, p2, family)", xy(0, -0.5), 0.25)
    _labelled_rect(0, 2, 6, 0.3, "")
    text("wall(p1, p2, family; width=0.3)", xy(0, 1.5), 0.25)
    _labelled_rect(0, 4, 5, 0.25, "")
    text("wall(p1, p2, family_tall; height=3.5)", xy(0, 3.5), 0.25)
  end,
)
