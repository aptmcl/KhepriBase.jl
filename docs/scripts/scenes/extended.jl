#=
Extended scene coverage — additional images commissioned to
illustrate parts of Khepri that the first pass left under-served.
Grouped by the doc page they're embedded in.
=#

# ==================================================================
# BIM -- Horizontal: slab thickness + ceiling + roof overhang
# ==================================================================

register_scene(
  id = "bim_horizontal_slab_thicknesses",
  section = "bim",
  filename = "horizontal-slab_thicknesses.png",
  backend = :blender,
  view = iso_view(6, 1.5, 0, 9),
  build = () -> begin
    # Three slabs side-by-side with different thicknesses
    thin  = slab_family(thickness=0.08)
    std   = slab_family(thickness=0.20)
    thick = slab_family(thickness=0.40)
    slab(rectangular_path(xy(0, 0),  3.5, 3.0), level(0), thin)
    slab(rectangular_path(xy(4, 0),  3.5, 3.0), level(0), std)
    slab(rectangular_path(xy(8, 0),  3.5, 3.0), level(0), thick)
  end,
)

register_scene(
  id = "bim_horizontal_roof_overhang",
  section = "bim",
  filename = "horizontal-roof_overhang.png",
  backend = :blender,
  view = iso_view(5, 4, 1.8, 14),
  build = () -> begin
    slab(rectangular_path(xy(0, 0), 10, 8), level(0))
    wall(closed_polygonal_path([
        xy(0, 0), xy(10, 0), xy(10, 8), xy(0, 8)]),
      level(0), level(3.0))
    roof(rectangular_path(xy(-0.8, -0.8), 11.6, 9.6), level(3.0))
  end,
)

register_scene(
  id = "bim_horizontal_multi_level",
  section = "bim",
  filename = "horizontal-multi_level.png",
  backend = :blender,
  view = iso_view(5, 4, 4.5, 16),
  build = () -> begin
    slab(rectangular_path(xy(0, 0), 10, 8), level(0))
    slab(rectangular_path(xy(0, 0), 10, 8), level(3.0))
    slab(rectangular_path(xy(0, 0), 10, 8), level(6.0))
    roof(rectangular_path(xy(-0.3, -0.3), 10.6, 8.6), level(9.0))
    for i in 0:2
      wall(closed_polygonal_path([
          xy(0, 0), xy(10, 0), xy(10, 8), xy(0, 8)]),
        level(3.0 * i), level(3.0 * (i + 1)))
    end
  end,
)

# ==================================================================
# BIM -- Structural: more variety
# ==================================================================

register_scene(
  id = "bim_structural_free_column",
  section = "bim",
  filename = "structural-free_column.png",
  backend = :blender,
  view = iso_view(0, 0, 2, 5),
  build = () -> begin
    free_column(xyz(0, 0, 0), 4.0)
  end,
)

register_scene(
  id = "bim_structural_beam_column_grid",
  section = "bim",
  filename = "structural-beam_column_grid.png",
  backend = :blender,
  view = iso_view(6, 4.5, 2, 16),
  build = () -> begin
    slab(rectangular_path(xy(0, 0), 12, 9), level(0))
    # Columns
    for x in 0:3:12, y in 0:3:9
      column(xy(x, y), 0, level(0), level(3.5))
    end
    # Beams connecting column tops along x
    for y in 0:3:9, x in 0:3:9
      beam(xyz(x, y, 3.5), xyz(x + 3, y, 3.5))
    end
    # Beams connecting column tops along y
    for x in 0:3:12, y in 0:3:6
      beam(xyz(x, y, 3.5), xyz(x, y + 3, 3.5))
    end
  end,
)

register_scene(
  id = "bim_structural_column_families",
  section = "bim",
  filename = "structural-column_families.png",
  backend = :blender,
  view = iso_view(3, 0, 1.75, 6),
  build = () -> begin
    # Rectangular, square, round columns side-by-side
    rect = column_family(profile=rectangular_profile(0.2, 0.4))
    sq   = column_family(profile=rectangular_profile(0.3, 0.3))
    rnd  = column_family(profile=circular_profile(0.2))
    column(xy(0, 0), 0, level(0), level(3.5), rect)
    column(xy(2, 0), 0, level(0), level(3.5), sq)
    column(xy(4, 0), 0, level(0), level(3.5), rnd)
  end,
)

register_scene(
  id = "bim_structural_truss_with_supports",
  section = "bim",
  filename = "structural-truss_with_supports.png",
  backend = :blender,
  view = iso_view(4, 0, 0.75, 7),
  build = () -> begin
    bottom = [xyz(2*i, 0, 0) for i in 0:4]
    top    = [xyz(2*i + 1, 0, 1.5) for i in 0:3]
    truss_node(bottom[1], fixed_truss_node_family)
    truss_node(bottom[end], fixed_truss_node_family)
    for i in 2:(length(bottom) - 1)
      truss_node(bottom[i])
    end
    for p in top
      truss_node(p)
    end
    for i in 1:(length(bottom) - 1)
      truss_bar(bottom[i], bottom[i + 1])
    end
    for i in 1:(length(top) - 1)
      truss_bar(top[i], top[i + 1])
    end
    for i in 1:length(top)
      truss_bar(bottom[i], top[i])
      truss_bar(bottom[i + 1], top[i])
    end
  end,
)

# ==================================================================
# BIM -- Circulation: further stair variations
# ==================================================================

register_scene(
  id = "bim_circulation_stair_x_direction",
  section = "bim",
  filename = "circulation-stair_x_direction.png",
  backend = :blender,
  view = iso_view(2.5, 0.5, 1.5, 5),
  build = () -> begin
    stair(xy(0, 0), vx(1), level(0), level(3.0))
  end,
)

register_scene(
  id = "bim_circulation_stair_wide",
  section = "bim",
  filename = "circulation-stair_wide.png",
  backend = :blender,
  view = iso_view(0.75, 2.5, 1.5, 5),
  build = () -> begin
    wide = stair_family(width=1.5, riser_height=0.15, tread_depth=0.3)
    stair(xy(0, 0), vy(1), level(0), level(3.0), wide)
  end,
)

register_scene(
  id = "bim_circulation_stair_open_riser",
  section = "bim",
  filename = "circulation-stair_open_riser.png",
  backend = :blender,
  view = iso_view(0.5, 2.5, 1.5, 5),
  build = () -> begin
    open = stair_family(has_risers=false)
    stair(xy(0, 0), vy(1), level(0), level(3.0), open)
  end,
)

# ==================================================================
# BIM -- Spaces: simpler case + adjacency highlight
# ==================================================================

register_scene(
  id = "bim_spaces_two_rooms_simple",
  section = "bim",
  filename = "spaces-two_rooms_simple.png",
  backend = :blender,
  view = iso_view(4, 2, 1.5, 8),
  build = () -> begin
    desc = room(:living, :living_room, 5.0, 4.0) |
           room(:kitchen, :kitchen, 3.0, 4.0)
    build(layout(desc))
  end,
)

register_scene(
  id = "bim_spaces_six_rooms_corridor",
  section = "bim",
  filename = "spaces-six_rooms_corridor.png",
  backend = :blender,
  view = iso_view(6, 5, 1.5, 14),
  build = () -> begin
    floor = (room(:o1, :office, 4.0, 4.0) |
             room(:o2, :office, 4.0, 4.0) |
             room(:o3, :office, 4.0, 4.0)) /
            room(:corridor, :corridor, 12.0, 2.0) /
            (room(:o4, :office, 4.0, 4.0) |
             room(:o5, :office, 4.0, 4.0) |
             room(:o6, :office, 4.0, 4.0))
    build(layout(floor))
  end,
)

# ==================================================================
# BIM -- Wall Graph: arcs
# ==================================================================

register_scene(
  id = "bim_wallgraph_arc_wall",
  section = "bim",
  filename = "wallgraph-arc_wall.png",
  backend = :blender,
  view = iso_view(3, 1.5, 1.5, 7),
  build = () -> begin
    g = wall_graph(level=level(0), height=3.0)
    a = junction!(g, xy(0, 0))
    b = junction!(g, xy(6, 0))
    # 180° arc around (3, 0) passing above the chord
    arc_segment!(g, a, b; center=xy(3, 0), amplitude=pi)
    build_walls(g)
  end,
)

register_scene(
  id = "bim_wallgraph_curved_room",
  section = "bim",
  filename = "wallgraph-curved_room.png",
  backend = :blender,
  view = iso_view(0, 0, 1.5, 7),
  build = () -> begin
    g = wall_graph(level=level(0), height=3.0)
    a = junction!(g, xy(-3, 0))
    b = junction!(g, xy(3, 0))
    # Two arcs (upper and lower) enclosing a lens-shaped room
    arc_segment!(g, a, b; center=xy(0, 0), amplitude=pi)
    arc_segment!(g, b, a; center=xy(0, 0), amplitude=pi)
    build_walls(g)
  end,
)

# ==================================================================
# BIM -- Furnishings & Lights
# ==================================================================

register_scene(
  id = "bim_lights_pointlight",
  section = "bim",
  filename = "lights-pointlight.png",
  backend = :blender,
  view = iso_view(3, 2, 1.5, 7),
  build = () -> begin
    slab(rectangular_path(xy(0, 0), 6, 4), level(0))
    wall(closed_polygonal_path([
        xy(0, 0), xy(6, 0), xy(6, 4), xy(0, 4)]),
      level(0), level(3.0))
    pointlight(xyz(3, 2, 2.5); intensity=2000.0)
  end,
)

# KhepriBlender only registers a 6-arg `b_spotlight`; the user-level
# `spotlight(...)` realization calls a 4-arg `b_spotlight` that isn't
# wired up for Blender.  Shelving the spotlight scene until a
# dispatching signature is added.

register_scene(
  id = "bim_furniture_chair",
  section = "bim",
  filename = "furniture-chair.png",
  backend = :blender,
  view = iso_view(0, 0, 0.5, 2.2),
  build = () -> begin
    chair(xy(0, 0), level(0))
  end,
)

register_scene(
  id = "bim_furniture_table",
  section = "bim",
  filename = "furniture-table.png",
  backend = :blender,
  view = iso_view(0, 0, 0.5, 2.8),
  build = () -> begin
    table(xy(0, 0), level(0))
  end,
)

register_scene(
  id = "bim_furniture_table_and_chairs",
  section = "bim",
  filename = "furniture-table_and_chairs.png",
  backend = :blender,
  view = iso_view(0, 0, 0.5, 3.5),
  build = () -> begin
    table_and_chairs(xy(0, 0), level(0))
  end,
)

# ==================================================================
# Reference -- Shapes: primitive gallery expanded
# ==================================================================

register_scene(
  id = "reference_shapes_2d",
  section = "reference",
  filename = "shapes-2d.png",
  backend = :blender,
  view = top_view(4.5, 0, 6),
  build = () -> begin
    surface_rectangle(xy(-0.5, -0.5), 1, 1)
    surface_circle(xy(2, 0), 0.6)
    surface_polygon([xy(3.3, -0.6), xy(4.3, -0.6),
                     xy(4.7, 0.0), xy(4.3, 0.6),
                     xy(3.3, 0.6), xy(2.9, 0.0)])
    surface_polygon([xy(5.5, -0.6), xy(6.8, -0.6),
                     xy(7.15, 0.0), xy(6.8, 0.6),
                     xy(5.5, 0.6), xy(5.15, 0.0)])
    surface_regular_polygon(6, xy(9, 0), 0.6, 0.0, true)
  end,
)

register_scene(
  id = "reference_shapes_solids",
  section = "reference",
  filename = "shapes-solids.png",
  backend = :blender,
  view = iso_view(5, 0, 0.75, 9),
  build = () -> begin
    sphere(xyz(0, 0, 1), 0.8)
    box(xyz(2, -0.5, 0), 1, 1, 1.4)
    cylinder(xyz(4.5, 0, 0), 0.6, 1.4)
    cone(xyz(6.5, 0, 0), 0.7, 1.4)
    regular_pyramid(4, xyz(8.5, 0, 0), 0.7, 0, 1.4)
    torus(xyz(10.5, 0, 0.5), 0.7, 0.25)
  end,
)

register_scene(
  id = "reference_shapes_csg_union",
  section = "reference",
  filename = "shapes-csg_union.png",
  backend = :blender,
  view = iso_view(0, 0, 0.5, 3),
  build = () -> begin
    b = box(xyz(-0.5, -0.5, 0), 1, 1, 1)
    s = sphere(xyz(0.5, 0.5, 1.0), 0.6)
    union(b, s)
  end,
)

register_scene(
  id = "reference_shapes_csg_intersection",
  section = "reference",
  filename = "shapes-csg_intersection.png",
  backend = :blender,
  view = iso_view(0, 0, 0.5, 3),
  build = () -> begin
    b = box(xyz(-0.6, -0.6, 0), 1.2, 1.2, 1.2)
    s = sphere(xyz(0, 0, 0.6), 0.75)
    intersection(b, s)
  end,
)

register_scene(
  id = "reference_shapes_csg_subtraction",
  section = "reference",
  filename = "shapes-csg_subtraction.png",
  backend = :blender,
  view = iso_view(0, 0, 0.5, 3),
  build = () -> begin
    b = box(xyz(-0.6, -0.6, 0), 1.2, 1.2, 1.2)
    s = sphere(xyz(0.6, 0.6, 1.2), 0.55)
    subtraction(b, s)
  end,
)

register_scene(
  id = "reference_shapes_extrusion",
  section = "reference",
  filename = "shapes-extrusion.png",
  backend = :blender,
  view = iso_view(0, 0, 1, 4),
  build = () -> begin
    extrusion(surface_polygon([xy(-0.7, -0.5), xy(0.7, -0.5),
                                xy(0.9, 0.0), xy(0.7, 0.5),
                                xy(-0.7, 0.5), xy(-0.9, 0.0)]),
              vz(2.0))
  end,
)

register_scene(
  id = "reference_shapes_paths",
  section = "reference",
  filename = "shapes-paths.png",
  backend = :blender,
  view = top_view(3, 0, 5),
  build = () -> begin
    line([xy(-0.5, 1.0), xy(0.5, -1.0), xy(1.5, 1.0)])
    circle(xy(3.0, 0.0), 0.7)
    polygon([xy(4.8, -0.7), xy(6.2, -0.7),
             xy(6.2, 0.7),  xy(4.8, 0.7)])
  end,
)

# ==================================================================
# Reference -- Architectural materials showcase
# ==================================================================

register_scene(
  id = "reference_materials_showcase",
  section = "reference",
  filename = "materials-showcase.png",
  backend = :blender,
  view = iso_view(5.5, 0, 0.5, 8.5),
  build = () -> begin
    # Row of spheres, each with a different architectural material.
    mats = [material_basic, material_metal, material_glass, material_wood,
            material_concrete, material_plaster, material_grass, material_clay]
    for (i, m) in enumerate(mats)
      sphere(xyz((i - 1) * 1.6, 0, 0.5), 0.5, material=m)
    end
    # Ground plane to catch shadows / give a reference.
    slab(rectangular_path(xy(-1.0, -1.0), length(mats) * 1.6, 2.0),
         level(-0.02))
  end,
)

# ==================================================================
# Reference -- Design combinators: transforms
# ==================================================================

register_scene(
  id = "reference_designs_scale",
  section = "reference",
  filename = "designs-scale.svg",
  backend = :svg,
  build = () -> begin
    _labelled_rect(0, 0, 4, 3, "a")
    _labelled_rect(5, 0, 6, 4.5, "scale(a, 1.5)")
  end,
)

register_scene(
  id = "reference_designs_with_height",
  section = "reference",
  filename = "designs-with_height.svg",
  backend = :svg,
  build = () -> begin
    # Two stacked slabs, different heights, shown in elevation
    surface_polygon([xy(0, 0), xy(4, 0), xy(4, 2.8), xy(0, 2.8)],
                    material=_fill_material)
    polygon([xy(0, 0), xy(4, 0), xy(4, 2.8), xy(0, 2.8), xy(0, 0)],
            material=_stroke_material)
    _label("h = 2.8", xy(0.2, 1.3); h=0.3)
    surface_polygon([xy(5, 0), xy(9, 0), xy(9, 4.2), xy(5, 4.2)],
                    material=_fill_material)
    polygon([xy(5, 0), xy(9, 0), xy(9, 4.2), xy(5, 4.2), xy(5, 0)],
            material=_stroke_material)
    _label("with_height(a, 4.2)", xy(5.2, 2.0); h=0.3)
  end,
)

register_scene(
  id = "reference_designs_repeat_mirror",
  section = "reference",
  filename = "designs-repeat_mirror.svg",
  backend = :svg,
  build = () -> begin
    # Four copies with mirror_alternate
    for i in 0:3
      fill = _fill_material
      _labelled_rect(i * 4, 0, 4, 3,
                     (i % 2 == 0) ? "u$(i+1)" : "u$(i+1)ᵣ")
    end
    text("repeat_unit(unit, 4; mirror_alternate=true)",
         xy(0, -1), 0.35, material=_label_material)
  end,
)

# ==================================================================
# Reference -- Design annotations: connect / disconnect / no_windows
# ==================================================================

register_scene(
  id = "reference_designs_annotation_connect",
  section = "reference",
  filename = "designs-annotation_connect.svg",
  backend = :svg,
  build = () -> begin
    _labelled_rect(0, 0, 4, 3, ":living")
    _labelled_rect(4, 0, 4, 3, ":kitchen")
    # Door marker at the shared wall
    surface_rectangle(xy(4 - 0.12, 1.0), 0.24, 1.0,
                      material=_highlight_material)
    _label(":door", xy(4.4, 1.3); h=0.28)
    text("connect(desc, :living, :kitchen; kind=:door)",
         xy(0, -1), 0.3, material=_label_material)
  end,
)

register_scene(
  id = "reference_designs_annotation_exterior",
  section = "reference",
  filename = "designs-annotation_exterior.svg",
  backend = :svg,
  build = () -> begin
    _labelled_rect(0, 0, 6, 4, ":living")
    # Exterior window marker on the bottom edge
    surface_rectangle(xy(2, -0.12), 2, 0.24,
                      material=_highlight_material)
    _label(":window", xy(2.1, -0.35); h=0.28)
    text("connect_exterior(desc, :living; kind=:window, face=:south)",
         xy(0, -1.2), 0.3, material=_label_material)
  end,
)

# ==================================================================
# Reference -- Design leaves: rooms at different sizes / heights
# ==================================================================

register_scene(
  id = "reference_designs_room_sizes",
  section = "reference",
  filename = "designs-room_sizes.svg",
  backend = :svg,
  build = () -> begin
    _labelled_rect(0, 0, 3, 2.5, "3 × 2.5")
    _labelled_rect(4, 0, 5, 4,   "5 × 4")
    _labelled_rect(10, 0, 7, 5,  "7 × 5")
    text("room(:id, :use, width, depth)",
         xy(0, -1), 0.3, material=_label_material)
  end,
)

# ==================================================================
# Concepts -- extra: grid, repeat_unit, annotations, transforms
# ==================================================================

register_scene(
  id = "concepts_composition_grid",
  section = "concepts",
  filename = "composition-grid.svg",
  backend = :svg,
  build = () -> begin
    for r in 0:2, c in 0:3
      _labelled_rect(c * 4, r * 3, 4, 3, "c_$(r+1)_$(c+1)";
                     text_size=0.28)
    end
    text("grid((r, c) -> room(..., 4, 3), 3, 4)",
         xy(0, -1), 0.35, material=_label_material)
  end,
)

register_scene(
  id = "concepts_composition_repeat",
  section = "concepts",
  filename = "composition-repeat.svg",
  backend = :svg,
  build = () -> begin
    for i in 0:3
      _labelled_rect(i * 5, 0, 5, 4, "u_$(i+1)"; text_size=0.3)
    end
    text("repeat_unit(u, 4; axis=:x)",
         xy(0, -1), 0.35, material=_label_material)
  end,
)

register_scene(
  id = "concepts_annotations_no_windows",
  section = "concepts",
  filename = "annotations-no_windows.svg",
  backend = :svg,
  build = () -> begin
    _labelled_rect(0, 0, 5, 4, ":living")
    _labelled_rect(5, 0, 4, 4, ":kitchen")
    # Show explicit window markers on :living's exterior but none on :kitchen
    for x in 0.5:1.5:4.0
      surface_rectangle(xy(x, -0.1), 0.8, 0.2,
                        material=_highlight_material)
    end
    _label("no_windows(desc, :kitchen)",
           xy(0, -1.1); h=0.3)
  end,
)

register_scene(
  id = "concepts_transforms_scale",
  section = "concepts",
  filename = "transforms-scale.svg",
  backend = :svg,
  build = () -> begin
    _labelled_rect(0, 0, 3, 2, "a")
    _labelled_rect(4, 0, 6, 4, "scale(a, 2)")
  end,
)

register_scene(
  id = "concepts_transforms_mirror",
  section = "concepts",
  filename = "transforms-mirror.svg",
  backend = :svg,
  build = () -> begin
    # Left: asymmetric L shape
    surface_polygon([xy(0, 0), xy(3, 0), xy(3, 1), xy(1, 1),
                     xy(1, 3), xy(0, 3)],
                    material=_fill_material)
    polygon([xy(0, 0), xy(3, 0), xy(3, 1), xy(1, 1),
             xy(1, 3), xy(0, 3), xy(0, 0)],
            material=_stroke_material)
    _label("a", xy(0.3, 0.3); h=0.35)
    # Right: mirror_x of same
    surface_polygon([xy(8, 0), xy(5, 0), xy(5, 1), xy(7, 1),
                     xy(7, 3), xy(8, 3)],
                    material=_fill_material)
    polygon([xy(8, 0), xy(5, 0), xy(5, 1), xy(7, 1),
             xy(7, 3), xy(8, 3), xy(8, 0)],
            material=_stroke_material)
    _label("mirror_x(a)", xy(5.2, 0.3); h=0.35)
  end,
)
