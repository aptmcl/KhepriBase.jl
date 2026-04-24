#=
Scenes for docs/src/reference/*.md — shape catalogues and worked
examples used in the reference section.
=#

# ==================================================================
# Shape primitives gallery (3D)
# ==================================================================

register_scene(
  id = "reference_shapes_primitives",
  section = "reference",
  filename = "shapes-primitives.png",
  backend = :blender,
  view = iso_view(6, 0, 1, 10),
  build = () -> begin
    # Row of primitives
    sphere(xyz(0, 0, 1), 1)
    box(xyz(2.5, -0.5, 0), 1, 1, 1.5)
    cylinder(xyz(5, 0, 0), 0.7, 1.5)
    cone(xyz(7, 0, 0), 0.8, 1.5)
    regular_pyramid(4, xyz(9, 0, 0), 0.8, 0, 1.5)
    torus(xyz(11.5, 0, 0.5), 0.8, 0.25)
  end,
)

# ==================================================================
# Camera view presets (same scene, different cameras)
# ==================================================================

_camera_demo_scene() = begin
  # Doors / windows omitted: the Blender b_subtract_ref encoding bug
  # breaks wall carve-outs in this context.
  slab(rectangular_path(xy(0, 0), 8, 6), level(0))
  wall(closed_polygonal_path([
      xy(0, 0), xy(8, 0), xy(8, 6), xy(0, 6)]),
    level(0), level(3.0))
end

register_scene(
  id = "reference_camera_iso",
  section = "reference",
  filename = "camera-iso.png",
  backend = :blender,
  view = iso_view(4, 3, 1.5, 10),
  build = _camera_demo_scene,
)

register_scene(
  id = "reference_camera_top",
  section = "reference",
  filename = "camera-top.png",
  backend = :blender,
  view = top_view(4, 3, 9),
  build = _camera_demo_scene,
)

register_scene(
  id = "reference_camera_elevation",
  section = "reference",
  filename = "camera-elevation.png",
  backend = :blender,
  view = front_view(4, 3, 1.5, 9),
  build = _camera_demo_scene,
)

# ==================================================================
# Layout engine — a worked example (top-down)
# ==================================================================

register_scene(
  id = "reference_layout_engine",
  section = "reference",
  filename = "layout-engine.png",
  backend = :blender,
  view = iso_view(4, 3.5, 1.5, 11),
  build = () -> begin
    desc = (room(:living, :living_room, 5.0, 4.0) |
            room(:kitchen, :kitchen,    3.0, 4.0)) /
           (room(:bed,    :bedroom,     4.0, 3.0) |
            room(:bath,   :bathroom,    2.5, 3.0))
    build(layout(desc))
  end,
)

# ==================================================================
# Design combinators — schematic SVG floorplans
# ==================================================================

register_scene(
  id = "reference_designs_repeat_unit",
  section = "reference",
  filename = "designs-repeat_unit.svg",
  backend = :svg,
  build = () -> begin
    for i in 0:2
      surface_rectangle(xy(6*i, 0), 6, 5)
      text("unit_$(i+1)", xy(6*i + 2, 2.3), 0.4)
    end
    text("repeat_unit(unit, 3; axis=:x)", xy(0, -1), 0.35)
  end,
)

register_scene(
  id = "reference_designs_grid",
  section = "reference",
  filename = "designs-grid.svg",
  backend = :svg,
  build = () -> begin
    for r in 0:1, c in 0:2
      surface_rectangle(xy(5*c, 4*r), 5, 4)
      text("c_$(r)_$(c)", xy(5*c + 1.5, 4*r + 1.8), 0.35)
    end
    text("grid((r,c) -> room(..., 5, 4), 2, 3)", xy(0, -1), 0.35)
  end,
)

register_scene(
  id = "reference_designs_mirror",
  section = "reference",
  filename = "designs-mirror.svg",
  backend = :svg,
  build = () -> begin
    # Left: asymmetric 3-room L
    surface_rectangle(xy(0, 0), 3, 3); text("a", xy(1.2, 1.3), 0.4)
    surface_rectangle(xy(3, 0), 3, 2); text("b", xy(4.2, 0.8), 0.4)
    surface_rectangle(xy(3, 2), 3, 1); text("c", xy(4.2, 2.3), 0.4)
    text("original", xy(1.5, -0.8), 0.35)
    # Right: mirrored
    surface_rectangle(xy(9, 0), 3, 3); text("a", xy(10.2, 1.3), 0.4)
    surface_rectangle(xy(6, 0), 3, 2); text("b", xy(7.2, 0.8), 0.4)
    surface_rectangle(xy(6, 2), 3, 1); text("c", xy(7.2, 2.3), 0.4)
    text("mirror_x", xy(7.5, -0.8), 0.35)
  end,
)

# ==================================================================
# Adjacencies reference example
# ==================================================================

register_scene(
  id = "reference_adjacencies_example",
  section = "reference",
  filename = "adjacencies-example.svg",
  backend = :svg,
  build = () -> begin
    # Show the 4-room house with labelled interior/exterior edges.
    surface_rectangle(xy(0, 0), 5, 4)
    surface_rectangle(xy(5, 0), 3, 4)
    surface_rectangle(xy(0, 4), 4, 3)
    surface_rectangle(xy(4, 4), 4, 3)
    text(":living",  xy(1.5, 1.8), 0.32)
    text(":kitchen", xy(5.6, 1.8), 0.32)
    text(":bed",     xy(1.0, 5.3), 0.32)
    text(":bath",    xy(5.0, 5.3), 0.32)

    # Interior shared walls (green-like thick rectangles)
    surface_rectangle(xy(5 - 0.08, 0.1), 0.16, 3.8)
    surface_rectangle(xy(0.1, 4 - 0.08), 3.8, 0.16)
    surface_rectangle(xy(4 - 0.08, 4.1), 0.16, 2.8)
    surface_rectangle(xy(5.1, 4 - 0.08), 2.8, 0.16)

    # Exterior edges emphasised with dashes
    for x in 0:0.5:7.5
      line(xy(x, 0), xy(x + 0.25, 0))         # south
      line(xy(x, 7), xy(x + 0.25, 7))         # north
    end
    for y in 0:0.5:6.5
      line(xy(0, y), xy(0, y + 0.25))         # west
      line(xy(8, y), xy(8, y + 0.25))         # east
    end
    text("interior: solid thick  |  exterior: dashed",
         xy(0, -0.9), 0.3)
  end,
)

# ==================================================================
# Designs subdivision gallery — one SVG per operator
# (All operators shown in concepts already; repeat a compact one here
# for the reference page)
# ==================================================================

register_scene(
  id = "reference_designs_assign",
  section = "reference",
  filename = "designs-assign.svg",
  backend = :svg,
  build = () -> begin
    surface_rectangle(xy(0, 0), 12, 4)
    line(xy(6, 0), xy(6, 4))
    line(xy(9, 0), xy(9, 4))
    text(":a",            xy(2.5, 1.8), 0.45)
    text(":open_office",  xy(6.4, 1.8), 0.35)
    text(":storage",      xy(9.4, 1.8), 0.35)
    text("assign(:a, :entrance)  …  assign(:b, :open_office)  …  assign(:c, :storage)",
         xy(0, -0.9), 0.3)
  end,
)
