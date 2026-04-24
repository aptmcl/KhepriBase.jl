#=
Scenes for docs/src/tutorials/*.md — 3D progression shots.
=#

# ==================================================================
# Building tutorial — 5 stages from empty to finished
# ==================================================================

register_scene(
  id = "tutorials_building_step1_slab",
  section = "tutorials",
  filename = "building-step1_slab.png",
  backend = :blender,
  view = VIEW_ISO_MEDIUM,
  build = () -> begin
    slab(rectangular_path(xy(0, 0), 10, 8), level(0))
  end,
)

register_scene(
  id = "tutorials_building_step2_walls",
  section = "tutorials",
  filename = "building-step2_walls.png",
  backend = :blender,
  view = VIEW_ISO_MEDIUM,
  build = () -> begin
    slab(rectangular_path(xy(0, 0), 10, 8), level(0))
    wall(closed_polygonal_path([
        xy(0, 0), xy(10, 0), xy(10, 8), xy(0, 8)]),
      level(0), level(3.0))
  end,
)

register_scene(
  id = "tutorials_building_step3_openings",
  section = "tutorials",
  filename = "building-step3_openings.png",
  backend = :blender,
  view = VIEW_ISO_MEDIUM,
  build = () -> begin
    slab(rectangular_path(xy(0, 0), 10, 8), level(0))
    w = wall(closed_polygonal_path([
        xy(0, 0), xy(10, 0), xy(10, 8), xy(0, 8)]),
      level(0), level(3.0))
    add_door(w, xy(2, 0))
    add_window(w, xy(6, 1.2))
    add_window(w, xy(8.5, 1.2))
  end,
)

register_scene(
  id = "tutorials_building_step4_roof",
  section = "tutorials",
  filename = "building-step4_roof.png",
  backend = :blender,
  view = VIEW_ISO_MEDIUM,
  build = () -> begin
    slab(rectangular_path(xy(0, 0), 10, 8), level(0))
    w = wall(closed_polygonal_path([
        xy(0, 0), xy(10, 0), xy(10, 8), xy(0, 8)]),
      level(0), level(3.0))
    add_door(w, xy(2, 0))
    add_window(w, xy(6, 1.2))
    roof(rectangular_path(xy(-0.3, -0.3), 10.6, 8.6), level(3.0))
  end,
)

register_scene(
  id = "tutorials_building_step5_two_storey",
  section = "tutorials",
  filename = "building-step5_two_storey.png",
  backend = :blender,
  view = VIEW_ISO_LARGE,
  build = () -> begin
    slab(rectangular_path(xy(0, 0), 10, 8), level(0))
    wall(closed_polygonal_path([
        xy(0, 0), xy(10, 0), xy(10, 8), xy(0, 8)]),
      level(0), level(3.0))
    slab(rectangular_path(xy(0, 0), 10, 8), level(3.0))
    wall(closed_polygonal_path([
        xy(0, 0), xy(10, 0), xy(10, 8), xy(0, 8)]),
      level(3.0), level(6.0))
    roof(rectangular_path(xy(-0.3, -0.3), 10.6, 8.6), level(6.0))
  end,
)

# ==================================================================
# Spaces tutorial progression
# ==================================================================

register_scene(
  id = "tutorials_spaces_two_rooms",
  section = "tutorials",
  filename = "spaces-two_rooms.png",
  backend = :blender,
  view = VIEW_ISO_SMALL,
  build = () -> begin
    desc = room(:a, :living_room, 5.0, 4.0) |
           room(:b, :kitchen,    3.0, 4.0)
    build(layout(desc))
  end,
)

register_scene(
  id = "tutorials_spaces_four_rooms",
  section = "tutorials",
  filename = "spaces-four_rooms.png",
  backend = :blender,
  view = VIEW_ISO_MEDIUM,
  build = () -> begin
    desc = (room(:living, :living_room, 5.0, 4.0) |
            room(:kitchen, :kitchen,    3.0, 4.0)) /
           (room(:bed,    :bedroom,     4.0, 3.0) |
            room(:bath,   :bathroom,    2.5, 3.0))
    build(layout(desc))
  end,
)

register_scene(
  id = "tutorials_spaces_multi_storey",
  section = "tutorials",
  filename = "spaces-multi_storey.png",
  backend = :blender,
  view = VIEW_ISO_LARGE,
  build = () -> begin
    ground = (room(:living, :living_room, 6.0, 5.0) |
              room(:kitchen, :kitchen,    4.0, 5.0))
    upper  = (room(:bed1, :bedroom, 5.0, 4.0) |
              room(:bed2, :bedroom, 3.0, 4.0) |
              room(:bath, :bathroom, 2.0, 4.0))
    build(layout(upper ^ ground))
  end,
)

# ==================================================================
# Wall graph tutorial progression
# ==================================================================

register_scene(
  id = "tutorials_wallgraph_step1",
  section = "tutorials",
  filename = "wallgraph-step1_one_wall.png",
  backend = :blender,
  view = VIEW_ISO_SMALL,
  build = () -> begin
    g = wall_graph()
    j1 = junction!(g, xy(0, 0))
    j2 = junction!(g, xy(6, 0))
    segment!(g, j1, j2)
    build_walls(g, level(0), level(3.0))
  end,
)

register_scene(
  id = "tutorials_wallgraph_step2_L",
  section = "tutorials",
  filename = "wallgraph-step2_L_shape.png",
  backend = :blender,
  view = VIEW_ISO_SMALL,
  build = () -> begin
    g = wall_graph()
    a = junction!(g, xy(0, 0))
    b = junction!(g, xy(5, 0))
    c = junction!(g, xy(5, 4))
    segment!(g, a, b); segment!(g, b, c)
    build_walls(g, level(0), level(3.0))
  end,
)

register_scene(
  id = "tutorials_wallgraph_step3_T",
  section = "tutorials",
  filename = "wallgraph-step3_T_junction.png",
  backend = :blender,
  view = VIEW_ISO_SMALL,
  build = () -> begin
    g = wall_graph()
    a = junction!(g, xy(0, 0))
    b = junction!(g, xy(8, 0))
    c = junction!(g, xy(4, 0))
    d = junction!(g, xy(4, 4))
    segment!(g, a, c); segment!(g, c, b); segment!(g, c, d)
    build_walls(g, level(0), level(3.0))
  end,
)

register_scene(
  id = "tutorials_wallgraph_step4_room",
  section = "tutorials",
  filename = "wallgraph-step4_closed_room.png",
  backend = :blender,
  view = VIEW_ISO_SMALL,
  build = () -> begin
    g = wall_graph()
    a = junction!(g, xy(0, 0))
    b = junction!(g, xy(6, 0))
    c = junction!(g, xy(6, 5))
    d = junction!(g, xy(0, 5))
    segment!(g, a, b); segment!(g, b, c)
    segment!(g, c, d); segment!(g, d, a)
    build_walls(g, level(0), level(3.0))
    slab(rectangular_path(xy(0, 0), 6, 5), level(0))
  end,
)

register_scene(
  id = "tutorials_wallgraph_step5_house",
  section = "tutorials",
  filename = "wallgraph-step5_house.png",
  backend = :blender,
  view = VIEW_ISO_MEDIUM,
  build = () -> begin
    g = wall_graph()
    # Perimeter of a 10x8 house with interior dividers
    p1 = junction!(g, xy(0, 0));  p2 = junction!(g, xy(10, 0))
    p3 = junction!(g, xy(10, 8)); p4 = junction!(g, xy(0, 8))
    mid_bot = junction!(g, xy(5, 0))
    mid_top = junction!(g, xy(5, 8))
    segment!(g, p1, mid_bot); segment!(g, mid_bot, p2)
    segment!(g, p2, p3); segment!(g, p3, mid_top)
    segment!(g, mid_top, p4); segment!(g, p4, p1)
    segment!(g, mid_bot, mid_top)
    build_walls(g, level(0), level(3.0))
    slab(rectangular_path(xy(0, 0), 10, 8), level(0))
  end,
)

# ==================================================================
# Rendering tutorial — four camera angles of same scene
# ==================================================================

_render_demo_scene() = begin
  slab(rectangular_path(xy(0, 0), 10, 8), level(0))
  w = wall(closed_polygonal_path([
      xy(0, 0), xy(10, 0), xy(10, 8), xy(0, 8)]),
    level(0), level(3.0))
  add_door(w, xy(2, 0))
  add_window(w, xy(6, 1.2))
  add_window(w, xy(4, 8))
  roof(rectangular_path(xy(-0.3, -0.3), 10.6, 8.6), level(3.0))
end

register_scene(
  id = "tutorials_rendering_iso",
  section = "tutorials",
  filename = "rendering-iso.png",
  backend = :blender,
  view = (eye=xyz(18, -18, 14), target=xyz(5, 4, 1.5)),
  build = _render_demo_scene,
)

register_scene(
  id = "tutorials_rendering_top",
  section = "tutorials",
  filename = "rendering-top.png",
  backend = :blender,
  view = (eye=xyz(5, 4, 30), target=xyz(5, 4, 0)),
  build = _render_demo_scene,
)

register_scene(
  id = "tutorials_rendering_front",
  section = "tutorials",
  filename = "rendering-front.png",
  backend = :blender,
  view = (eye=xyz(5, -15, 3), target=xyz(5, 4, 2)),
  build = _render_demo_scene,
)

register_scene(
  id = "tutorials_rendering_close",
  section = "tutorials",
  filename = "rendering-close.png",
  backend = :blender,
  view = (eye=xyz(12, -6, 3), target=xyz(2, 0, 1.5)),
  build = _render_demo_scene,
)

# ==================================================================
# Algorithmic tutorial — parametric structures
# ==================================================================

register_scene(
  id = "tutorials_algorithmic_column_grid",
  section = "tutorials",
  filename = "algorithmic-column_grid.png",
  backend = :blender,
  view = VIEW_ISO_LARGE,
  build = () -> begin
    slab(rectangular_path(xy(0, 0), 24, 18), level(0))
    for i in 0:6, j in 0:4
      column(xy(4*i, 4.5*j), 0, level(0), level(4.0))
    end
  end,
)

register_scene(
  id = "tutorials_algorithmic_recursive_tower",
  section = "tutorials",
  filename = "algorithmic-recursive_tower.png",
  backend = :blender,
  view = VIEW_ISO_LARGE,
  build = () -> begin
    # Stack 6 stepping-back floors
    for i in 0:5
      sz = 10 - i
      off = i * 0.5
      slab(rectangular_path(xy(off, off), sz, sz), level(3.0 * i))
      if i < 5
        wall(closed_polygonal_path([
            xy(off, off), xy(off+sz, off),
            xy(off+sz, off+sz), xy(off, off+sz)]),
          level(3.0 * i), level(3.0 * (i + 1)))
      end
    end
  end,
)

register_scene(
  id = "tutorials_algorithmic_radial_columns",
  section = "tutorials",
  filename = "algorithmic-radial_columns.png",
  backend = :blender,
  view = VIEW_ISO_MEDIUM,
  build = () -> begin
    slab(surface_regular_polygon_path(12, xy(0, 0), 10), level(0))
    for θ in range(0, 2π; length=13)[1:end-1]
      column(xy(8*cos(θ), 8*sin(θ)), 0, level(0), level(4.5))
    end
  end,
)

# ==================================================================
# Isenberg — same building bottom-up vs top-down
# ==================================================================

register_scene(
  id = "tutorials_isenberg_bottom_up",
  section = "tutorials",
  filename = "isenberg-bottom_up.png",
  backend = :blender,
  view = VIEW_ISO_LARGE,
  build = () -> begin
    # Simple stand-in: 3 offices in a row, corridor, 3 offices
    ground = (room(:o1, :office, 4.0, 4.0) |
              room(:o2, :office, 4.0, 4.0) |
              room(:o3, :office, 4.0, 4.0)) /
             room(:corridor, :corridor, 12.0, 2.0) /
             (room(:o4, :office, 4.0, 4.0) |
              room(:o5, :office, 4.0, 4.0) |
              room(:o6, :office, 4.0, 4.0))
    build(layout(ground))
  end,
)

register_scene(
  id = "tutorials_isenberg_top_down",
  section = "tutorials",
  filename = "isenberg-top_down.png",
  backend = :blender,
  view = VIEW_ISO_LARGE,
  build = () -> begin
    # Start from an envelope, carve a corridor, subdivide remaining
    env  = envelope(12.0, 10.0, 3.0; id=:floor)
    # subdivide into north (offices), corridor, south (offices)
    desc = subdivide_y(env, [0.4, 0.2, 0.4], [:n, :corridor, :s])
    desc = refine(desc, :n,
      n_env -> partition_x(n_env, 3, :no))
    desc = refine(desc, :s,
      s_env -> partition_x(s_env, 3, :so))
    desc = assign(desc, :corridor, :corridor)
    build(layout(desc))
  end,
)
