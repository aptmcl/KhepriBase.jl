#=
Scenes for docs/src/bim/*.md — 3D renders produced by KhepriBlender.

Each scene declares the center and characteristic radius of its
content via `iso_view(...)` so the camera actually frames the
geometry instead of defaulting to the world origin.
=#

# ==================================================================
# Horizontal elements: slab + roof
# ==================================================================

register_scene(
  id = "bim_horizontal_slab",
  section = "bim",
  filename = "horizontal-slab.png",
  backend = :blender,
  view = iso_view(5, 4, 0, 10),
  build = () -> begin
    slab(rectangular_path(xy(0, 0), 10, 8), level(0))
  end,
)

register_scene(
  id = "bim_horizontal_l_slab",
  section = "bim",
  filename = "horizontal-l_slab.png",
  backend = :blender,
  view = iso_view(5, 4, 0, 10),
  build = () -> begin
    l = closed_polygonal_path([
      xy(0, 0), xy(10, 0), xy(10, 4),
      xy(5, 4), xy(5, 8), xy(0, 8)])
    slab(l, level(0))
  end,
)

register_scene(
  id = "bim_horizontal_roof",
  section = "bim",
  filename = "horizontal-roof.png",
  backend = :blender,
  view = iso_view(5, 4, 1.5, 12),
  build = () -> begin
    ground = level(0)
    slab(rectangular_path(xy(0, 0), 10, 8), ground)
    roof(rectangular_path(xy(-0.3, -0.3), 10.6, 8.6), level(3.0))
  end,
)

# ==================================================================
# Vertical elements: walls, doors, windows
# ==================================================================

register_scene(
  id = "bim_vertical_wall",
  section = "bim",
  filename = "vertical-wall.png",
  backend = :blender,
  view = iso_view(4, 0, 1.5, 7),
  build = () -> begin
    wall(open_polygonal_path([xy(0, 0), xy(8, 0)]),
         level(0), level(3.0))
  end,
)

register_scene(
  id = "bim_vertical_L_walls",
  section = "bim",
  filename = "vertical-L_walls.png",
  backend = :blender,
  view = iso_view(2.5, 2.5, 1.5, 7),
  build = () -> begin
    b, t = level(0), level(3.0)
    wall(open_polygonal_path([xy(0, 0), xy(5, 0)]), b, t)
    wall(open_polygonal_path([xy(5, 0), xy(5, 5)]), b, t)
  end,
)

# Skipping wall-with-openings renders: the KhepriBlender
# `b_subtract_ref(NativeRefs, NativeRefs)` path (used for door/window
# carve-outs) hits a NativeRefs → Int32 conversion bug inside the
# Python-side encoder.  Leaving the scene shelved until that is
# fixed; the wall-only variants above already illustrate the geometry.

# ==================================================================
# Structural elements
# ==================================================================

register_scene(
  id = "bim_structural_beam",
  section = "bim",
  filename = "structural-beam.png",
  backend = :blender,
  view = iso_view(4, 0, 3, 6),
  build = () -> begin
    beam(xyz(0, 0, 3), xyz(8, 0, 3))
  end,
)

register_scene(
  id = "bim_structural_column",
  section = "bim",
  filename = "structural-column.png",
  backend = :blender,
  view = iso_view(0, 0, 1.75, 4),
  build = () -> begin
    column(xy(0, 0), 0, level(0), level(3.5))
  end,
)

register_scene(
  id = "bim_structural_grid_columns",
  section = "bim",
  filename = "structural-grid_columns.png",
  backend = :blender,
  view = iso_view(6, 4.5, 1.5, 14),
  build = () -> begin
    slab(rectangular_path(xy(0, 0), 12, 9), level(0))
    for x in 0:3:12, y in 0:3:9
      column(xy(x, y), 0, level(0), level(3.0))
    end
    slab(rectangular_path(xy(0, 0), 12, 9), level(3.0))
  end,
)

register_scene(
  id = "bim_structural_truss",
  section = "bim",
  filename = "structural-truss.png",
  backend = :blender,
  view = iso_view(5, 0, 0.75, 8),
  build = () -> begin
    # Simple Warren truss — 5 bottom + 4 top nodes
    bottom = [xyz(2*i, 0, 0) for i in 0:5]
    top    = [xyz(2*i + 1, 0, 1.5) for i in 0:4]
    nb = [truss_node(p) for p in bottom]
    nt = [truss_node(p) for p in top]
    # Bottom chord
    for i in 1:(length(bottom) - 1)
      truss_bar(bottom[i], bottom[i+1])
    end
    # Top chord
    for i in 1:(length(top) - 1)
      truss_bar(top[i], top[i+1])
    end
    # Diagonals and verticals
    for i in 1:length(top)
      truss_bar(bottom[i],     top[i])
      truss_bar(bottom[i + 1], top[i])
    end
  end,
)

# ==================================================================
# Circulation: stairs
# ==================================================================

register_scene(
  id = "bim_circulation_straight_stair",
  section = "bim",
  filename = "circulation-straight_stair.png",
  backend = :blender,
  view = iso_view(0.5, 2.5, 1.5, 6),
  build = () -> begin
    stair(xy(0, 0), vy(1), level(0), level(3.0))
  end,
)

register_scene(
  id = "bim_circulation_spiral_stair",
  section = "bim",
  filename = "circulation-spiral_stair.png",
  backend = :blender,
  view = iso_view(0, 0, 1.5, 5),
  build = () -> begin
    spiral_stair(xy(0, 0), 1.5, 0, 2*pi, true, level(0), level(3.0))
  end,
)

register_scene(
  id = "bim_circulation_spiral_half",
  section = "bim",
  filename = "circulation-spiral_half.png",
  backend = :blender,
  view = iso_view(0, 0, 1.5, 5),
  build = () -> begin
    spiral_stair(xy(0, 0), 2.0, 0, pi, false, level(0), level(3.0))
  end,
)

# ==================================================================
# Spaces: 4-room floorplan
# ==================================================================

register_scene(
  id = "bim_spaces_4room_plan",
  section = "bim",
  filename = "spaces-4room_plan.png",
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

register_scene(
  id = "bim_spaces_two_storey",
  section = "bim",
  filename = "spaces-two_storey.png",
  backend = :blender,
  view = iso_view(5, 2.5, 3, 14),
  build = () -> begin
    ground = room(:living, :living_room, 6.0, 5.0) |
             room(:kitchen, :kitchen, 4.0, 5.0)
    upper  = room(:bed1, :bedroom, 4.0, 3.0) |
             room(:bed2, :bedroom, 3.0, 3.0) |
             room(:bath, :bathroom, 3.0, 3.0)
    desc = upper ^ ground
    build(layout(desc))
  end,
)

# ==================================================================
# Wall graph: junctions and a network
# ==================================================================

register_scene(
  id = "bim_wallgraph_single_wall",
  section = "bim",
  filename = "wallgraph-single_wall.png",
  backend = :blender,
  view = iso_view(3, 0, 1.5, 5),
  build = () -> begin
    g = wall_graph(level=level(0), height=3.0)
    j1 = junction!(g, xy(0, 0))
    j2 = junction!(g, xy(6, 0))
    segment!(g, j1, j2)
    build_walls(g)
  end,
)

register_scene(
  id = "bim_wallgraph_L_junction",
  section = "bim",
  filename = "wallgraph-l_junction.png",
  backend = :blender,
  view = iso_view(2.5, 2, 1.5, 6),
  build = () -> begin
    g = wall_graph(level=level(0), height=3.0)
    j1 = junction!(g, xy(0, 0))
    j2 = junction!(g, xy(5, 0))
    j3 = junction!(g, xy(5, 4))
    segment!(g, j1, j2)
    segment!(g, j2, j3)
    build_walls(g)
  end,
)

register_scene(
  id = "bim_wallgraph_t_junction",
  section = "bim",
  filename = "wallgraph-t_junction.png",
  backend = :blender,
  view = iso_view(4, 2, 1.5, 8),
  build = () -> begin
    g = wall_graph(level=level(0), height=3.0)
    a = junction!(g, xy(0, 0))
    b = junction!(g, xy(8, 0))
    c = junction!(g, xy(4, 0))
    d = junction!(g, xy(4, 4))
    segment!(g, a, c)
    segment!(g, c, b)
    segment!(g, c, d)
    build_walls(g)
  end,
)

register_scene(
  id = "bim_wallgraph_cross_junction",
  section = "bim",
  filename = "wallgraph-cross_junction.png",
  backend = :blender,
  view = iso_view(0, 0, 1.5, 8),
  build = () -> begin
    g = wall_graph(level=level(0), height=3.0)
    c = junction!(g, xy(0, 0))
    n = junction!(g, xy(0, 4))
    s = junction!(g, xy(0, -4))
    e = junction!(g, xy(4, 0))
    w = junction!(g, xy(-4, 0))
    segment!(g, n, c); segment!(g, c, s)
    segment!(g, w, c); segment!(g, c, e)
    build_walls(g)
  end,
)

register_scene(
  id = "bim_wallgraph_full_house",
  section = "bim",
  filename = "wallgraph-full_house.png",
  backend = :blender,
  view = iso_view(5, 4, 1.5, 12),
  build = () -> begin
    g = wall_graph(level=level(0), height=3.0)
    # Outline corners
    p1 = junction!(g, xy(0, 0))
    p2 = junction!(g, xy(10, 0))
    p3 = junction!(g, xy(10, 8))
    p4 = junction!(g, xy(0, 8))
    # Interior partition: T-junction between p1-p2 and up to p3-p4
    pmid_bot = junction!(g, xy(5, 0))
    pmid_top = junction!(g, xy(5, 8))
    # Perimeter
    segment!(g, p1, pmid_bot)
    segment!(g, pmid_bot, p2)
    segment!(g, p2, p3)
    segment!(g, p3, pmid_top)
    segment!(g, pmid_top, p4)
    segment!(g, p4, p1)
    # Interior partition
    segment!(g, pmid_bot, pmid_top)
    build_walls(g)
    slab(rectangular_path(xy(0, 0), 10, 8), level(0))
  end,
)
