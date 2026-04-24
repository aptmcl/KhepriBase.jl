#=
Extended scene set #2 — progressive tutorial shots.

Each tutorial has a numbered step sequence; this file materialises
most of those steps as rendered images so the doc reader sees what
every chunk of code produces, not just the final result.
=#

# ==================================================================
# Tutorials / Building a Complete Building — cumulative office
# ==================================================================
#=
Builds the tutorial's 2-storey office up to `up_to_step`.  Each
register_scene below pins a different `up_to_step` so the doc page
can illustrate each section.  The coordinates and family values
match the tutorial's code block-for-block.
=#
function _build_office(up_to_step::Int)
  ground      = level(0.0)
  first_floor = level(3.5)
  roof_level  = level(7.0)
  ext_wall    = wall_family(thickness=0.3)
  int_wall    = wall_family(thickness=0.15)
  main_door   = door_family(width=1.2, height=2.2)
  office_door = door_family(width=0.9, height=2.1)
  tall_window = window_family(width=1.4, height=1.6)
  col_family  = column_family(profile=rectangular_profile(0.3, 0.3))
  office_stair = stair_family(width=1.2, riser_height=0.175, tread_depth=0.28)
  building_region = rectangular_path(xy(0, 0), 16, 12)

  # Step 1: ground slab
  slab(building_region, ground)
  up_to_step <= 1 && return

  # Step 2: exterior walls
  exterior = wall(
    closed_polygonal_path([
      xy(0, 0), xy(16, 0), xy(16, 12), xy(0, 12)]),
    ground, first_floor, ext_wall)
  up_to_step <= 2 && return

  # Step 3: doors + windows on exterior
  add_door(exterior, xy(2, 0), main_door)
  for xc in (5, 8, 11, 18, 21, 24, 30, 33, 36, 42, 45)
    add_window(exterior, xy(xc, 1.0), tall_window)
  end
  up_to_step <= 3 && return

  # Step 4: interior partitions + office doors
  corridor_wall = wall(
    open_polygonal_path([xy(0.3, 6), xy(15.7, 6)]),
    ground, first_floor, int_wall)
  wall(open_polygonal_path([xy(8, 0.3), xy(8, 5.85)]),
       ground, first_floor, int_wall)
  add_door(corridor_wall, xy(3, 0), office_door)
  add_door(corridor_wall, xy(10, 0), office_door)
  up_to_step <= 4 && return

  # Step 5: columns
  for x in (0.15, 8, 15.85), y in (0.15, 6, 11.85)
    column(xy(x, y), 0, ground, first_floor, col_family)
  end
  up_to_step <= 5 && return

  # Step 6: stairwell
  stair(xy(13, 8), vy(1), ground, first_floor, office_stair)
  railing(open_polygonal_path([xy(13, 8), xyz(13, 13.5, 3.5)]), ground)
  railing(open_polygonal_path([xy(14.2, 8), xyz(14.2, 13.5, 3.5)]), ground)
  up_to_step <= 6 && return

  # Step 7: first-floor slab
  slab(building_region, first_floor)
  up_to_step <= 7 && return

  # Step 8: first-floor exterior + partitions + columns
  exterior_1f = wall(
    closed_polygonal_path([
      xy(0, 0), xy(16, 0), xy(16, 12), xy(0, 12)]),
    first_floor, roof_level, ext_wall)
  for xc in (5, 8, 11, 18, 21, 24, 30, 33, 36, 42, 45)
    add_window(exterior_1f, xy(xc, 1.0), tall_window)
  end
  corridor_1f = wall(
    open_polygonal_path([xy(0.3, 6), xy(15.7, 6)]),
    first_floor, roof_level, int_wall)
  add_door(corridor_1f, xy(3, 0), office_door)
  add_door(corridor_1f, xy(10, 0), office_door)
  for x in (0.15, 8, 15.85), y in (0.15, 6, 11.85)
    column(xy(x, y), 0, first_floor, roof_level, col_family)
  end
  up_to_step <= 8 && return

  # Step 9: balcony
  balcony_region = rectangular_path(xy(4, -2), 8, 2)
  slab(balcony_region, first_floor, slab_family(thickness=0.15))
  railing(open_polygonal_path([xy(4, -2), xy(12, -2), xy(12, 0)]),
          first_floor)
  railing(open_polygonal_path([xy(4, 0), xy(4, -2)]), first_floor)
  up_to_step <= 9 && return

  # Step 10: roof with perimeter railing
  roof(rectangular_path(xy(-0.3, -0.3), 16.6, 12.6), roof_level)
  railing(open_polygonal_path([
      xy(0, 0), xy(16, 0), xy(16, 12), xy(0, 12), xy(0, 0)]),
    roof_level, nothing, railing_family(height=1.1))
  up_to_step <= 10 && return

  # Step 11: furnishings
  table_and_chairs(xy(4, 9), 0, ground)
  conference = table_chair_family(
    table_family=table_family(length=2.4, width=1.0),
    chairs_top=1, chairs_bottom=1,
    chairs_right=3, chairs_left=3)
  table_and_chairs(xy(4, 9), 0, first_floor, conference)
  for (x, y) in ((3, 2), (3, 4), (10, 2), (10, 4))
    table(xy(x, y), 0, ground, table_family(length=1.4, width=0.7))
    chair(xy(x, y - 0.6), 0, ground)
  end
  up_to_step <= 11 && return

  # Step 12: lighting
  for x in 4:4:12, y in (3, 9)
    pointlight(xyz(x, y, 3.2); color=rgb(1, 0.98, 0.95),
               intensity=800.0, level=ground)
    pointlight(xyz(x, y, 3.2); color=rgb(1, 0.98, 0.95),
               intensity=800.0, level=first_floor)
  end
  return
end

const _OFFICE_VIEW = iso_view(8, 6, 3.5, 22)

register_scene(id = "tutorials_building_01_slab",
  section = "tutorials", filename = "building-01_slab.png",
  backend = :blender, view = iso_view(8, 6, 0, 14),
  build = () -> _build_office(1))

register_scene(id = "tutorials_building_02_walls",
  section = "tutorials", filename = "building-02_walls.png",
  backend = :blender, view = _OFFICE_VIEW,
  build = () -> _build_office(2))

register_scene(id = "tutorials_building_03_openings",
  section = "tutorials", filename = "building-03_openings.png",
  backend = :blender, view = _OFFICE_VIEW,
  build = () -> _build_office(3))

register_scene(id = "tutorials_building_04_partitions",
  section = "tutorials", filename = "building-04_partitions.png",
  backend = :blender, view = _OFFICE_VIEW,
  build = () -> _build_office(4))

register_scene(id = "tutorials_building_05_columns",
  section = "tutorials", filename = "building-05_columns.png",
  backend = :blender, view = _OFFICE_VIEW,
  build = () -> _build_office(5))

register_scene(id = "tutorials_building_06_stairwell",
  section = "tutorials", filename = "building-06_stairwell.png",
  backend = :blender, view = _OFFICE_VIEW,
  build = () -> _build_office(6))

register_scene(id = "tutorials_building_07_first_floor",
  section = "tutorials", filename = "building-07_first_floor.png",
  backend = :blender, view = iso_view(8, 6, 4, 22),
  build = () -> _build_office(7))

register_scene(id = "tutorials_building_08_upper_walls",
  section = "tutorials", filename = "building-08_upper_walls.png",
  backend = :blender, view = iso_view(8, 6, 4, 22),
  build = () -> _build_office(8))

register_scene(id = "tutorials_building_09_balcony",
  section = "tutorials", filename = "building-09_balcony.png",
  backend = :blender, view = iso_view(8, 5, 4, 22),
  build = () -> _build_office(9))

register_scene(id = "tutorials_building_10_roof",
  section = "tutorials", filename = "building-10_roof.png",
  backend = :blender, view = iso_view(8, 6, 5, 22),
  build = () -> _build_office(10))

register_scene(id = "tutorials_building_11_furnishings",
  section = "tutorials", filename = "building-11_furnishings.png",
  backend = :blender, view = iso_view(8, 6, 5, 22),
  build = () -> _build_office(11))

register_scene(id = "tutorials_building_12_hero",
  section = "tutorials", filename = "building-12_hero.png",
  backend = :blender, view = (eye = xyz(25, -15, 12),
                               target = xyz(8, 6, 3)),
  build = () -> _build_office(12))

# ==================================================================
# Tutorials / Space-First — house progression + examples
# ==================================================================

register_scene(
  id = "tutorials_spaces_two_rooms_door",
  section = "tutorials",
  filename = "spaces-two_rooms_door.png",
  backend = :blender,
  view = iso_view(5, 2, 1.5, 9),
  build = () -> begin
    plan = floor_plan()
    room_a = add_space(plan, "A", rectangular_path(u0(), 5, 4))
    room_b = add_space(plan, "B", rectangular_path(xy(5, 0), 5, 4))
    add_door(plan, room_a, room_b)
    build(plan)
  end,
)

register_scene(
  id = "tutorials_spaces_house",
  section = "tutorials",
  filename = "spaces-house.png",
  backend = :blender,
  view = iso_view(5, 5, 1.5, 13),
  build = () -> begin
    plan = floor_plan(height=2.8,
                      wall_family=wall_family(thickness=0.2),
                      slab_family=slab_family(thickness=0.25))
    living   = add_space(plan, "Living",
      closed_polygonal_path([xy(0,0), xy(6,0), xy(6,5), xy(0,5)]);
      kind=:room)
    kitchen  = add_space(plan, "Kitchen",
      closed_polygonal_path([xy(6,0), xy(10,0), xy(10,5), xy(6,5)]);
      kind=:kitchen)
    corridor = add_space(plan, "Corridor",
      closed_polygonal_path([xy(0,5), xy(10,5), xy(10,6.2), xy(0,6.2)]);
      kind=:corridor)
    bed1     = add_space(plan, "Master",
      closed_polygonal_path([xy(0,6.2), xy(4.5,6.2), xy(4.5,10), xy(0,10)]);
      kind=:bedroom)
    bed2     = add_space(plan, "Bed 2",
      closed_polygonal_path([xy(4.5,6.2), xy(8,6.2), xy(8,10), xy(4.5,10)]);
      kind=:bedroom)
    bath     = add_space(plan, "Bath",
      closed_polygonal_path([xy(8,6.2), xy(10,6.2), xy(10,10), xy(8,10)]);
      kind=:wc)
    add_door(plan, living, corridor)
    add_door(plan, kitchen, corridor)
    add_door(plan, bed1, corridor)
    add_door(plan, bed2, corridor)
    add_door(plan, bath, corridor)
    add_door(plan, living, kitchen)
    add_door(plan, living, :exterior; loc=xy(3.0, 0))
    build(plan)
  end,
)

register_scene(
  id = "tutorials_spaces_grid_offices",
  section = "tutorials",
  filename = "spaces-grid_offices.png",
  backend = :blender,
  view = iso_view(9, 6, 1.5, 20),
  build = () -> begin
    plan = floor_plan()
    offices = Dict{Tuple{Int,Int}, Any}()
    for r in 1:3, c in 1:4
      x, y = (c - 1) * 5, (r - 1) * 4
      offices[(r, c)] = add_space(plan, "o_$(r)_$(c)",
        rectangular_path(xy(x, y), 5, 4); kind=:office)
    end
    build(plan)
  end,
)

register_scene(
  id = "tutorials_spaces_radial",
  section = "tutorials",
  filename = "spaces-radial.png",
  backend = :blender,
  view = iso_view(0, 0, 1.5, 12),
  build = () -> begin
    plan = floor_plan()
    n = 6
    r_in, r_out = 3.0, 7.0
    for i in 1:n
      θ0 = 2π * (i - 1) / n
      θ1 = 2π * i / n
      add_space(plan, "r$i", closed_polygonal_path([
        xy(r_in * cos(θ0), r_in * sin(θ0)),
        xy(r_out * cos(θ0), r_out * sin(θ0)),
        xy(r_out * cos(θ1), r_out * sin(θ1)),
        xy(r_in * cos(θ1), r_in * sin(θ1))]);
        kind=:room)
    end
    build(plan)
  end,
)

# ==================================================================
# Tutorials / Rendering — visual_style variations
# ==================================================================

_render_demo_scene_v2() = begin
  slab(rectangular_path(xy(0, 0), 8, 6), level(0))
  wall(closed_polygonal_path([
      xy(0, 0), xy(8, 0), xy(8, 6), xy(0, 6)]),
    level(0), level(3.0))
  roof(rectangular_path(xy(-0.2, -0.2), 8.4, 6.4), level(3.0))
  for x in 1:2:7
    cylinder(xyz(x, 5, 0), 0.3, 3.0)
  end
  table_and_chairs(xy(3, 3), 0, level(0))
end

# Three visual styles rendered from the same camera.  The driver's
# render_one handles the :shaded default; we need a different
# visual_style so we render each via a direct `render_view` call.
# Easiest: use the same scene pattern but tag different styles via a
# post-build hook.  Here we just produce three base images and let
# the driver stamp :shaded on all three for now; readers will see
# the same scene three times until a visual-style hook is added.
register_scene(
  id = "tutorials_rendering_style_shaded",
  section = "tutorials",
  filename = "rendering-style_shaded.png",
  backend = :blender,
  view = iso_view(4, 3, 1.8, 11),
  build = _render_demo_scene_v2,
)

register_scene(
  id = "tutorials_rendering_exposure_bright",
  section = "tutorials",
  filename = "rendering-exposure_bright.png",
  backend = :blender,
  view = iso_view(4, 3, 1.8, 11),
  build = () -> begin
    _render_demo_scene_v2()
    # Add two bright pointlights for an over-exposed look.
    pointlight(xyz(2, 3, 2.5); intensity=3000.0)
    pointlight(xyz(6, 3, 2.5); intensity=3000.0)
  end,
)

register_scene(
  id = "tutorials_rendering_night",
  section = "tutorials",
  filename = "rendering-night.png",
  backend = :blender,
  view = iso_view(4, 3, 1.8, 11),
  build = () -> begin
    _render_demo_scene_v2()
    # Night-like: just a single warm pointlight above the table.
    pointlight(xyz(4, 3, 2.6); color=rgb(1.0, 0.75, 0.55),
               intensity=1800.0)
  end,
)

# ==================================================================
# Tutorials / Algorithmic — utility-function visualisations
# ==================================================================

register_scene(
  id = "tutorials_algorithmic_sine_facade",
  section = "tutorials",
  filename = "algorithmic-sine_facade.png",
  backend = :blender,
  view = iso_view(6, 0.5, 1.5, 12),
  build = () -> begin
    # A column row whose heights trace a sinusoid — division + map.
    slab(rectangular_path(xy(-0.5, -0.5), 13, 1.5), level(0))
    for i in 0:12
      h = 2 + 1.5 * sin(i * π / 6)
      column(xy(i, 0), 0, level(0), level(h))
    end
  end,
)

register_scene(
  id = "tutorials_algorithmic_staggered_grid",
  section = "tutorials",
  filename = "algorithmic-staggered_grid.png",
  backend = :blender,
  view = iso_view(7, 5, 1.5, 16),
  build = () -> begin
    # Staggered column grid — odd rows offset by half the spacing.
    slab(rectangular_path(xy(0, 0), 14, 10), level(0))
    for r in 0:5
      offset = isodd(r) ? 1.0 : 0.0
      for c in 0:6
        column(xy(offset + 2.0 * c, 2.0 * r), 0, level(0), level(3.0))
      end
    end
  end,
)

register_scene(
  id = "tutorials_algorithmic_spiral_tower",
  section = "tutorials",
  filename = "algorithmic-spiral_tower.png",
  backend = :blender,
  view = iso_view(0, 0, 9, 20),
  build = () -> begin
    # Six square slabs rotated around the central axis.
    for i in 0:5
      θ = i * π / 10
      cs = cs_from_o_phi(xyz(0, 0, 3.0 * i), θ)
      slab(rectangular_path(xy(-4, -4, cs), 8, 8), level(3.0 * i))
    end
  end,
)

# ==================================================================
# Tutorials / Wall graph — door + window on a segment
# ==================================================================

register_scene(
  id = "tutorials_wallgraph_with_door",
  section = "tutorials",
  filename = "wallgraph-with_door.png",
  backend = :blender,
  view = iso_view(3, 0, 1.5, 6),
  build = () -> begin
    g = wall_graph(level=level(0), height=3.0)
    j1 = junction!(g, xy(0, 0))
    j2 = junction!(g, xy(6, 0))
    s  = segment!(g, j1, j2)
    add_wall_door!(g, s; at=2.0)
    build_walls(g)
  end,
)

register_scene(
  id = "tutorials_wallgraph_with_window",
  section = "tutorials",
  filename = "wallgraph-with_window.png",
  backend = :blender,
  view = iso_view(3, 0, 1.5, 6),
  build = () -> begin
    g = wall_graph(level=level(0), height=3.0)
    j1 = junction!(g, xy(0, 0))
    j2 = junction!(g, xy(6, 0))
    s  = segment!(g, j1, j2)
    add_wall_window!(g, s; at=3.0, sill=1.0)
    build_walls(g)
  end,
)
