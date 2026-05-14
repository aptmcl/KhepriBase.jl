#=
Scenes for docs/src/tutorials/*.md - 3D progression shots.
=#

# ==================================================================
# Building tutorial - 4 stages from empty to two-storey
# ==================================================================

register_scene(
  id = "tutorials_building_step1_slab",
  section = "tutorials",
  filename = "building-step1_slab.png",
  backend = :blender,
  view = iso_view(5, 4, 0, 10),
  build = () -> begin
    slab(rectangular_path(xy(0, 0), 10, 8), level(0))
  end,
)

register_scene(
  id = "tutorials_building_step2_walls",
  section = "tutorials",
  filename = "building-step2_walls.png",
  backend = :blender,
  view = iso_view(5, 4, 1.5, 11),
  build = () -> begin
    slab(rectangular_path(xy(0, 0), 10, 8), level(0))
    wall(closed_polygonal_path([
        xy(0, 0), xy(10, 0), xy(10, 8), xy(0, 8)]),
      level(0), level(3.0))
  end,
)

# Step 3/4 originally placed a door + windows on the perimeter wall;
# KhepriBlender's subtract_ref path for that combination has a
# NativeRefs encoding bug, so the openings are dropped for now and
# the tutorial shot progresses from walls directly to the roof.

register_scene(
  id = "tutorials_building_step3_roof",
  section = "tutorials",
  filename = "building-step3_roof.png",
  backend = :blender,
  view = iso_view(5, 4, 1.8, 12),
  build = () -> begin
    slab(rectangular_path(xy(0, 0), 10, 8), level(0))
    wall(closed_polygonal_path([
        xy(0, 0), xy(10, 0), xy(10, 8), xy(0, 8)]),
      level(0), level(3.0))
    roof(rectangular_path(xy(-0.3, -0.3), 10.6, 8.6), level(3.0))
  end,
)

register_scene(
  id = "tutorials_building_step5_two_storey",
  section = "tutorials",
  filename = "building-step5_two_storey.png",
  backend = :blender,
  view = iso_view(5, 4, 3, 16),
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
  view = iso_view(4, 2, 1.5, 8),
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
  id = "tutorials_spaces_multi_storey",
  section = "tutorials",
  filename = "spaces-multi_storey.png",
  backend = :blender,
  view = iso_view(5, 2.5, 3, 15),
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
  id = "tutorials_wallgraph_step2_L",
  section = "tutorials",
  filename = "wallgraph-step2_L_shape.png",
  backend = :blender,
  view = iso_view(2.5, 2, 1.5, 6),
  build = () -> begin
    g = wall_graph(level=level(0), height=3.0)
    a = junction!(g, xy(0, 0))
    b = junction!(g, xy(5, 0))
    c = junction!(g, xy(5, 4))
    segment!(g, a, b); segment!(g, b, c)
    build_walls(g)
  end,
)

register_scene(
  id = "tutorials_wallgraph_step3_T",
  section = "tutorials",
  filename = "wallgraph-step3_T_junction.png",
  backend = :blender,
  view = iso_view(4, 2, 1.5, 7),
  build = () -> begin
    g = wall_graph(level=level(0), height=3.0)
    a = junction!(g, xy(0, 0))
    b = junction!(g, xy(8, 0))
    c = junction!(g, xy(4, 0))
    d = junction!(g, xy(4, 4))
    segment!(g, a, c); segment!(g, c, b); segment!(g, c, d)
    build_walls(g)
  end,
)

register_scene(
  id = "tutorials_wallgraph_step4_room",
  section = "tutorials",
  filename = "wallgraph-step4_closed_room.png",
  backend = :blender,
  view = iso_view(3, 2.5, 1.5, 7),
  build = () -> begin
    g = wall_graph(level=level(0), height=3.0)
    a = junction!(g, xy(0, 0))
    b = junction!(g, xy(6, 0))
    c = junction!(g, xy(6, 5))
    d = junction!(g, xy(0, 5))
    segment!(g, a, b); segment!(g, b, c)
    segment!(g, c, d); segment!(g, d, a)
    build_walls(g)
    slab(rectangular_path(xy(0, 0), 6, 5), level(0))
  end,
)

register_scene(
  id = "tutorials_wallgraph_step5_house",
  section = "tutorials",
  filename = "wallgraph-step5_house.png",
  backend = :blender,
  view = iso_view(5, 4, 1.5, 12),
  build = () -> begin
    g = wall_graph(level=level(0), height=3.0)
    # Perimeter of a 10x8 house with interior dividers
    p1 = junction!(g, xy(0, 0));  p2 = junction!(g, xy(10, 0))
    p3 = junction!(g, xy(10, 8)); p4 = junction!(g, xy(0, 8))
    mid_bot = junction!(g, xy(5, 0))
    mid_top = junction!(g, xy(5, 8))
    segment!(g, p1, mid_bot); segment!(g, mid_bot, p2)
    segment!(g, p2, p3); segment!(g, p3, mid_top)
    segment!(g, mid_top, p4); segment!(g, p4, p1)
    segment!(g, mid_bot, mid_top)
    build_walls(g)
    slab(rectangular_path(xy(0, 0), 10, 8), level(0))
  end,
)

# ==================================================================
# Rendering tutorial - four camera angles of same scene
# ==================================================================

#=
Door/window carve-outs hit a KhepriBlender `b_subtract_ref`
encoding bug (NativeRefs → Int32), so the rendering demo scene
ships as a walls-plus-roof without openings.
=#
_render_demo_scene() = begin
  slab(rectangular_path(xy(0, 0), 10, 8), level(0))
  wall(closed_polygonal_path([
      xy(0, 0), xy(10, 0), xy(10, 8), xy(0, 8)]),
    level(0), level(3.0))
  roof(rectangular_path(xy(-0.3, -0.3), 10.6, 8.6), level(3.0))
end

register_scene(
  id = "tutorials_rendering_iso",
  section = "tutorials",
  filename = "rendering-iso.png",
  backend = :blender,
  view = iso_view(5, 4, 1.8, 14),
  build = _render_demo_scene,
)

register_scene(
  id = "tutorials_rendering_top",
  section = "tutorials",
  filename = "rendering-top.png",
  backend = :blender,
  view = top_view(5, 4, 11),
  build = _render_demo_scene,
)

register_scene(
  id = "tutorials_rendering_front",
  section = "tutorials",
  filename = "rendering-front.png",
  backend = :blender,
  view = front_view(5, 4, 1.8, 11),
  build = _render_demo_scene,
)

register_scene(
  id = "tutorials_rendering_close",
  section = "tutorials",
  filename = "rendering-close.png",
  backend = :blender,
  view = (eye=xyz(12, -3, 3.0), target=xyz(2, 4, 1.5)),
  build = _render_demo_scene,
)

# ==================================================================
# Algorithmic tutorial - parametric structures
# ==================================================================

register_scene(
  id = "tutorials_algorithmic_column_grid",
  section = "tutorials",
  filename = "algorithmic-column_grid.png",
  backend = :blender,
  view = iso_view(12, 9, 2, 26),
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
  view = iso_view(5, 5, 9, 22),
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
  view = iso_view(0, 0, 2.25, 14),
  build = () -> begin
    slab(circular_path(xy(0, 0), 10), level(0))
    for θ in range(0, 2π; length=13)[1:end-1]
      column(xy(8*cos(θ), 8*sin(θ)), 0, level(0), level(4.5))
    end
  end,
)

# ==================================================================
# Isenberg - same building bottom-up vs top-down
# ==================================================================

#=
The bottom-up tutorial composes the building one named Space at a
time, with `polar_sector_path` as the only geometric primitive.  The
seven scenes below mirror the seven code chunks in the tutorial,
each rendering reveals exactly what the next chunk adds:

  step1  one polar sector                  the building block
  step2  inner band of n_rooms wedges      a fan around the courtyard
  step3  + the corridor arc                a continuous curved corridor
  step4  + outer band of n_rooms wedges    full double-loaded layout
  step5  + doors and exterior windows      visible openings
  step6  ground floor with lobby           projection wedge as one open hall
  step7  three storeys (full building)     the hero shot

Phases 1-6 share one isometric view targeted at floor level so the
reader watches geometry land in the same frame.  Phase 7 widens out
so the third storey stays in the picture.

The 270° sweep covers +x → +y → -x → -y; the missing wedge (south-east)
is where we put the camera, looking toward the courtyard.

Parameters are inlined in every closure rather than shared at module
scope so each scene is self-contained — copy-paste a single closure
into the REPL and it produces exactly the rendering shown in the
tutorial section it illustrates.  The values match the tutorial's
own `Parameters` block: r_inner=10, r_outer=25, n_rooms=18, etc.
=#

# Building footprint: outer radius 25, sweep 270°, so the plan
# spans roughly 50 × 50 m around the courtyard with the missing
# wedge in the south-east.  At r = 30 the camera ends up ~60 m
# from a ~50 m-diagonal subject, which crops the corners with
# Blender's default 50mm lens; r = 50/55 gives ~100/110 m of
# camera distance and leaves comfortable margin on all sides
# while still keeping the same isometric vantage point.
const _ISENBERG_BU_VIEW_FLOOR = iso_view(0, 0, 1.5, 45)
const _ISENBERG_BU_VIEW_BLDG  = iso_view(0, 0, 4.5, 50)

# Shared closure body.  Returns the upper-floor function bound to the
# parameters in `params`, with three flags that progressively turn on
# the corridor, outer band, and openings.  Each step scene calls this
# with a different combination of flags.
_isenberg_bu_add_upper_floor!(plan, params, theta_start, theta_end;
                              with_corridor, with_outer, with_openings) = begin
  dθ = (theta_end - theta_start) / params.n_rooms

  inner_rooms = [
    add_space(plan, "inner_$i",
      polar_sector_path(params.center, params.r_inner, params.r_corr_in,
                        theta_start + (i - 1) * dθ,
                        theta_start +  i      * dθ;
                        n_arc = params.n_arc);
      kind = :office)
    for i in 1:params.n_rooms]

  corridor = with_corridor ?
    add_space(plan, "corridor",
      polar_sector_path(params.center, params.r_corr_in, params.r_corr_out,
                        theta_start, theta_end;
                        n_arc = params.n_arc * params.n_rooms);
      kind = :corridor) : nothing

  outer_rooms = with_outer ? [
    add_space(plan, "outer_$i",
      polar_sector_path(params.center, params.r_corr_out, params.r_outer,
                        theta_start + (i - 1) * dθ,
                        theta_start +  i      * dθ;
                        n_arc = params.n_arc);
      kind = :office)
    for i in 1:params.n_rooms] : []

  if with_openings && corridor !== nothing
    for r in inner_rooms
      add_door(plan, r, corridor)
    end
    for r in outer_rooms
      add_door(plan, r, corridor)
    end
    for (i, r) in enumerate(outer_rooms)
      θ_mid = theta_start + (i - 0.5) * dθ
      add_window(plan, r, :exterior,
        loc = params.center + vpol(params.r_outer, θ_mid),
        family = window_family(width = 1.4, height = 1.5))
    end
  end

  (inner_rooms, corridor, outer_rooms)
end

# Default parameter bundle; matches the tutorial's `Parameters` block.
#
# `n_arc` is forced > 0 here even though the tutorial documents
# `n_arc = 0` (arc-native) as the production default.  Arc-native
# `polar_sector_path` returns a `ClosedPathSequence` whose `ArcPath`
# components reach KhepriBlender's wall and slab realizers via
# `b_wall_no_openings` / `b_slab`, where the discretised offsets and
# face polylines don't agree on vertex counts and the curved walls
# render as degenerate strips while the slabs render as flat fans.
# Polygonal discretisation (`n_arc = 12`) sidesteps that and produces
# a visually smooth result at this size.  The tutorial text still
# documents `n_arc = 0` as the production default — BIM backends like
# AutoCAD/Revit emit native curved walls from arc-native paths.
_isenberg_bu_params() = (
  center        = u0(),
  r_inner       = 10.0,
  r_outer       = 25.0,
  arc_start     = 0.0,
  projection    = π,
  arc_end       = 3π/2,
  n_rooms       = 18,
  corridor_span = 2.0,
  floor_h       = 3.0,
  n_arc         = 12,
  r_corr_in     = (10.0 + 25.0)/2 - 2.0/2,
  r_corr_out    = (10.0 + 25.0)/2 + 2.0/2,
)

# ---- Phase 1: one polar sector ----
register_scene(
  id = "tutorials_isenberg_bottom_up_step1_wedge",
  section = "tutorials",
  filename = "isenberg-bottom_up-step1_wedge.png",
  backend = :blender,
  view = _ISENBERG_BU_VIEW_FLOOR,
  build = () -> begin
    p = _isenberg_bu_params()
    dθ = (p.arc_end - p.arc_start) / p.n_rooms
    plan = floor_plan(height = p.floor_h,
                      wall_family = wall_family(thickness = 0.2),
                      slab_family = slab_family(thickness = 0.4))
    add_space(plan, "wedge",
      polar_sector_path(p.center, p.r_inner, p.r_corr_in,
                        p.arc_start, p.arc_start + dθ;
                        n_arc = p.n_arc);
      kind = :office)
    build(plan)
  end,
)

# ---- Phase 2: inner band of n_rooms wedges ----
register_scene(
  id = "tutorials_isenberg_bottom_up_step2_inner_band",
  section = "tutorials",
  filename = "isenberg-bottom_up-step2_inner_band.png",
  backend = :blender,
  view = _ISENBERG_BU_VIEW_FLOOR,
  build = () -> begin
    p = _isenberg_bu_params()
    plan = floor_plan(height = p.floor_h,
                      wall_family = wall_family(thickness = 0.2),
                      slab_family = slab_family(thickness = 0.4))
    _isenberg_bu_add_upper_floor!(plan, p, p.arc_start, p.arc_end;
      with_corridor = false, with_outer = false, with_openings = false)
    build(plan)
  end,
)

# ---- Phase 3: + corridor band ----
register_scene(
  id = "tutorials_isenberg_bottom_up_step3_corridor",
  section = "tutorials",
  filename = "isenberg-bottom_up-step3_corridor.png",
  backend = :blender,
  view = _ISENBERG_BU_VIEW_FLOOR,
  build = () -> begin
    p = _isenberg_bu_params()
    plan = floor_plan(height = p.floor_h,
                      wall_family = wall_family(thickness = 0.2),
                      slab_family = slab_family(thickness = 0.4))
    _isenberg_bu_add_upper_floor!(plan, p, p.arc_start, p.arc_end;
      with_corridor = true, with_outer = false, with_openings = false)
    build(plan)
  end,
)

# ---- Phase 4: + outer band of rooms ----
register_scene(
  id = "tutorials_isenberg_bottom_up_step4_outer_band",
  section = "tutorials",
  filename = "isenberg-bottom_up-step4_outer_band.png",
  backend = :blender,
  view = _ISENBERG_BU_VIEW_FLOOR,
  build = () -> begin
    p = _isenberg_bu_params()
    plan = floor_plan(height = p.floor_h,
                      wall_family = wall_family(thickness = 0.2),
                      slab_family = slab_family(thickness = 0.4))
    _isenberg_bu_add_upper_floor!(plan, p, p.arc_start, p.arc_end;
      with_corridor = true, with_outer = true, with_openings = false)
    build(plan)
  end,
)

# ---- Phase 5: + doors and exterior windows ----
register_scene(
  id = "tutorials_isenberg_bottom_up_step5_openings",
  section = "tutorials",
  filename = "isenberg-bottom_up-step5_openings.png",
  backend = :blender,
  view = _ISENBERG_BU_VIEW_FLOOR,
  build = () -> begin
    p = _isenberg_bu_params()
    plan = floor_plan(height = p.floor_h,
                      wall_family = wall_family(thickness = 0.2),
                      slab_family = slab_family(thickness = 0.4))
    _isenberg_bu_add_upper_floor!(plan, p, p.arc_start, p.arc_end;
      with_corridor = true, with_outer = true, with_openings = true)
    build(plan)
  end,
)

# ---- Phase 6: ground floor with lobby ----
register_scene(
  id = "tutorials_isenberg_bottom_up_step6_ground_floor",
  section = "tutorials",
  filename = "isenberg-bottom_up-step6_ground_floor.png",
  backend = :blender,
  view = _ISENBERG_BU_VIEW_FLOOR,
  build = () -> begin
    p = _isenberg_bu_params()
    plan = floor_plan(height = p.floor_h,
                      wall_family = wall_family(thickness = 0.2),
                      slab_family = slab_family(thickness = 0.4))
    # Semicircular half: full upper-floor pattern with openings
    _isenberg_bu_add_upper_floor!(plan, p, p.arc_start, p.projection;
      with_corridor = true, with_outer = true, with_openings = true)
    # Projection wedge: one big lobby
    lobby = add_space(plan, "lobby",
      polar_sector_path(p.center, p.r_inner, p.r_outer,
                        p.projection, p.arc_end;
                        n_arc = p.n_arc * 6);
      kind = :lobby)
    θ_door = (p.projection + p.arc_end) / 2
    add_door(plan, lobby, :exterior,
             loc = p.center + vpol(p.r_outer, θ_door))
    for t in (0.25, 0.5, 0.75)
      θ = p.projection + t * (p.arc_end - p.projection)
      add_window(plan, lobby, :exterior,
                 loc = p.center + vpol(p.r_outer, θ),
                 family = window_family(width = 1.8, height = 2.4))
    end
    build(plan)
  end,
)

# ---- Phase 7: three storeys (the hero shot) ----
_isenberg_bu_full_building = () -> begin
  p = _isenberg_bu_params()
  n_floors = 3
  plan = floor_plan(height = p.floor_h,
                    wall_family = wall_family(thickness = 0.2),
                    slab_family = slab_family(thickness = 0.4))
  # Ground floor
  _isenberg_bu_add_upper_floor!(plan, p, p.arc_start, p.projection;
    with_corridor = true, with_outer = true, with_openings = true)
  lobby = add_space(plan, "lobby",
    polar_sector_path(p.center, p.r_inner, p.r_outer,
                      p.projection, p.arc_end;
                      n_arc = p.n_arc * 6);
    kind = :lobby)
  θ_door = (p.projection + p.arc_end) / 2
  add_door(plan, lobby, :exterior,
           loc = p.center + vpol(p.r_outer, θ_door))
  for t in (0.25, 0.5, 0.75)
    θ = p.projection + t * (p.arc_end - p.projection)
    add_window(plan, lobby, :exterior,
               loc = p.center + vpol(p.r_outer, θ),
               family = window_family(width = 1.8, height = 2.4))
  end
  # Upper floors
  for _ in 2:n_floors
    add_storey!(plan; height = p.floor_h)
    _isenberg_bu_add_upper_floor!(plan, p, p.arc_start, p.arc_end;
      with_corridor = true, with_outer = true, with_openings = true)
  end
  build(plan)
end

register_scene(
  id = "tutorials_isenberg_bottom_up_step7_three_storeys",
  section = "tutorials",
  filename = "isenberg-bottom_up-step7_three_storeys.png",
  backend = :blender,
  view = _ISENBERG_BU_VIEW_BLDG,
  build = _isenberg_bu_full_building,
)

# Hero shot at the top of the tutorial: same as step 7, kept under the
# legacy filename so existing references in the built docs still resolve.
register_scene(
  id = "tutorials_isenberg_bottom_up",
  section = "tutorials",
  filename = "isenberg-bottom_up.png",
  backend = :blender,
  view = _ISENBERG_BU_VIEW_BLDG,
  build = _isenberg_bu_full_building,
)

register_scene(
  id = "tutorials_isenberg_top_down",
  section = "tutorials",
  filename = "isenberg-top_down.png",
  backend = :blender,
  view = iso_view(6, 5, 1.5, 13),
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
