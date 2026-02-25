#=
Scene 1 — Interior Design Vignette
Reverse-engineered from Radiance reference scene (Rendering with Radiance, Chapter 8).

A furnished interior with table, stools, vase with rose, mirror, lamp shades,
downlights, sconces, checkerboard tiles, and baseboard.

Source files: scene1/scene1/{material1.rad, light1.rad, scene1.vf, scene1.all}
             scene1/scene1/lib/{wall_ceiling, table_1, stool_1, lamp_shade_1,
             tile_x4_1, leaf_1, leaves_1, flower_1, vase_1, mirror_1,
             downlight_a1, downlight_b1, fixture_1, baseboard_1}.rad

Known limitations:
- genbox edge radius (-r .05) → sharp-edged boxes (Khepri box has no fillet)
- genbox beveled edges (-b .025) → sharp-edged boxes
- genprism tapered stool legs → prism approximation
- Radiance spotlight with cone angle → Khepri spotlight approximation
- xform mirror (-mx) → manually mirrored leaf polygons
=#

# Change this line to switch backends:
# using KhepriRadiance
# using KhepriMitsuba

## ─── Materials ─────────────────────────────────────────────────────────────

# Light sources
# "void light lumens 0 0 3 200 200 200" — 55W tungsten spherical lamp
lumens_mat = standard_material(
  name="lumens",
  base_color=rgba(1.0, 1.0, 1.0, 1.0),
  emissive=rgba(200.0, 200.0, 200.0, 1.0),
  data=BackendParameter(RAD => radiance_light_material("lumens", red=200, green=200, blue=200)))

# "void light diffuse 0 0 3 2.6 1.7 .6" — warm decorative panel light
# (light1.rad overrides material1.rad values)
diffuse_light_mat = standard_material(
  name="diffuse",
  base_color=rgba(1.0, 0.65, 0.23, 1.0),
  emissive=rgba(2.6, 1.7, 0.6, 1.0),
  data=BackendParameter(RAD => radiance_light_material("diffuse", red=2.6, green=1.7, blue=0.6)))

# "void light lumensf 0 0 3 15.794 10.413 3.993" — warm globe behind panel
lumensf_mat = standard_material(
  name="lumensf",
  base_color=rgba(1.0, 0.66, 0.25, 1.0),
  emissive=rgba(15.794, 10.413, 3.993, 1.0),
  data=BackendParameter(RAD => radiance_light_material("lumensf", red=15.794, green=10.413, blue=3.993)))

# "void light lightring 0 0 3 229.130493 151.068357 57.0"
lightring_mat = standard_material(
  name="lightring",
  base_color=rgba(1.0, 0.66, 0.25, 1.0),
  emissive=rgba(229.130493, 151.068357, 57.0, 1.0),
  data=BackendParameter(RAD => radiance_light_material("lightring", red=229.130493, green=151.068357, blue=57.0)))

# Plastic materials
# "void plastic brown_satin_paint 0 0 5 0.2 0.1 0.01 0.05 0.03"
brown_satin_paint_mat = standard_material(
  name="brown_satin_paint",
  base_color=rgba(0.2, 0.1, 0.01, 1.0),
  roughness=0.03,
  data=BackendParameter(RAD => radiance_plastic_material("brown_satin_paint", red=0.2, green=0.1, blue=0.01, specularity=0.05, roughness=0.03)))

# "void plastic red_porcelain 0 0 5 .7 .05 .05 .05 .02"
red_porcelain_mat = standard_material(
  name="red_porcelain",
  base_color=rgba(0.7, 0.05, 0.05, 1.0),
  roughness=0.02,
  data=BackendParameter(RAD => radiance_plastic_material("red_porcelain", red=0.7, green=0.05, blue=0.05, specularity=0.05, roughness=0.02)))

# "void plastic cream_porcelain 0 0 5 .6 .45 .3 .05 .02"
cream_porcelain_mat = standard_material(
  name="cream_porcelain",
  base_color=rgba(0.6, 0.45, 0.3, 1.0),
  roughness=0.02,
  data=BackendParameter(RAD => radiance_plastic_material("cream_porcelain", red=0.6, green=0.45, blue=0.3, specularity=0.05, roughness=0.02)))

# "void plastic green_gloss 0 0 5 .1 .6 .2 .1 .02"
green_gloss_mat = standard_material(
  name="green_gloss",
  base_color=rgba(0.1, 0.6, 0.2, 1.0),
  roughness=0.02,
  data=BackendParameter(RAD => radiance_plastic_material("green_gloss", red=0.1, green=0.6, blue=0.2, specularity=0.1, roughness=0.02)))

# "void plastic white_matte 0 0 5 .8 .8 .8 0 0"
white_matte_mat = standard_material(
  name="white_matte",
  base_color=rgba(0.8, 0.8, 0.8, 1.0),
  roughness=1.0,
  data=BackendParameter(RAD => radiance_plastic_material("white_matte", gray=0.8)))

# "void plastic red_tile 0 0 5 .5 .01 .05 0.02 0.03"
red_tile_mat = standard_material(
  name="red_tile",
  base_color=rgba(0.5, 0.01, 0.05, 1.0),
  roughness=0.03,
  data=BackendParameter(RAD => radiance_plastic_material("red_tile", red=0.5, green=0.01, blue=0.05, specularity=0.02, roughness=0.03)))

# "void plastic white_tile 0 0 5 .6 .5 .3 0.02 0.03"
white_tile_mat = standard_material(
  name="white_tile",
  base_color=rgba(0.6, 0.5, 0.3, 1.0),
  roughness=0.03,
  data=BackendParameter(RAD => radiance_plastic_material("white_tile", red=0.6, green=0.5, blue=0.3, specularity=0.02, roughness=0.03)))

# "void plastic red1 0 0 5 1 0 0 0 0"
red1_mat = standard_material(
  name="red1",
  base_color=rgba(1.0, 0.0, 0.0, 1.0),
  roughness=1.0,
  data=BackendParameter(RAD => radiance_plastic_material("red1", red=1.0, green=0.0, blue=0.0)))

# "void plastic green3 0 0 5 0 .8 .1 0 0"
green3_mat = standard_material(
  name="green3",
  base_color=rgba(0.0, 0.8, 0.1, 1.0),
  roughness=1.0,
  data=BackendParameter(RAD => radiance_plastic_material("green3", red=0.0, green=0.8, blue=0.1)))

# "void plastic green1 0 0 5 0 1 0 0 0"
green1_mat = standard_material(
  name="green1",
  base_color=rgba(0.0, 1.0, 0.0, 1.0),
  roughness=1.0,
  data=BackendParameter(RAD => radiance_plastic_material("green1", red=0.0, green=1.0, blue=0.0)))

# "void mirror mirror_glass 0 0 3 .8 .8 .8"
mirror_glass_mat = standard_material(
  name="mirror_glass",
  base_color=rgba(0.8, 0.8, 0.8, 1.0),
  metallic=1.0,
  roughness=0.0,
  data=BackendParameter(RAD => RadianceMaterial("mirror_glass", "mirror", 0.8, 0.8, 0.8, nothing, nothing, nothing, nothing)))

# "void metal antique_copper 0 0 5 .136 .102 .083 .3 .2"
antique_copper_mat = standard_material(
  name="antique_copper",
  base_color=rgba(0.136, 0.102, 0.083, 1.0),
  metallic=1.0,
  roughness=0.2,
  data=BackendParameter(RAD => radiance_metal_material("antique_copper", red=0.136, green=0.102, blue=0.083, specularity=0.3, roughness=0.2)))

# "void plastic frame_brown 0 0 5 .2 .05 .01 0 0"
frame_brown_mat = standard_material(
  name="frame_brown",
  base_color=rgba(0.2, 0.05, 0.01, 1.0),
  roughness=1.0,
  data=BackendParameter(RAD => radiance_plastic_material("frame_brown", red=0.2, green=0.05, blue=0.01)))


## ─── Wall and Ceiling ──────────────────────────────────────────────────────
# "!genbox white_matte wall 8 .2 8 | xform -t -4 4 0"
box(xyz(-4, 4, 0), 8, 0.2, 8, material=white_matte_mat)

# "!genbox white_matte ceiling 8 8 .2 | xform -t -4 -4 8"
box(xyz(-4, -4, 8), 8, 8, 0.2, material=white_matte_mat)


## ─── Table ─────────────────────────────────────────────────────────────────
# Table top: 2.5 × 2.5 × 0.15 at z=3.15 (edge radius ignored)
# "!genbox brown_satin_paint table_top 2.5 2.5 0.15 -r .05 | xform -t -1.25 -1.25 3.15"
box(xyz(-1.25, -1.25, 3.15), 2.5, 2.5, 0.15, material=brown_satin_paint_mat)

# Table base: 1.2 × 1.2 × 0.15
# "!genbox brown_satin_paint table_base 1.2 1.2 0.15 | xform -t -.6 -.6 0"
box(xyz(-0.6, -0.6, 0), 1.2, 1.2, 0.15, material=brown_satin_paint_mat)

# Table legs: 0.15 × 0.15 × 3.15 each
# SW: xform -t -.6 -.6 0
box(xyz(-0.6, -0.6, 0), 0.15, 0.15, 3.15, material=brown_satin_paint_mat)
# SE: xform -t .45 -.6 0
box(xyz(0.45, -0.6, 0), 0.15, 0.15, 3.15, material=brown_satin_paint_mat)
# NE: xform -t .45 .45 0
box(xyz(0.45, 0.45, 0), 0.15, 0.15, 3.15, material=brown_satin_paint_mat)
# NW: xform -t -.6 .45 0
box(xyz(-0.6, 0.45, 0), 0.15, 0.15, 3.15, material=brown_satin_paint_mat)


## ─── Stools ────────────────────────────────────────────────────────────────
# Helper: creates a single stool at origin, returns nothing (shapes are side effects)
function make_stool(origin, rot_z=0)
  let cs = rot_z == 0 ?
           loc_from_o_vx_vy(origin, vx(1), vy(1)) :
           loc_from_o_vx_vy(origin,
             vxy(cos(deg2rad(rot_z)), sin(deg2rad(rot_z))),
             vxy(-sin(deg2rad(rot_z)), cos(deg2rad(rot_z))))
    # Seat top polygon at z=2.6
    surface_polygon(
      cs + vxyz(-0.75, -0.75, 2.6), cs + vxyz(0.75, -0.75, 2.6),
      cs + vxyz(0.75, 0.75, 2.6), cs + vxyz(-0.75, 0.75, 2.6),
      material=brown_satin_paint_mat)
    # Side skirts (z=2.45 to z=2.6)
    # North side (y=+0.75)
    surface_polygon(
      cs + vxyz(-0.75, 0.75, 2.45), cs + vxyz(-0.75, 0.75, 2.6),
      cs + vxyz(0.75, 0.75, 2.6), cs + vxyz(0.75, 0.75, 2.45),
      material=brown_satin_paint_mat)
    # South side (y=-0.75)
    surface_polygon(
      cs + vxyz(-0.75, -0.75, 2.45), cs + vxyz(0.75, -0.75, 2.45),
      cs + vxyz(0.75, -0.75, 2.6), cs + vxyz(-0.75, -0.75, 2.6),
      material=brown_satin_paint_mat)
    # East side (x=-0.75)
    surface_polygon(
      cs + vxyz(-0.75, -0.75, 2.45), cs + vxyz(-0.75, -0.75, 2.6),
      cs + vxyz(-0.75, 0.75, 2.6), cs + vxyz(-0.75, 0.75, 2.45),
      material=brown_satin_paint_mat)
    # West side (x=+0.75)
    surface_polygon(
      cs + vxyz(0.75, -0.75, 2.45), cs + vxyz(0.75, 0.75, 2.45),
      cs + vxyz(0.75, 0.75, 2.6), cs + vxyz(0.75, -0.75, 2.6),
      material=brown_satin_paint_mat)
    # Tapered legs as prisms (4 vertices each, extruded 2.5 in z)
    # SW leg
    prism([cs + vxy(-0.7, -0.7), cs + vxy(-0.6, -0.7),
           cs + vxy(-0.6, -0.6), cs + vxy(-0.7, -0.6)],
          2.5, brown_satin_paint_mat)
    # SE leg
    prism([cs + vxy(0.7, -0.7), cs + vxy(0.7, -0.6),
           cs + vxy(0.6, -0.6), cs + vxy(0.6, -0.7)],
          2.5, brown_satin_paint_mat)
    # NE leg
    prism([cs + vxy(0.7, 0.7), cs + vxy(0.6, 0.7),
           cs + vxy(0.6, 0.6), cs + vxy(0.7, 0.6)],
          2.5, brown_satin_paint_mat)
    # NW leg
    prism([cs + vxy(-0.7, 0.7), cs + vxy(-0.7, 0.6),
           cs + vxy(-0.6, 0.6), cs + vxy(-0.6, 0.7)],
          2.5, brown_satin_paint_mat)
  end
end

# Stool 1: "!xform -n stool1 -t -2.0 0 0 lib/stool_1.rad"
make_stool(xyz(-2.0, 0, 0))
# Stool 2: "!xform -n stool2 -rz 30 -t 2.5 0 0 lib/stool_1.rad"
make_stool(xyz(2.5, 0, 0), 30)


## ─── Vase ──────────────────────────────────────────────────────────────────
# Vase sits at table origin (0, 0). Assembled from vase_1.rad.
# Outer (red porcelain):
#   cone vase_a: from z=3.3 to z=3.4, r_bot=0.20, r_top=0.18
cone_frustum(xyz(0, 0, 3.3), 0.20, 0.1, 0.18, material=red_porcelain_mat)
#   sphere vase_b: center (0, 0, 3.6), radius 0.3
sphere(xyz(0, 0, 3.6), 0.3, material=red_porcelain_mat)
#   cylinder vase_c: from z=3.8 to z=4.6, radius 0.05
cylinder(xyz(0, 0, 3.8), 0.05, 0.8, material=red_porcelain_mat)
#   cone vase_d: from z=4.6 to z=4.625, r_bot=0.05, r_top=0.08
cone_frustum(xyz(0, 0, 4.6), 0.05, 0.025, 0.08, material=red_porcelain_mat)

# Inner (cream porcelain):
#   ring vase_e: at z=4.625, normal up, inner=0.08, outer=0.06
#   (This is the lip ring — actually inner < outer reversed in Radiance:
#    ring vase_e 0 0 8 0 0 4.625 0 0 1 .08 .06 → inner=0.06, outer=0.08)
surface_ring(xyz(0, 0, 4.625), 0.06, 0.08, material=cream_porcelain_mat)
#   cone vase_f: from z=4.625 to z=4.6, r_bot=0.06, r_top=0.03
cone_frustum(xyz(0, 0, 4.625), 0.06, -0.025, 0.03, material=cream_porcelain_mat)
#   cylinder vase_g: from z=3.8 to z=4.6, radius 0.03
cylinder(xyz(0, 0, 3.8), 0.03, 0.8, material=cream_porcelain_mat)


## ─── Rose ──────────────────────────────────────────────────────────────────
# Rose positioned by: "!xform -n rose -s 2.5 -rz -30 -ry 2 -t 0 0 5.05 lib/flower_1.rad"
# Scale=2.5, rz=-30°, ry=2°, translate to (0, 0, 5.05)
# For simplicity we apply the main transform (scale + translate) and approximate angles.
let s = 2.5,
    rz = deg2rad(-30),
    rose_origin = xyz(0, 0, 5.05)
  # Rose bud: cone from z=0 to z=0.075*s, r_bot=0.02*s, r_top=0
  cone(rose_origin, 0.02*s, 0.075*s, material=red1_mat)
  # Rose ball: sphere at origin, r=0.02*s
  sphere(rose_origin, 0.02*s, material=red1_mat)
  # Stem cone: from z=-0.01*s to z=-0.03*s, r_bot=0.02*s, r_top=0.004*s
  cone_frustum(rose_origin + vz(-0.01*s), 0.02*s, -0.02*s, 0.004*s, material=green3_mat)
  # Stem cylinder: from z=-0.03*s to z=-0.4*s, r=0.004*s
  cylinder(rose_origin + vz(-0.03*s), 0.004*s, -0.37*s, material=green3_mat)

  # Leaves — simplified as small polygon patches along the stem
  # Original: 9-vertex leaf polygon, mirrored and arrayed with xform transforms
  # Each leaf half (from leaf_1.rad):
  #   (0,0,0), (-.01,.1,0), (-.1,.2,0), (-.09,.2,0), (-.15,.33,0),
  #   (-.12,.32,0), (-.11,.52,0), (-.09,.51,0), (0,.62,0)
  # Two branches of leaves at different positions along stem.
  # We approximate with simplified leaf polygons at key positions.

  # Branch 1: scale=1*2.5, rx=45, at z=-0.075*s from rose_origin
  let leaf_s = 0.12 * s,
      branch1_z = rose_origin + vz(-0.075 * s)
    # Left leaf
    surface_polygon(
      branch1_z,
      branch1_z + vxyz(-0.01*leaf_s, 0.1*leaf_s, 0),
      branch1_z + vxyz(-0.15*leaf_s, 0.33*leaf_s, 0),
      branch1_z + vxyz(-0.11*leaf_s, 0.52*leaf_s, 0),
      branch1_z + vxyz(0, 0.62*leaf_s, 0),
      material=green1_mat)
    # Right leaf (mirrored)
    surface_polygon(
      branch1_z,
      branch1_z + vxyz(0.01*leaf_s, 0.1*leaf_s, 0),
      branch1_z + vxyz(0.15*leaf_s, 0.33*leaf_s, 0),
      branch1_z + vxyz(0.11*leaf_s, 0.52*leaf_s, 0),
      branch1_z + vxyz(0, 0.62*leaf_s, 0),
      material=green1_mat)
  end

  # Branch 2: scale=0.7*2.5, rx=-65, rz=140, at z=-0.125*s from rose_origin
  let leaf_s = 0.15 * s * 0.7,
      branch2_z = rose_origin + vz(-0.125 * s)
    surface_polygon(
      branch2_z,
      branch2_z + vxyz(-0.01*leaf_s, 0.1*leaf_s, 0),
      branch2_z + vxyz(-0.15*leaf_s, 0.33*leaf_s, 0),
      branch2_z + vxyz(-0.11*leaf_s, 0.52*leaf_s, 0),
      branch2_z + vxyz(0, 0.62*leaf_s, 0),
      material=green1_mat)
    surface_polygon(
      branch2_z,
      branch2_z + vxyz(0.01*leaf_s, 0.1*leaf_s, 0),
      branch2_z + vxyz(0.15*leaf_s, 0.33*leaf_s, 0),
      branch2_z + vxyz(0.11*leaf_s, 0.52*leaf_s, 0),
      branch2_z + vxyz(0, 0.62*leaf_s, 0),
      material=green1_mat)
  end
end


## ─── Mirror ────────────────────────────────────────────────────────────────
# "!xform -n mirror -rx 5 -t 0 3.7 5 lib/mirror_1.rad"
# Mirror ring: radius 1.25, facing -y direction (tilted 5° forward by rx=5)
# mirror_glass ring mirror: at origin, normal=(0,-1,0), inner=0, outer=1.25
let mirror_cs = loc_from_o_vx_vy(
      xyz(0, 3.7, 5),
      vx(1),
      vxyz(0, cos(deg2rad(5)), sin(deg2rad(5))))
  # Mirror face: ring with mirror material
  surface_circle(mirror_cs, 1.25, material=mirror_glass_mat)
  # Frame: copper cylinder r=1.35 from y=0 to y=-0.1
  cylinder(mirror_cs, 1.35, loc_from_o_vx_vy(
    xyz(0, 3.7 - 0.1*cos(deg2rad(5)), 5 - 0.1*sin(deg2rad(5))),
    vx(1), vxyz(0, cos(deg2rad(5)), sin(deg2rad(5)))),
    material=antique_copper_mat)
  # Inner frame tube: r=1.25
  cylinder(mirror_cs, 1.25, loc_from_o_vx_vy(
    xyz(0, 3.7 - 0.1*cos(deg2rad(5)), 5 - 0.1*sin(deg2rad(5))),
    vx(1), vxyz(0, cos(deg2rad(5)), sin(deg2rad(5)))),
    material=antique_copper_mat)
  # Back ring
  surface_ring(
    loc_from_o_vx_vy(
      xyz(0, 3.7 - 0.1*cos(deg2rad(5)), 5 - 0.1*sin(deg2rad(5))),
      vx(1), vxyz(0, cos(deg2rad(5)), sin(deg2rad(5)))),
    1.25, 1.35, material=antique_copper_mat)
end


## ─── Lamp Shades ───────────────────────────────────────────────────────────
# Lamp shades were commented out in scene1.all ("reserve canopy lights for a different project")
# but we include them for completeness, positioned at (-3, 0, 6.5) and (3, 0, 6.5).
function make_lamp_shade(origin)
  # Outer shade (green_gloss):
  #   cone shade_out_1: z=0→0.2, r=0.75→0.7
  cone_frustum(origin, 0.75, 0.2, 0.70, material=green_gloss_mat)
  #   cone shade_out_2: z=0.2→0.4, r=0.7→0.3
  cone_frustum(origin + vz(0.2), 0.70, 0.2, 0.30, material=green_gloss_mat)
  #   cone shade_out_3: z=0.4→0.7, r=0.3→0.2
  cone_frustum(origin + vz(0.4), 0.30, 0.3, 0.20, material=green_gloss_mat)
  #   ring shade_out_4: z=0.7, inner=0.2, outer=0 → cap (solid disk)
  surface_circle(origin + vz(0.7), 0.2, material=green_gloss_mat)

  # Inner shade (white_matte):
  #   ring shade_in_0: z=0, normal down, inner=0.75, outer=0.725
  #   (bottom ring between outer and inner shells)
  surface_ring(origin, 0.725, 0.75, material=white_matte_mat)
  #   cone shade_in_1: z=0→0.2, r=0.725→0.675
  cone_frustum(origin, 0.725, 0.2, 0.675, material=white_matte_mat)
  #   cone shade_in_2: z=0.2→0.4, r=0.675→0.275
  cone_frustum(origin + vz(0.2), 0.675, 0.2, 0.275, material=white_matte_mat)
  #   cone shade_in_3: z=0.4→0.675, r=0.275→0.175
  cone_frustum(origin + vz(0.4), 0.275, 0.275, 0.175, material=white_matte_mat)
  #   ring shade_in_4: z=0.675, inner=0.175, outer=0
  surface_circle(origin + vz(0.675), 0.175, material=white_matte_mat)

  # Light bulb: sphere at z=0.25, r=0.25
  sphere(origin + vz(0.25), 0.25, material=lumens_mat)
end

# Commented out in original scene1.all:
# make_lamp_shade(xyz(-3, 0, 6.5))
# make_lamp_shade(xyz(3, 0, 6.5))


## ─── Floor Tiles ───────────────────────────────────────────────────────────
# "!xform -n floor -t -4 -4 0 -a 4 -t 2 0 0 -a 4 -t 0 2 0 -i 1 lib/tile_x4_1.rad"
# 4×4 array of 4'×4' tile blocks → 16'×16' floor (but room is 8×8, so 4×4 array
# with tile_x4 = 2×2 tiles each 1'×1')
# tile_x4_1.rad: 4 tiles in checkerboard pattern, each 1×1×0.1 with beveled edges
# Total floor: origin at (-4, -4, 0), tiling 4 blocks of 2' each in x and y
for bx in 0:3
  for by in 0:3
    let base_x = -4 + bx * 2,
        base_y = -4 + by * 2
      # tile_x4_1.rad: 2×2 checkerboard of 1'×1'×0.1 tiles at z=-0.1
      # (0,0): red, (1,0): white, (1,1): red, (0,1): white
      box(xyz(base_x, base_y, -0.1), 1, 1, 0.1, material=red_tile_mat)
      box(xyz(base_x + 1, base_y, -0.1), 1, 1, 0.1, material=white_tile_mat)
      box(xyz(base_x + 1, base_y + 1, -0.1), 1, 1, 0.1, material=red_tile_mat)
      box(xyz(base_x, base_y + 1, -0.1), 1, 1, 0.1, material=white_tile_mat)
    end
  end
end


## ─── Baseboard ─────────────────────────────────────────────────────────────
# "!genprism frame_brown baseboard1 7 0 0 0 -.1 .05 -.1 .1 -.08 .4 -.08 .5 -.04 .5 0
#   -l 0 0 8 | xform -ry -90 -t 8 0 0"
# 7-vertex profile extruded 8' in z direction, then rotated -90° around y and translated.
# After xform -ry -90 -t 8 0 0: the prism runs along the x-axis from x=8 to x=0,
# at y=4 (the wall face), with the profile in the y-z plane.
# Profile in genprism local coords (x,y): (0,0), (0,-.1), (.05,-.1), (.1,-.08), (.4,-.08), (.5,-.04), (.5,0)
# After -ry -90 -t 8 0 0, genprism -l 0 0 8 extrusion becomes along -x direction
# The baseboard runs along the north wall at y=4.
# genprism coordinates (x,y) map to (z,y) in world after -ry -90,
# then extrusion length 8 maps to x from 8 to 0.
# Actually: -ry -90 rotates the z-axis extrusion to point in -x, and -t 8 0 0 offsets.
# Profile vertices (z_local, y_local) → world (x, y_offset, z) at wall base:
let wall_y = 4.0,
    # The baseboard profile in the y-z plane (looking from +x toward -x)
    profile = [
      xyz(-4, wall_y, 0),       # (0, 0)
      xyz(-4, wall_y - 0.1, 0),  # (0, -0.1)
      xyz(-4, wall_y - 0.1, 0.05),  # (0.05, -0.1)
      xyz(-4, wall_y - 0.08, 0.1),  # (0.1, -0.08)
      xyz(-4, wall_y - 0.08, 0.4),  # (0.4, -0.08)
      xyz(-4, wall_y - 0.04, 0.5),  # (0.5, -0.04)
      xyz(-4, wall_y, 0.5)]      # (0.5, 0)
  prism(profile, vx(8), frame_brown_mat)
end


## ─── Wall Sconces ──────────────────────────────────────────────────────────
# "!xform -n fix -ry 45 -t -2.75 4 6.5 lib/fixture_1.rad"
# "!xform -n fix -ry 45 -t  2.75 4 6.5 lib/fixture_1.rad"
# fixture_1.rad: 4 cylindrical posts, diffuse polygon lens, warm light bulb
function make_wall_sconce(origin)
  # 4 corner posts (cylinders r=0.025)
  # Posts run from (x, 0, z) to (x, -0.45, z) in local coords
  # After -ry 45 rotation, the fixture is angled toward the viewer
  # For simplicity we place posts perpendicular to wall (in -y direction)
  for (dx, dz) in [(-0.45, -0.45), (0.45, -0.45), (0.45, 0.45), (-0.45, 0.45)]
    cylinder(origin + vxyz(dx, 0, dz), 0.025,
             origin + vxyz(dx, -0.45, dz),
             material=red_porcelain_mat)
  end
  # Diffuse light panel: polygon at y=-0.5
  surface_polygon(
    origin + vxyz(-0.5, -0.5, -0.5),
    origin + vxyz(0.5, -0.5, -0.5),
    origin + vxyz(0.5, -0.5, 0.5),
    origin + vxyz(-0.5, -0.5, 0.5),
    material=diffuse_light_mat)
  # Warm light bulb: sphere at y=-0.25, r=0.2
  sphere(origin + vxyz(0, -0.25, 0), 0.2, material=lumensf_mat)
end

make_wall_sconce(xyz(-2.75, 4, 6.5))
make_wall_sconce(xyz(2.75, 4, 6.5))


## ─── Downlights ────────────────────────────────────────────────────────────
# Recessed fluorescent downlights (diffuse, 6" aperture)
# "!xform -n flood -t -3 -1 7.99 -a 2 -t 6 0 0 lib/downlight_a1.rad"
# Two downlights at (-3, -1, 7.99) and (3, -1, 7.99)
# downlight_a1.rad: lightring ring downlight, at origin, normal=(0,0,-1), inner=0.25, outer=0
# → emissive disk facing downward
surface_circle(xyz(-3, -1, 7.99), 0.25, material=lightring_mat)
surface_circle(xyz(3, -1, 7.99), 0.25, material=lightring_mat)

# Recessed spotlight
# "!xform -n spot -t 0 -.5 7.99 lib/downlight_b1.rad"
# downlight_b1.rad: lightring2 ring downlight2, same geometry but spotlight distribution
# lightring2 is a spotlight — we use an emissive ring as approximation
# (Radiance spotlight clips light into a cone; Khepri spotlight is separate)
let spot_lightring2_mat = standard_material(
    name="lightring2",
    base_color=rgba(1.0, 0.66, 0.06, 1.0),
    emissive=rgba(1344.0, 888.0, 85.0, 1.0),
    data=BackendParameter(RAD => RadianceMaterial("lightring2", "spotlight", 1344, 888, 85, nothing, nothing, nothing, nothing)))
  # Spotlight: "void spotlight lightring2 0 0 7 1344 888 85 40 0 0 -.1"
  # Cone angle 40°, direction (0, 0, -0.1). We approximate as emissive ring.
  surface_circle(xyz(0, -0.5, 7.99), 0.25, material=spot_lightring2_mat)
end


## ─── Camera ────────────────────────────────────────────────────────────────
# From scene1.vf: -vp 0 -15 4.3 -vd 0 1 0 -vu 0 0 1 -vh 45 -vv 45
let vp = xyz(0, -15, 4.3),
    vd = vxyz(0, 1, 0)
  set_view(vp, vp + vd, lens=45)
end
