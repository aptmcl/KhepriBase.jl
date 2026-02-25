#=
Scene 2 — Art Gallery
Reverse-engineered from Radiance reference scene (Rendering with Radiance, Chapter 13).

A full art gallery with pitched roof, skylight, archway, windows, doors,
8+ art pieces, furniture, and track lighting.

Source files: scene2/scene2/{sc2bldg.rad, sc2light.rad, scene_art.rad, art.mat}
             scene2/scene2/lib/{floor, walls, ceiling, window, doorbase, doorx2,
             building.mat, luminaire.mat, sconce, track_*}.rad
             scene2/scene2/art_lib/{tapestry, unfurled, feather, fthrbase, waves,
             lotus123, sign, cafe_pic, champagne, tea_cup, goblet, phone,
             tble2_c1, tble4_c1, sofa1_c1, sofa2_c1, chair_c1, deskc_c1,
             desk_c1}.rad

Known limitations:
- Procedural textures (adobe.cal, carpet.cal, dirt.cal) → flat color approximation
- mixtext (text on acrylic sign) → flat gold rectangle
- glow/illum → emissive materials (no shadow-casting distinction)
- antimatter → boolean subtraction (only on backends with boolean ops; approximated otherwise)
- genbox edge radius → sharp-edged boxes (Khepri box has no fillet parameter)
- colorpict image mapping → flat color fallback on backends without texture support
- genworm curved pipes → straight cylinder approximations
- genrev Hermite curves → cone_frustum approximations
- gensurf parametric surfaces → surface_polygon approximation for waves
=#

# Change this line to switch backends:
# using KhepriRadiance
# using KhepriMitsuba

## ═══════════════════════════════════════════════════════════════════════════
## MATERIALS
## ═══════════════════════════════════════════════════════════════════════════

## ─── Building Materials ────────────────────────────────────────────────────
# "ad plastic stucco_white" — adobe-textured wall (flat approximation)
# Original: void texfunc ad → ad plastic stucco_white 0 0 5 .8 .7 .5 0 0
stucco_white_mat = standard_material(
  name="stucco_white",
  base_color=rgba(0.8, 0.7, 0.5, 1.0),
  roughness=1.0,
  data=BackendParameter(RAD => radiance_plastic_material("stucco_white", red=0.8, green=0.7, blue=0.5)))

# "void plastic stucco_white2" — high reflectance ceiling
stucco_white2_mat = standard_material(
  name="stucco_white2",
  base_color=rgba(0.875, 0.825, 0.75, 1.0),
  roughness=1.0,
  data=BackendParameter(RAD => radiance_plastic_material("stucco_white2", red=0.875, green=0.825, blue=0.75)))

# "carp plastic carpet_blue" — carpet pattern (flat approximation)
# Original uses colorfunc with carpet.cal, base plastic .9 .9 .9
carpet_blue_mat = standard_material(
  name="carpet_blue",
  base_color=rgba(0.63, 0.72, 0.81, 1.0),
  roughness=1.0,
  data=BackendParameter(RAD => radiance_plastic_material("carpet_blue", red=0.63, green=0.72, blue=0.81)))

# "void plastic red_roofing"
red_roofing_mat = standard_material(
  name="red_roofing",
  base_color=rgba(0.6, 0.05, 0.05, 1.0),
  roughness=0.03,
  data=BackendParameter(RAD => radiance_plastic_material("red_roofing", red=0.6, green=0.05, blue=0.05, specularity=0.03, roughness=0.03)))

# Door/window frame materials
frame_mat = standard_material(
  name="frame_mat",
  base_color=rgba(0.6, 0.5, 0.4, 1.0),
  roughness=0.02,
  data=BackendParameter(RAD => radiance_plastic_material("frame_mat", red=0.6, green=0.5, blue=0.4, specularity=0.02, roughness=0.02)))

frame_mat2 = standard_material(
  name="frame_mat2",
  base_color=rgba(0.6, 0.5, 0.4, 1.0),
  roughness=0.02,
  data=BackendParameter(RAD => radiance_plastic_material("frame_mat2", red=0.6, green=0.5, blue=0.4, specularity=0.02, roughness=0.02)))

sill_mat = standard_material(
  name="sill_mat",
  base_color=rgba(0.6, 0.6, 0.6, 1.0),
  metallic=1.0,
  roughness=0.05,
  data=BackendParameter(RAD => radiance_metal_material("sill_mat", red=0.6, green=0.6, blue=0.6, specularity=0.85, roughness=0.05)))

handle_mat = standard_material(
  name="handle_mat",
  base_color=rgba(0.6, 0.1, 0.1, 1.0),
  roughness=1.0,
  data=BackendParameter(RAD => radiance_plastic_material("handle_mat", red=0.6, green=0.1, blue=0.1)))

door_glass_mat = standard_material(
  name="door_glass",
  base_color=rgba(0.9, 0.9, 0.9, 0.3),
  transmission=0.9,
  ior=1.5,
  data=BackendParameter(RAD => radiance_glass_material("door_glass", gray=0.9)))

# Luminaire finish
enamel_white_mat = standard_material(
  name="enamel_white",
  base_color=rgba(0.7, 0.6, 0.5, 1.0),
  roughness=0.02,
  data=BackendParameter(RAD => radiance_plastic_material("enamel_white", red=0.7, green=0.6, blue=0.5, specularity=0.03, roughness=0.02)))

# Sky illuminance (emissive for daylight simulation)
sky_illum_mat = standard_material(
  name="sky_illum",
  base_color=rgba(0.88, 0.88, 0.98, 1.0),
  emissive=rgba(0.88, 0.88, 0.98, 1.0),
  data=BackendParameter(RAD => radiance_light_material("sky_illum", red=0.88, green=0.88, blue=0.98)))

## ─── Art Materials ─────────────────────────────────────────────────────────
# Accent pedestals
accent_white_mat = standard_material(
  name="accent_white",
  base_color=rgba(0.8, 0.8, 0.8, 1.0),
  roughness=1.0,
  data=BackendParameter(RAD => radiance_plastic_material("accent_white", gray=0.8)))

accent_green_mat = standard_material(
  name="accent_green",
  base_color=rgba(0.01, 0.9, 0.02, 1.0),
  roughness=1.0,
  data=BackendParameter(RAD => radiance_plastic_material("accent_green", red=0.01, green=0.9, blue=0.02)))

accent_red_mat = standard_material(
  name="accent_red",
  base_color=rgba(0.4, 0.01, 0.05, 1.0),
  roughness=0.025,
  data=BackendParameter(RAD => radiance_plastic_material("accent_red", red=0.4, green=0.01, blue=0.05, specularity=0.04, roughness=0.025)))

# Tapestry glass materials
dk_red_g_mat = standard_material(
  name="dk_red_g",
  base_color=rgba(0.2, 0.01, 0.02, 0.5),
  transmission=0.2,
  ior=1.5,
  data=BackendParameter(RAD => radiance_glass_material("dk_red_g", red=0.2, green=0.01, blue=0.02)))

yellow_g_mat = standard_material(
  name="yellow_g",
  base_color=rgba(0.6, 0.6, 0.25, 0.5),
  transmission=0.6,
  ior=1.5,
  data=BackendParameter(RAD => radiance_glass_material("yellow_g", red=0.6, green=0.6, blue=0.25)))

dk_green_g_mat = standard_material(
  name="dk_green_g",
  base_color=rgba(0.01, 0.2, 0.05, 0.5),
  transmission=0.2,
  ior=1.5,
  data=BackendParameter(RAD => radiance_glass_material("dk_green_g", red=0.01, green=0.2, blue=0.05)))

amber_g_mat = standard_material(
  name="amber_g",
  base_color=rgba(0.25, 0.1, 0.01, 0.5),
  transmission=0.25,
  ior=1.5,
  data=BackendParameter(RAD => radiance_glass_material("amber_g", red=0.25, green=0.1, blue=0.01)))

dk_mag_g_mat = standard_material(
  name="dk_mag_g",
  base_color=rgba(0.25, 0.01, 0.2, 0.5),
  transmission=0.25,
  ior=1.5,
  data=BackendParameter(RAD => radiance_glass_material("dk_mag_g", red=0.25, green=0.01, blue=0.2)))

dk_blue_g_mat = standard_material(
  name="dk_blue_g",
  base_color=rgba(0.02, 0.01, 0.4, 0.5),
  transmission=0.4,
  ior=1.5,
  data=BackendParameter(RAD => radiance_glass_material("dk_blue_g", red=0.02, green=0.01, blue=0.4)))

white_plaster_mat = standard_material(
  name="white_plaster",
  base_color=rgba(0.8, 0.8, 0.8, 1.0),
  roughness=1.0,
  data=BackendParameter(RAD => radiance_plastic_material("white_plaster", gray=0.8)))

# Unfurled sculpture
rod_color_mat = standard_material(
  name="rod_color",
  base_color=rgba(0.9, 0.8, 0.7, 1.0),
  roughness=0.02,
  data=BackendParameter(RAD => radiance_plastic_material("rod_color", red=0.9, green=0.8, blue=0.7, specularity=0.04, roughness=0.02)))

# Feather mobile
quill_material_mat = standard_material(
  name="quill_material",
  base_color=rgba(0.6, 0.8, 0.9, 1.0),
  roughness=0.02,
  data=BackendParameter(RAD => radiance_plastic_material("quill_material", red=0.6, green=0.8, blue=0.9, specularity=0.03, roughness=0.02)))

pearl_mat = standard_material(
  name="pearl",
  base_color=rgba(0.5, 0.8, 0.9, 1.0),
  roughness=0.03,
  data=BackendParameter(RAD => radiance_plastic_material("pearl", red=0.5, green=0.8, blue=0.9, specularity=0.2, roughness=0.03)))

# Waves
cepia_mat = standard_material(
  name="cepia",
  base_color=rgba(0.4, 0.25, 0.05, 1.0),
  roughness=1.0,
  data=BackendParameter(RAD => radiance_plastic_material("cepia", red=0.4, green=0.25, blue=0.05)))

blue_wave_mat = standard_material(
  name="blue",
  base_color=rgba(0.15, 0.2, 0.8, 1.0),
  roughness=1.0,
  data=BackendParameter(RAD => radiance_plastic_material("blue", red=0.15, green=0.2, blue=0.8)))

pewter_mat = standard_material(
  name="pewter",
  base_color=rgba(0.8, 0.8, 0.82, 1.0),
  metallic=1.0,
  roughness=0.05,
  data=BackendParameter(RAD => radiance_metal_material("pewter", red=0.8, green=0.8, blue=0.82, specularity=0.85, roughness=0.05)))

gold_wave_mat = standard_material(
  name="gold",
  base_color=rgba(0.8, 0.6, 0.3, 1.0),
  metallic=1.0,
  roughness=0.03,
  data=BackendParameter(RAD => radiance_metal_material("gold", red=0.8, green=0.6, blue=0.3, specularity=0.85, roughness=0.03)))

# Lotus glass
glass_outer_mat = standard_material(
  name="glass_outer",
  base_color=rgba(0.8, 0.8, 0.9, 0.3),
  transmission=0.8,
  ior=1.5,
  data=BackendParameter(RAD => radiance_dielectric_material("glass_outer", red=0.8, green=0.8, blue=0.9)))

# Sign materials
acrylic_mat = standard_material(
  name="acrylic",
  base_color=rgba(0.98, 0.98, 0.98, 0.3),
  transmission=0.98,
  ior=1.4,
  data=BackendParameter(RAD => radiance_dielectric_material("acrylic", gray=0.98)))

gold_leaf_mat = standard_material(
  name="gold_leaf",
  base_color=rgba(0.68, 0.27, 0.002, 1.0),
  metallic=1.0,
  roughness=0.05,
  data=BackendParameter(RAD => radiance_metal_material("gold_leaf", red=0.68, green=0.27, blue=0.002, specularity=0.875, roughness=0.05)))

# Cafe painting — flat color fallback; backends with texture support use data field
cafe_painting_mat = standard_material(
  name="cafe_art",
  base_color=rgba(0.5, 0.4, 0.3, 1.0),
  roughness=1.0,
  data=BackendParameter(RAD => radiance_plastic_material("cafe_art", red=0.5, green=0.4, blue=0.3)))

# Cups and goblets
goldc_mat = standard_material(
  name="goldc",
  base_color=rgba(1.0, 0.7, 0.4, 1.0),
  metallic=1.0,
  roughness=0.02,
  data=BackendParameter(RAD => radiance_metal_material("goldc", red=1.0, green=0.7, blue=0.4, specularity=0.9, roughness=0.02)))

silverc_mat = standard_material(
  name="silverc",
  base_color=rgba(0.8, 0.8, 0.81, 1.0),
  metallic=1.0,
  roughness=0.02,
  data=BackendParameter(RAD => radiance_metal_material("silverc", red=0.8, green=0.8, blue=0.81, specularity=0.9, roughness=0.02)))

clear_liquid_mat = standard_material(
  name="clear_liquid",
  base_color=rgba(0.8, 0.8, 0.6, 0.3),
  transmission=0.8,
  ior=1.1,
  data=BackendParameter(RAD => radiance_dielectric_material("clear_liquid", red=0.8, green=0.8, blue=0.6)))

# Pottery (dirt.cal noise → flat grey approximation)
pottery_mat = standard_material(
  name="pottery",
  base_color=rgba(0.5, 0.5, 0.46, 1.0),
  roughness=0.01,
  data=BackendParameter(RAD => radiance_plastic_material("pottery", red=0.5, green=0.5, blue=0.46, specularity=0.02, roughness=0.01)))

china_mat = standard_material(
  name="china",
  base_color=rgba(0.9, 0.9, 0.9, 1.0),
  roughness=0.02,
  data=BackendParameter(RAD => radiance_plastic_material("china", red=0.9, green=0.9, blue=0.9, specularity=0.03, roughness=0.02)))

## ─── Furniture Materials ───────────────────────────────────────────────────
table_color_mat = standard_material(
  name="table_color",
  base_color=rgba(0.1, 0.1, 0.1, 1.0),
  roughness=0.025,
  data=BackendParameter(RAD => radiance_plastic_material("table_color", red=0.1, green=0.1, blue=0.1, specularity=0.025, roughness=0.025)))

pipe_color_mat = standard_material(
  name="pipe_color",
  base_color=rgba(0.7, 0.7, 0.7, 1.0),
  metallic=1.0,
  roughness=0.02,
  data=BackendParameter(RAD => radiance_metal_material("pipe_color", gray=0.7, specularity=0.9, roughness=0.02)))

seat_color_mat = standard_material(
  name="seat_color",
  base_color=rgba(0.8, 0.6, 0.05, 1.0),
  roughness=1.0,
  data=BackendParameter(RAD => radiance_plastic_material("seat_color", red=0.8, green=0.6, blue=0.05)))

desktop_color_mat = standard_material(
  name="desktop_color",
  base_color=rgba(0.8, 0.6, 0.05, 1.0),
  roughness=1.0,
  data=BackendParameter(RAD => radiance_plastic_material("desktop_color", red=0.8, green=0.6, blue=0.05)))

deskmid_color_mat = standard_material(
  name="deskmid_color",
  base_color=rgba(0.1, 0.1, 0.1, 1.0),
  roughness=0.025,
  data=BackendParameter(RAD => radiance_plastic_material("deskmid_color", red=0.1, green=0.1, blue=0.1, specularity=0.025, roughness=0.025)))


## ═══════════════════════════════════════════════════════════════════════════
## BUILDING SHELL
## ═══════════════════════════════════════════════════════════════════════════

## ─── Floor ─────────────────────────────────────────────────────────────────
# "carpet_blue polygon floor" — 40' × 25' at z=0
surface_polygon(
  xyz(0.01, 0.01, 0), xyz(39.99, 0.01, 0),
  xyz(39.9, 24.99, 0), xyz(0.01, 24.99, 0),
  material=carpet_blue_mat)

## ─── Walls ─────────────────────────────────────────────────────────────────
# North wall: "!genbox stucco_white wall_north 40 .5 10.5 | xform -t 0 24.5 0"
box(xyz(0, 24.5, 0), 40, 0.5, 10.5, material=stucco_white_mat)

# South wall (3 pieces around door opening):
# "!genbox stucco_white wall_south1 31.2 .5 10.5"
box(xyz(0, 0, 0), 31.2, 0.5, 10.5, material=stucco_white_mat)
# "!genbox stucco_white wall_south1 7 .5 2.5 | xform -t 31.2 0 8" (door header)
box(xyz(31.2, 0, 8), 7, 0.5, 2.5, material=stucco_white_mat)
# "!genbox stucco_white wall_south3 1.8 .5 10.5 | xform -t 38.2 0 0"
box(xyz(38.2, 0, 0), 1.8, 0.5, 10.5, material=stucco_white_mat)

# West wall: genprism with 8 vertices, pitched roof profile
# "!genprism stucco_white wall_west 8
#   0 25  0 0  10.5 0  14 10.15  19.4 10.15  19.4 14.85  14 14.85  10.5 25
#   -l 0 0 .5 | xform -ry -90 -t .5 0 0"
# After -ry -90 -t .5 0 0: genprism local (x,y) → world (z,y), extrusion in -x → x=0.5 to 0
# Profile in (z, y) coords:
prism(
  [xyz(0, 25, 0), xyz(0, 0, 0), xyz(0, 0, 10.5), xyz(0, 10.15, 14),
   xyz(0, 10.15, 19.4), xyz(0, 14.85, 19.4), xyz(0, 14.85, 14), xyz(0, 25, 10.5)],
  vx(0.5), stucco_white_mat)

# Center wall partition: 12 vertices with archway
# "!genprism stucco_white wall_center 12
#   0 25  0 16  8 16  8 9  0 9  0 0  10.5 0  14 10.15
#   19.4 10.15  19.4 14.85  14 14.85  10.5 25
#   -l 0 0 .5 | xform -ry -90 -t 30 0 0"
prism(
  [xyz(30, 25, 0), xyz(30, 16, 0), xyz(30, 16, 8), xyz(30, 9, 8),
   xyz(30, 9, 0), xyz(30, 0, 0), xyz(30, 0, 10.5), xyz(30, 10.15, 14),
   xyz(30, 10.15, 19.4), xyz(30, 14.85, 19.4), xyz(30, 14.85, 14), xyz(30, 25, 10.5)],
  vx(-0.5), stucco_white_mat)

# East wall: 17 vertices with 2 window openings
# "!genprism stucco_white wall_east 17
#   0 25  0 22.5  8 22.5  8 18  2 18  2 22.5  0 22.5
#   0 7   8 7     8 2.5   2 2.5  2 7  0 7
#   0 0   10.5 0  15.3 12.5  10.5 25
#   -l 0 0 .5 | xform -ry -90 -t 40 0 0"
prism(
  [xyz(40, 25, 0), xyz(40, 22.5, 0), xyz(40, 22.5, 8), xyz(40, 18, 8),
   xyz(40, 18, 2), xyz(40, 22.5, 2), xyz(40, 22.5, 0),
   xyz(40, 7, 0), xyz(40, 7, 8), xyz(40, 2.5, 8),
   xyz(40, 2.5, 2), xyz(40, 7, 2), xyz(40, 7, 0),
   xyz(40, 0, 0), xyz(40, 0, 10.5), xyz(40, 12.5, 15.3), xyz(40, 25, 10.5)],
  vx(-0.5), stucco_white_mat)

# Skylight walls
# "!genbox stucco_white wall_skylight_s 29.95 .5 5.4 | xform -t .01 10.15 14"
box(xyz(0.01, 10.15, 14), 29.95, 0.5, 5.4, material=stucco_white_mat)
# "!genbox stucco_white wall_skylight_n 29.95 .5 5.4 | xform -t .01 14.35 14"
box(xyz(0.01, 14.35, 14), 29.95, 0.5, 5.4, material=stucco_white_mat)

## ─── Ceiling and Roof ──────────────────────────────────────────────────────
# South-west ceiling: 29.98 × 11.2 × 0.5, rotated -21° around x, then translated
# "!genbox stucco_white2 ceiling_south_w 29.98 11.2 .5 |
#   xform -t 0.01 0 -.5 -rx 21 -t 0 0 10.5"
# The 21° pitch means the ceiling slopes upward from south to north
# At y=0: z≈10.0, at y=11.2: z≈10.0 + 11.2*sin(21°) ≈ 14.0
let pitch = deg2rad(21),
    cs_sw = loc_from_o_vx_vy(
      xyz(0.01, 0, 10.0),
      vx(1),
      vxyz(0, cos(pitch), sin(pitch)))
  box(cs_sw, 29.98, 11.2, 0.5, material=stucco_white2_mat)
end

# South-west roof
let pitch = deg2rad(21),
    cs_swr = loc_from_o_vx_vy(
      xyz(-0.19, -0.2*cos(pitch), 10.5 - 0.2*sin(pitch)),
      vx(1),
      vxyz(0, cos(pitch), sin(pitch)))
  box(cs_swr, 30.18, 11.4, 0.1, material=red_roofing_mat)
end

# North-west ceiling: mirrored from south
let pitch = deg2rad(21),
    cs_nw = loc_from_o_vx_vy(
      xyz(0.01, 25, 10.0),
      vx(1),
      vxyz(0, -cos(pitch), sin(pitch)))
  box(cs_nw, 29.98, 11.2, 0.5, material=stucco_white2_mat)
end

# North-west roof
let pitch = deg2rad(21),
    cs_nwr = loc_from_o_vx_vy(
      xyz(-0.19, 25 + 0.2*cos(pitch), 10.5 - 0.2*sin(pitch)),
      vx(1),
      vxyz(0, -cos(pitch), sin(pitch)))
  box(cs_nwr, 30.18, 11.4, 0.1, material=red_roofing_mat)
end

# South-east ceiling
let pitch = deg2rad(21),
    cs_se = loc_from_o_vx_vy(
      xyz(30, 0, 10.0),
      vx(1),
      vxyz(0, cos(pitch), sin(pitch)))
  box(cs_se, 9.99, 13.45, 0.5, material=stucco_white2_mat)
end

# South-east roof
let pitch = deg2rad(21),
    cs_ser = loc_from_o_vx_vy(
      xyz(30, -0.2*cos(pitch), 10.5 - 0.2*sin(pitch)),
      vx(1),
      vxyz(0, cos(pitch), sin(pitch)))
  box(cs_ser, 10.19, 13.65, 0.1, material=red_roofing_mat)
end

# North-east ceiling
let pitch = deg2rad(21),
    cs_ne = loc_from_o_vx_vy(
      xyz(30, 25, 10.0),
      vx(1),
      vxyz(0, -cos(pitch), sin(pitch)))
  box(cs_ne, 9.99, 13.45, 0.5, material=stucco_white2_mat)
end

# North-east roof
let pitch = deg2rad(21),
    cs_ner = loc_from_o_vx_vy(
      xyz(30, 25 + 0.2*cos(pitch), 10.5 - 0.2*sin(pitch)),
      vx(1),
      vxyz(0, -cos(pitch), sin(pitch)))
  box(cs_ner, 10.19, 13.7, 0.1, material=red_roofing_mat)
end

# Daylight reflector beneath skylight
# "!genprism stucco_white2 reflector 3
#   0 -3.5  2.45 0  0 3.5 -l 0 0 29.98
#   | xform -ry -90 -t 29.99 12.5 8.01"
# Triangle prism: after -ry -90, the 29.98 extrusion goes in -x direction
prism(
  [xyz(29.99, 12.5 - 3.5, 8.01),
   xyz(29.99, 12.5, 8.01 + 2.45),
   xyz(29.99, 12.5 + 3.5, 8.01)],
  vx(-29.98), stucco_white2_mat)

## ─── Windows ───────────────────────────────────────────────────────────────
# Helper: window at a given position (4.5' × 6' with frame and glass)
function make_window(origin_cs)
  # Frame: 4 border pieces around glass opening
  box(origin_cs, 0.1, 0.5, 6, material=frame_mat)                        # left
  box(origin_cs + vxyz(0.1, 0, 5.9), 4.3, 0.5, 0.1, material=frame_mat)  # top
  box(origin_cs + vxyz(4.4, 0, 0), 0.1, 0.5, 6, material=frame_mat)      # right
  box(origin_cs + vxyz(0.1, 0, 0), 4.3, 0.5, 0.1, material=frame_mat)    # bottom
  # Sash: inner mullion frame
  box(origin_cs + vxyz(0.1, 0.3, 0.1), 0.1, 0.1, 5.8, material=frame_mat)
  box(origin_cs + vxyz(0.2, 0.3, 5.8), 4.1, 0.1, 0.1, material=frame_mat)
  box(origin_cs + vxyz(4.3, 0.3, 0.1), 0.1, 0.1, 5.8, material=frame_mat)
  box(origin_cs + vxyz(0.2, 0.3, 0.1), 4.1, 0.1, 0.1, material=frame_mat)
  # Glass panel
  surface_polygon(
    origin_cs + vxyz(0.2, 0.35, 0.2),
    origin_cs + vxyz(4.3, 0.35, 0.2),
    origin_cs + vxyz(4.3, 0.35, 5.8),
    origin_cs + vxyz(0.2, 0.35, 5.8),
    material=door_glass_mat)
end

# SE window: "!xform -n se_window -rz 90 -t 40 2.5 2 window.rad"
# After -rz 90: window faces along -y → placed at east wall facing inward
make_window(loc_from_o_vx_vy(xyz(40, 2.5, 2), vy(-1), vx(1)))

# NE window: "!xform -n ne_window -rz 90 -t 40 18 2 window.rad"
make_window(loc_from_o_vx_vy(xyz(40, 18, 2), vy(-1), vx(1)))

## ─── Doors ─────────────────────────────────────────────────────────────────
# Double door: "!xform -n s_doors -t 31.2 0 0 doorx2.rad"
# doorx2.rad: 7' wide × 8' tall × 0.5' deep frame + 2 glass doors
let door_origin = xyz(31.2, 0, 0)
  # Door frame
  box(door_origin, 0.1, 0.5, 8, material=frame_mat)                           # left jamb
  box(door_origin + vxyz(0.1, 0, 7.9), 6.8, 0.5, 0.1, material=frame_mat)    # top jamb
  box(door_origin + vxyz(6.9, 0, 0), 0.1, 0.5, 8, material=frame_mat)        # right jamb
  box(door_origin + vxyz(0.1, -0.05, 0), 6.8, 0.55, 0.05, material=sill_mat) # sill

  # West door (doorbase at x=0.1, y=0.1)
  let dw = door_origin + vxyz(0.1, 0.1, 0)
    # Mullions
    box(dw, 0.1, 0.1, 7.85, material=frame_mat2)
    box(dw + vxyz(0.1, 0, 7.8), 3.19, 0.1, 0.1, material=frame_mat2)
    box(dw + vxyz(3.29, 0, 0.05), 0.1, 0.1, 7.85, material=frame_mat2)
    box(dw + vxyz(0.1, 0, 0.05), 3.19, 0.1, 0.45, material=frame_mat2)
    # Glass panel
    surface_polygon(
      dw + vxyz(0.1, 0.05, 0.15), dw + vxyz(3.29, 0.05, 0.15),
      dw + vxyz(3.29, 0.05, 7.85), dw + vxyz(0.1, 0.05, 7.85),
      material=door_glass_mat)
    # Door handles
    cylinder(dw + vxyz(3.25, -0.1, 3.0), 0.05, dw + vxyz(3.25, -0.1, 4.0),
             material=frame_mat)
    cylinder(dw + vxyz(0.05, 0.1, 3.5), 0.05, dw + vxyz(3.35, 0.1, 3.5),
             material=frame_mat)
  end

  # East door (mirrored: -mx from x=6.9, y=0.1)
  let de = door_origin + vxyz(6.9, 0.1, 0)
    # Mullions (mirrored in x)
    box(de + vxyz(-0.1, 0, 0.05), 0.1, 0.1, 7.85, material=frame_mat2)
    box(de + vxyz(-3.29, 0, 7.8), 3.19, 0.1, 0.1, material=frame_mat2)
    box(de + vxyz(-3.39, 0, 0.05), 0.1, 0.1, 7.85, material=frame_mat2)
    box(de + vxyz(-3.29, 0, 0.05), 3.19, 0.1, 0.45, material=frame_mat2)
    # Glass panel
    surface_polygon(
      de + vxyz(-3.29, 0.05, 0.15), de + vxyz(-0.1, 0.05, 0.15),
      de + vxyz(-0.1, 0.05, 7.85), de + vxyz(-3.29, 0.05, 7.85),
      material=door_glass_mat)
    # Door handles
    cylinder(de + vxyz(-3.25, -0.1, 3.0), 0.05, de + vxyz(-3.25, -0.1, 4.0),
             material=frame_mat)
    cylinder(de + vxyz(-0.05, 0.1, 3.5), 0.05, de + vxyz(-3.35, 0.1, 3.5),
             material=frame_mat)
  end
end


## ═══════════════════════════════════════════════════════════════════════════
## ART INSTALLATIONS
## ═══════════════════════════════════════════════════════════════════════════

## ─── Tapestry ──────────────────────────────────────────────────────────────
# "!xform -n tap -rz 90 -t .5 12.5 4.25 art_lib/tapestry.rad"
# Tapestry insertion point at center of 10'×8' screen.
# After -rz 90 -t .5 12.5 4.25: local -y → +x, local x → +y, local z → z
# Bands face west wall at x≈0.5, centered at y=12.5, z=4.25
# Local coords: x∈[-3,3]→y∈[9.5,15.5], y=-2→x=2.5, z∈[-2,2]→z∈[2.25,6.25]
let tx = 0.5, ty = 12.5, tz = 4.25,
    # Transform: local(x,y,z) → world(ty+x, tx-y, tz+z) after rz 90
    tw(lx, ly, lz) = xyz(tx - ly, ty + lx, tz + lz)
  # Band 1: dk_red_g, z=-2 to -0.5
  surface_polygon(tw(-3, -2, -2), tw(3, -2, -2), tw(3, -2, -0.5), tw(-3, -2, -0.5),
    material=dk_red_g_mat)
  # Band 2: yellow_g, z=-0.5 to 0.5
  surface_polygon(tw(-3, -2, -0.5), tw(3, -2, -0.5), tw(3, -2, 0.5), tw(-3, -2, 0.5),
    material=yellow_g_mat)
  # Band 3: dk_green_g, z=0.5 to 1
  surface_polygon(tw(-3, -2, 0.5), tw(3, -2, 0.5), tw(3, -2, 1), tw(-3, -2, 1),
    material=dk_green_g_mat)
  # Band 4: amber_g, z=1 to 1.4
  surface_polygon(tw(-3, -2, 1), tw(3, -2, 1), tw(3, -2, 1.4), tw(-3, -2, 1.4),
    material=amber_g_mat)
  # Band 5: dk_mag_g, z=1.4 to 1.7
  surface_polygon(tw(-3, -2, 1.4), tw(3, -2, 1.4), tw(3, -2, 1.7), tw(-3, -2, 1.7),
    material=dk_mag_g_mat)
  # Band 6: dk_blue_g, z=1.7 to 1.9
  surface_polygon(tw(-3, -2, 1.7), tw(3, -2, 1.7), tw(3, -2, 1.9), tw(-3, -2, 1.9),
    material=dk_blue_g_mat)
  # Band 7: dk_red_g, z=1.9 to 2
  surface_polygon(tw(-3, -2, 1.9), tw(3, -2, 1.9), tw(3, -2, 2), tw(-3, -2, 2),
    material=dk_red_g_mat)
  # Horizon stripe: dk_blue_g at z≈0 (y=-2.01 in local)
  surface_polygon(
    tw(-3, -2.01, -0.025), tw(3, -2.01, -0.025),
    tw(3, -2.01, 0.025), tw(-3, -2.01, 0.025),
    material=dk_blue_g_mat)
  # Backing panel: 10' × 0.1' × 7.5' at local (-5, -0.1, -4)
  box(tw(-5, -0.1, -4), 10, 0.1, 7.5, material=white_plaster_mat)
end

## ─── Lotus Installation ────────────────────────────────────────────────────
# Pedestal: "!genbox accent_green pedistal 1 1 3 | xform -t 3.5 3.5 0"
box(xyz(3.5, 3.5, 0), 1, 1, 3, material=accent_green_mat)
# Base cone: center (4, 4, 3) to (4, 4, 3.25), r_bot=0.4, r_top=0.25
cone_frustum(xyz(4, 4, 3), 0.4, 0.25, 0.25, material=accent_white_mat)
# Coaster ring: at (4, 4, 3.25), inner=0, outer=0.25
surface_circle(xyz(4, 4, 3.25), 0.25, material=accent_green_mat)

# Lotus: "!xform -n lotus -s .0833 -t 4 4 3.26 art_lib/lotus123.rad"
# Scale 0.0833 (inch to feet). Lotus is ~6" diameter → ~0.5' after scale.
# lotus123.rad: 4 glass spheres + antimatter bowl
# After scaling: sphere radius = 3 * 0.0833 = 0.25', centers offset by 0.0833
let ls = 0.0833, lo = xyz(4, 4, 3.26)
  # 4 outer leaf spheres (radius 3*ls ≈ 0.25)
  sphere(lo + vxyz(-1*ls, -1*ls, 2.5*ls), 3*ls, material=glass_outer_mat)
  sphere(lo + vxyz(1*ls, 1*ls, 2.5*ls), 3*ls, material=glass_outer_mat)
  sphere(lo + vxyz(-1*ls, 1*ls, 2.5*ls), 3*ls, material=glass_outer_mat)
  sphere(lo + vxyz(1*ls, -1*ls, 2.5*ls), 3*ls, material=glass_outer_mat)
  # Note: antimatter bowl subtraction is backend-dependent.
  # Backends with boolean ops could use subtracted_solids here.
  # The lotus bowl (hollow_1 sphere at z=5.5*ls, r=4*ls) scoops the interior.
end

## ─── Unfurled Sculpture ────────────────────────────────────────────────────
# Base: "accent_red cylinder u_base 0 0 7 8 21 0 8 21 1.5 1.5"
cylinder(xyz(8, 21, 0), 1.5, 1.5, material=accent_red_mat)
# Top ring: "accent_red ring u_baset 0 0 8 8 21 1.5 0 0 1 1.5 0"
surface_circle(xyz(8, 21, 1.5), 1.5, material=accent_red_mat)

# "!xform -n unf -t 8 21 2.075 art_lib/unfurled.rad"
# Unfurled: continuous rod sculpture — genworm torus sections → cylinder approximations
let uo = xyz(8, 21, 2.075)
  # Base torus (180° arc, R=1, r=0.1) centered at local origin, in XY plane
  # The base arc sweeps from one side to the other
  # Approximate with a half-circle of cylinder segments
  # After xform -rz 180 -t -.14 0 -.5: shifted
  let base_c = uo + vxyz(-0.14, 0, -0.5)
    for i in 0:11
      let a0 = pi * i / 12,
          a1 = pi * (i + 1) / 12,
          p0 = base_c + vxyz(-cos(a0), -sin(a0), 0),
          p1 = base_c + vxyz(-cos(a1), -sin(a1), 0)
        cylinder(p0, 0.1, p1, material=rod_color_mat)
      end
    end
  end

  # Horizontal tapered cone: "(0.2, 1, -0.5) to (-1.14, 0, -0.5)"
  cone_frustum(uo + vxyz(0.2, 1, -0.5), 0.075,
               uo + vxyz(-1.14, 0, -0.5), 0.1,
               material=rod_color_mat)
  # Sphere at cone tip
  sphere(uo + vxyz(0.2, 1, -0.5), 0.075, material=rod_color_mat)

  # Vertical leaning cone: "(0.2, 1, -0.5) to (0.3, -1.1, 4.2)"
  cone_frustum(uo + vxyz(0.2, 1, -0.5), 0.075,
               uo + vxyz(0.3, -1.1, 4.2), 0.02,
               material=rod_color_mat)
  # Sphere at top of leaning cone
  sphere(uo + vxyz(0.3, -1.1, 4.2), 0.175, material=rod_color_mat)
end

## ─── Feather Mobile ────────────────────────────────────────────────────────
# "!xform -n feath -s 1 -rz -60 -t 15 5 2.5 art_lib/feather.rad"
# feather.rad: 50 cones arrayed with progressive scale (0.975 factor) + rotations
# fthrbase.rad: "quill_material cone vane 0 0 8 0 0 0 2 0 0 .1 .025"
# Each vane: cone from (0,0,0) to (2,0,0), r_bot=0.1, r_top=0.025
# Mirrored set with -my
let fo = xyz(15, 5, 2.5),
    rz_base = deg2rad(-60)
  # Generate 50 feather vanes with progressive transformations
  # Each iteration: scale by 0.975, rotate ry=-2.5°, rx=0.5°, rz=2°, translate (0.2, 0, 0.2)
  # We approximate: each vane is a cone, progressively scaled and offset
  for i in 0:49
    let s = 0.975^i,
        # Approximate cumulative position: spiral pattern
        angle = i * deg2rad(2),  # rz accumulation
        height = i * 0.2 * s,   # z accumulation
        dx = i * 0.2 * s * cos(rz_base + angle),
        dy = i * 0.2 * s * sin(rz_base + angle),
        p = fo + vxyz(dx, dy, height),
        dir = vxyz(cos(rz_base + angle), sin(rz_base + angle), 0)
      # Forward vane
      cone_frustum(p, 0.1*s, p + dir * 2*s, 0.025*s, material=quill_material_mat)
      # Mirrored vane (-my: negate y component of direction)
      let dir_m = vxyz(dir.x, -dir.y, dir.z)
        cone_frustum(p, 0.1*s * 0.95, p + dir_m * 2*s * 0.95, 0.025*s * 0.95,
                     material=quill_material_mat)
      end
    end
  end
  # Pearl head: genworm with sphere-like shape
  sphere(fo + vxyz(0, 0, -0.5), 0.3 * 0.6, material=pearl_mat)
end

## ─── Waves ─────────────────────────────────────────────────────────────────
# gensurf creates a 5'×5' sinusoidal surface panel
# "gensurf cepia waves '5-s*(5-0)' '5-t*(5-0)' 'mag(s,t)*.25*sin(5*PI*t)*sin(7*PI*s)'
#   25 35 -s -e 'mag(s,t)=t' | xform -rx 90 -t -2.5 -.26 -2.5"
# After -rx 90: the surface is rotated to be wall-mounted (XZ plane)
# We approximate with a flat polygon representing the panel face.
# Each wave instance uses a different material.
function make_wave_panel(origin, rot_y, rot_z, mat)
  # 5'×5' flat panel approximation in rotated coordinate system
  # The actual gensurf creates a sinusoidal surface — we approximate as flat
  let cy = cos(deg2rad(rot_y)), sy = sin(deg2rad(rot_y)),
      cz = cos(deg2rad(rot_z)), sz = sin(deg2rad(rot_z)),
      # Local panel corners: (-2.5, 0, -2.5) to (2.5, 0, 2.5) after centering
      corners = [vxyz(-2.5, 0, -2.5), vxyz(2.5, 0, -2.5),
                 vxyz(2.5, 0, 2.5), vxyz(-2.5, 0, 2.5)]
    surface_polygon(
      [origin + c for c in corners],
      material=mat)
  end
end

# Wave instances from scene_art.rad:
# wave1: "-ry 90 -rz 90 -t 30 4 5" (cepia default)
make_wave_panel(xyz(30, 4, 5), 90, 90, cepia_mat)
# wave2: "-ry -90 -rz 90 -t 30 21 5" (blue)
make_wave_panel(xyz(30, 21, 5), -90, 90, blue_wave_mat)
# wave3: "-ry 45 -t 25.75 24.5 5" (gold)
make_wave_panel(xyz(25.75, 24.5, 5), 45, 0, gold_wave_mat)
# wave4: "-ry -45 -rz -90 -t 29.5 4.0 5" (blue)
make_wave_panel(xyz(29.5, 4.0, 5), -45, -90, blue_wave_mat)
# wave5: "-ry -90 -rz 90 -t .5 3.5 5" (gold)
make_wave_panel(xyz(0.5, 3.5, 5), -90, 90, gold_wave_mat)
# wave6: "-ry 90 -rz 90 -t .5 21.5 5" (pewter)
make_wave_panel(xyz(0.5, 21.5, 5), 90, 90, pewter_mat)

## ─── Sign ──────────────────────────────────────────────────────────────────
# "!xform -n sign1 -t 30.75 24.44 6 art_lib/sign.rad"
# sign.rad: 8.55' × 0.05' × 1.5' box — acrylic with gold leaf lettering
# mixtext not supported → flat gold front, acrylic sides
let so = xyz(30.75, 24.44, 6)
  # Front face (gold leaf — approximation of mixtext)
  surface_polygon(
    so + vxyz(8.55, 0, 0), so + vxyz(8.55, 0, 1.5),
    so + vxyz(0, 0, 1.5), so,
    material=gold_leaf_mat)
  # Left side
  surface_polygon(
    so + vxyz(0, 0, 1.5), so + vxyz(0, 0.05, 1.5),
    so + vxyz(0, 0.05, 0), so,
    material=acrylic_mat)
  # Bottom
  surface_polygon(
    so + vxyz(0, 0.05, 0), so + vxyz(8.55, 0.05, 0),
    so + vxyz(8.55, 0, 0), so,
    material=acrylic_mat)
  # Back
  surface_polygon(
    so + vxyz(8.55, 0.05, 0), so + vxyz(0, 0.05, 0),
    so + vxyz(0, 0.05, 1.5), so + vxyz(8.55, 0.05, 1.5),
    material=acrylic_mat)
  # Right side
  surface_polygon(
    so + vxyz(8.55, 0, 1.5), so + vxyz(8.55, 0, 0),
    so + vxyz(8.55, 0.05, 0), so + vxyz(8.55, 0.05, 1.5),
    material=acrylic_mat)
  # Top
  surface_polygon(
    so + vxyz(0, 0.05, 1.5), so + vxyz(0, 0, 1.5),
    so + vxyz(8.55, 0, 1.5), so + vxyz(8.55, 0.05, 1.5),
    material=acrylic_mat)
end

## ─── Cafe Painting ─────────────────────────────────────────────────────────
# "!xform -n painting -t 14 24.4 2.5 art_lib/cafe_pic.rad"
# cafe_pic.rad: 6'×6' polygon with colorpict texture mapped from scene1.pic
# colorpict not supported on all backends → flat color fallback
let po = xyz(14, 24.4, 2.5)
  surface_polygon(
    po, po + vxyz(6, 0, 0),
    po + vxyz(6, 0, 6), po + vxyz(0, 0, 6),
    material=cafe_painting_mat)
end


## ═══════════════════════════════════════════════════════════════════════════
## CUPS, GOBLETS, AND VESSELS
## ═══════════════════════════════════════════════════════════════════════════

## ─── Champagne Bowls ───────────────────────────────────────────────────────
# "!xform -n c1 -t 1.2 0 0 -s 1.5 -a 8 -s .9 -rz 50 -i 1
#   -t -.4 0 0 -rz -30 -t 26.5 3.5 1.3 art_lib/champagne.rad"
# 8 instances in a spiral arrangement, progressively scaled
# champagne.rad: genrev bowl ~1 unit tall, with stem and base
# Approximate each as: cone_frustum (bowl) + cylinder (stem) + cone_frustum (base)
function make_champagne(origin, scale)
  let s = scale
    # Bowl: approximately r=0.275*s at top, narrowing to stem
    cone_frustum(origin + vz(0.6*s), 0.025*s, 0.4*s, 0.275*s, material=goldc_mat)
    # Rim ring
    surface_ring(origin + vz(1.0*s), 0.26*s, 0.275*s, material=silverc_mat)
    # Stem: cylinder from z=0.2 to z=0.6
    cylinder(origin + vz(0.2*s), 0.025*s, 0.4*s, material=goldc_mat)
    # Base: wider at bottom
    cone_frustum(origin, 0.2*s, 0.2*s, 0.025*s, material=goldc_mat)
    # Liquid ring at top
    surface_circle(origin + vz(0.97*s), 0.245*s, material=clear_liquid_mat)
  end
end

# 8 champagne bowls in spiral at table (26.5, 3.5, 1.3)
let table_center = xyz(26.5, 3.5, 1.3)
  for i in 0:7
    let s = 1.5 * 0.9^i,
        angle = deg2rad(-30 + i * 50),
        r = 1.2 * s - 0.4,
        pos = table_center + vxyz(r * cos(angle), r * sin(angle), 0)
      make_champagne(pos, s)
    end
  end
end

## ─── Tea Cups ──────────────────────────────────────────────────────────────
# tea_cup.rad: genrev cup ~1 unit tall + saucer
# Approximate cup as cone_frustum + cylinder handle + disk saucer
function make_tea_cup(origin, scale, rot_z)
  let s = scale
    # Cup body: genrev hermite → approximate as cone_frustum
    # Bottom r≈0.2*s, top r≈0.85*s, height=1*s
    cone_frustum(origin, 0.2*s, 1.0*s, 0.85*s, material=china_mat)
    # Rim ring
    surface_ring(origin + vz(1.0*s), 0.8075*s, 0.85*s, material=china_mat)
    # Saucer: flat disk at base
    # genrev saucer from ~0 to 1.4*s radius, height ~0.15*s
    cone_frustum(origin, 0.5*s, 0.15*s, 1.4*s, material=china_mat)
    # Handle: small cylinder loop on the side
    let ha = deg2rad(rot_z),
        hdir = vxyz(cos(ha), sin(ha), 0)
      cylinder(origin + hdir * 0.85*s + vz(0.5*s), 0.07*s,
               origin + hdir * 0.85*s + vz(0.8*s),
               material=china_mat)
    end
  end
end

# "!xform -n c2 -s .25 -rz -15 -t 35 20 2.45 art_lib/tea_cup.rad"
make_tea_cup(xyz(35, 20, 2.45), 0.25, -15)
# "!xform -n c3 -s .25 -rz 45 -t 38 13 1.3 art_lib/tea_cup.rad"
make_tea_cup(xyz(38, 13, 1.3), 0.25, 45)

## ─── Goblet ────────────────────────────────────────────────────────────────
# "!xform -n c4 -t 18 11 1.3 art_lib/goblet.rad"
# goblet.rad: 2 genrev surfaces ~1 unit tall, pottery material
let go = xyz(18, 11, 1.3)
  # Top bowl: hermite curve → approximate as cone_frustum
  # z=0.04+0.4=0.44 to z=0.04+1.0=1.04, r from 0.025 to 0.25
  cone_frustum(go + vz(0.44), 0.025, 0.6, 0.25, material=pottery_mat)
  # Base: z=0.04 to z=0.04+0.4=0.44, r from 0.2 to 0.025
  cone_frustum(go + vz(0.04), 0.2, 0.4, 0.025, material=pottery_mat)
end

## ─── Phone ─────────────────────────────────────────────────────────────────
# "!xform -n p1 -s .0833 -rz 200 -t 38.2 20.2 2.45 art_lib/phone.rad"
# phone.rad: constructed in inches. After s=0.0833 → feet scale
# Simplified as a small box approximation
let ps = 0.0833,
    po = xyz(38.2, 20.2, 2.45)
  # Base: ~3.5" × 9" × 1.5" → 0.29' × 0.75' × 0.125'
  box(po + vxyz(-0.15, -0.375, 0), 0.29, 0.75, 0.125, material=accent_red_mat)
  # Receiver: raised on top
  box(po + vxyz(-0.08, -0.35, 0.125), 0.17, 0.7, 0.1, material=accent_red_mat)
end


## ═══════════════════════════════════════════════════════════════════════════
## FURNITURE
## ═══════════════════════════════════════════════════════════════════════════

## ─── Coffee Table (4' square) ──────────────────────────────────────────────
# tble4_c1.rad: 4' square, 1.3' tall. Insertion at floor center.
# genworm curved pipes → approximate with cylinders
function make_table4(origin)
  let o = origin
    # Table top: 3.8' × 3.8' × 0.05' at z=1.25
    box(o + vxyz(-1.9, -1.9, 1.25), 3.8, 3.8, 0.05, material=table_color_mat)
    # 4 straight pipe rails at z=1.25
    cylinder(o + vxyz(-1.5, 2, 1.25), 0.075, o + vxyz(1.5, 2, 1.25), material=pipe_color_mat)
    cylinder(o + vxyz(-1.5, -2, 1.25), 0.075, o + vxyz(1.5, -2, 1.25), material=pipe_color_mat)
    cylinder(o + vxyz(2, 1.5, 1.25), 0.075, o + vxyz(2, -1.5, 1.25), material=pipe_color_mat)
    cylinder(o + vxyz(-2, 1.5, 1.25), 0.075, o + vxyz(-2, -1.5, 1.25), material=pipe_color_mat)
    # 4 legs (genworm curved arcs → approximate as straight cylinders from rail end to floor)
    cylinder(o + vxyz(0, 2, 1.25), 0.075, o + vxyz(0, 2, 0), material=pipe_color_mat)
    cylinder(o + vxyz(0, -2, 1.25), 0.075, o + vxyz(0, -2, 0), material=pipe_color_mat)
    cylinder(o + vxyz(2, 0, 1.25), 0.075, o + vxyz(2, 0, 0), material=pipe_color_mat)
    cylinder(o + vxyz(-2, 0, 1.25), 0.075, o + vxyz(-2, 0, 0), material=pipe_color_mat)
  end
end

# "!xform -n 4_table -t 19 12.5 0 art_lib/tble4_c1.rad"
make_table4(xyz(19, 12.5, 0))
# "!xform -n 4_table -t 26.5 3.5 0 art_lib/tble4_c1.rad"
make_table4(xyz(26.5, 3.5, 0))

## ─── Side Table (2' square) ────────────────────────────────────────────────
function make_table2(origin)
  let o = origin
    box(o + vxyz(-0.9, -0.9, 1.25), 1.8, 1.8, 0.05, material=table_color_mat)
    # Pipes at z=1.25
    cylinder(o + vxyz(-0.5, 1, 1.25), 0.075, o + vxyz(0.5, 1, 1.25), material=pipe_color_mat)
    cylinder(o + vxyz(-0.5, -1, 1.25), 0.075, o + vxyz(0.5, -1, 1.25), material=pipe_color_mat)
    cylinder(o + vxyz(1, 0.5, 1.25), 0.075, o + vxyz(1, -0.5, 1.25), material=pipe_color_mat)
    cylinder(o + vxyz(-1, 0.5, 1.25), 0.075, o + vxyz(-1, -0.5, 1.25), material=pipe_color_mat)
    # Legs
    cylinder(o + vxyz(0, 1, 1.25), 0.075, o + vxyz(0.75, 1, 0), material=pipe_color_mat)
    cylinder(o + vxyz(0, -1, 1.25), 0.075, o + vxyz(0.75, -1, 0), material=pipe_color_mat)
    cylinder(o + vxyz(0, 1, 1.25), 0.075, o + vxyz(-0.75, 1, 0), material=pipe_color_mat)
    cylinder(o + vxyz(0, -1, 1.25), 0.075, o + vxyz(-0.75, -1, 0), material=pipe_color_mat)
  end
end

# "!xform -n 2_table -t 38 13.1 0 art_lib/tble2_c1.rad"
make_table2(xyz(38, 13.1, 0))

## ─── Sofa (single) ────────────────────────────────────────────────────────
# sofa1_c1.rad: 5' wide, insertion at floor center-front of seat
function make_sofa1(origin, rot_z)
  let cs = loc_from_o_vx_vy(
        origin,
        vxy(cos(deg2rad(rot_z)), sin(deg2rad(rot_z))),
        vxy(-sin(deg2rad(rot_z)), cos(deg2rad(rot_z))))
    # Seat cushion: 5' × 2' × 0.4' at z=1.3 (tilted -15°, approximated flat)
    box(cs + vxyz(-2.5, 0, 1.3), 5, 2, 0.4, material=seat_color_mat)
    # Back: 5' × 0.3' × 1.9' at z=1.3 (tilted -20°, approximated flat)
    box(cs + vxyz(-2.5, 1.75, 1.3), 5, 0.3, 1.9, material=seat_color_mat)
    # Side supports (simplified as vertical cylinders)
    cylinder(cs + vxyz(-2.54, 0, 0), 0.075, cs + vxyz(-2.54, 0, 2.5), material=pipe_color_mat)
    cylinder(cs + vxyz(2.54, 0, 0), 0.075, cs + vxyz(2.54, 0, 2.5), material=pipe_color_mat)
    # Legs
    cylinder(cs + vxyz(-2.42, 2.5, 1.65), 0.075, cs + vxyz(-2.42, 2.5, 0), material=pipe_color_mat)
    cylinder(cs + vxyz(2.42, 2.5, 1.65), 0.075, cs + vxyz(2.42, 2.5, 0), material=pipe_color_mat)
  end
end

# "!xform -n sofa1 -rz -90 -t 36.5 9.25 0 art_lib/sofa1_c1.rad"
make_sofa1(xyz(36.5, 9.25, 0), -90)

## ─── Sofa (back-to-back) ──────────────────────────────────────────────────
function make_sofa2(origin, rot_z)
  let cs = loc_from_o_vx_vy(
        origin,
        vxy(cos(deg2rad(rot_z)), sin(deg2rad(rot_z))),
        vxy(-sin(deg2rad(rot_z)), cos(deg2rad(rot_z))))
    # Front seat: 5' × 2' × 0.4' at z=1.3
    box(cs + vxyz(-2.5, -2.5, 1.3), 5, 2, 0.4, material=seat_color_mat)
    # Front back: 5' × 0.3' × 1.9'
    box(cs + vxyz(-2.5, -0.75, 1.3), 5, 0.3, 1.9, material=seat_color_mat)
    # Rear seat (mirrored)
    box(cs + vxyz(-2.5, 0.5, 1.3), 5, 2, 0.4, material=seat_color_mat)
    # Rear back
    box(cs + vxyz(-2.5, 0.45, 1.3), 5, 0.3, 1.9, material=seat_color_mat)
    # Side supports
    cylinder(cs + vxyz(-2.54, -2.8, 0), 0.075, cs + vxyz(-2.54, -2.8, 2.5), material=pipe_color_mat)
    cylinder(cs + vxyz(2.54, -2.8, 0), 0.075, cs + vxyz(2.54, -2.8, 2.5), material=pipe_color_mat)
    # Legs
    cylinder(cs + vxyz(-2.42, 0, 1.65), 0.075, cs + vxyz(-2.42, 0, 0), material=pipe_color_mat)
    cylinder(cs + vxyz(2.42, 0, 1.65), 0.075, cs + vxyz(2.42, 0, 0), material=pipe_color_mat)
  end
end

# "!xform -n sofa2 -rz 180 -t 14 12.5 0 art_lib/sofa2_c1.rad"
make_sofa2(xyz(14, 12.5, 0), 180)

## ─── Chair ─────────────────────────────────────────────────────────────────
function make_chair(origin, rot_z)
  let cs = loc_from_o_vx_vy(
        origin,
        vxy(cos(deg2rad(rot_z)), sin(deg2rad(rot_z))),
        vxy(-sin(deg2rad(rot_z)), cos(deg2rad(rot_z))))
    # Seat: 2' × 2' × 0.4' at z=1.3
    box(cs + vxyz(-1, 0, 1.3), 2, 2, 0.4, material=seat_color_mat)
    # Back: 2' × 0.3' × 1.9'
    box(cs + vxyz(-1, 1.75, 1.3), 2, 0.3, 1.9, material=seat_color_mat)
    # Side supports
    cylinder(cs + vxyz(-1.04, -0.3, 0), 0.075, cs + vxyz(-1.04, -0.3, 2.5), material=pipe_color_mat)
    cylinder(cs + vxyz(1.04, -0.3, 0), 0.075, cs + vxyz(1.04, -0.3, 2.5), material=pipe_color_mat)
    # Legs
    cylinder(cs + vxyz(-0.92, 2.5, 1.65), 0.075, cs + vxyz(-0.92, 2.5, 0), material=pipe_color_mat)
    cylinder(cs + vxyz(0.92, 2.5, 1.65), 0.075, cs + vxyz(0.92, 2.5, 0), material=pipe_color_mat)
  end
end

# "!xform -n chair -rz -90 -t 26.5 17.5 0 art_lib/chair_c1.rad"
make_chair(xyz(26.5, 17.5, 0), -90)
# "!xform -n chair -rz -135 -t 36.7 16.8 0 art_lib/chair_c1.rad"
make_chair(xyz(36.7, 16.8, 0), -135)

## ─── Desk ──────────────────────────────────────────────────────────────────
# "!xform -n desk -t 36.2 18 0 art_lib/desk_c1.rad"
# desk_c1.rad: 6' × 3' × 2.35' desk with curved front
let do_origin = xyz(36.2, 18, 0)
  # Desk top: 6.2' × 3' × 0.1' at z=2.35
  box(do_origin + vxyz(-3.1, 0, 2.35), 6.2, 3, 0.1, material=desktop_color_mat)
  # Drawer unit: 5.8' × 2.8' × 0.25' at z=2.05
  box(do_origin + vxyz(-2.9, 0.1, 2.05), 5.8, 2.8, 0.25, material=deskmid_color_mat)
  # Sculpted front panel (genprism 17-vertex profile → approximate as box)
  # Profile spans x=-2.9 to 2.9, z=1.3 to 2.05
  box(do_origin + vxyz(-2.9, 0.1, 1.3), 5.8, 0.1, 0.75, material=deskmid_color_mat)
  # Front support pipe (genworm → cylinder approximation)
  cylinder(do_origin + vxyz(-2, 0.1, 0), 0.075,
           do_origin + vxyz(2, 0.1, 0), material=pipe_color_mat)
  # Rear support pipe
  cylinder(do_origin + vxyz(-2, 2.9, 0), 0.075,
           do_origin + vxyz(2, 2.9, 0), material=pipe_color_mat)
end

## ─── Desk Chair ────────────────────────────────────────────────────────────
# "!xform -n desk_chair -rz 15 -t 36.2 21.25 0 art_lib/deskc_c1.rad"
let dco = xyz(36.2, 21.25, 0),
    rot = 15
  let cs = loc_from_o_vx_vy(
        dco,
        vxy(cos(deg2rad(rot)), sin(deg2rad(rot))),
        vxy(-sin(deg2rad(rot)), cos(deg2rad(rot))))
    # Seat: 2' × 2' × 0.4'
    box(cs + vxyz(-1, 0, 1.15), 2, 2, 0.4, material=seat_color_mat)
    # Lower back
    box(cs + vxyz(-1, 1.75, 1.3), 2, 0.3, 1.9, material=seat_color_mat)
    # Upper back/headrest
    box(cs + vxyz(-1, 2.2, 3.0), 2, 0.3, 0.75, material=seat_color_mat)
    # Arms (left and right)
    box(cs + vxyz(-1.17, 0.7, 2.0), 0.35, 1.5, 0.2, material=seat_color_mat)
    box(cs + vxyz(0.82, 0.7, 2.0), 0.35, 1.5, 0.2, material=seat_color_mat)
    # Supports and legs
    cylinder(cs + vxyz(-1.04, -0.3, 0), 0.075, cs + vxyz(-1.04, -0.3, 2.5), material=pipe_color_mat)
    cylinder(cs + vxyz(1.04, -0.3, 0), 0.075, cs + vxyz(1.04, -0.3, 2.5), material=pipe_color_mat)
    cylinder(cs + vxyz(-0.9, 2.5, 1.65), 0.075, cs + vxyz(-0.9, 2.5, 0), material=pipe_color_mat)
    cylinder(cs + vxyz(0.9, 2.5, 1.65), 0.075, cs + vxyz(0.9, 2.5, 0), material=pipe_color_mat)
  end
end


## ═══════════════════════════════════════════════════════════════════════════
## LIGHTING
## ═══════════════════════════════════════════════════════════════════════════

## ─── Track Lighting ────────────────────────────────────────────────────────
# 4 ceiling tracks (genbox enamel_white)
# South gallery track: "!genbox enamel_white track1 24 .15 .15 | xform -t 3 4.35 11.55"
box(xyz(3, 4.35, 11.55), 24, 0.15, 0.15, material=enamel_white_mat)
# North gallery track: "!genbox enamel_white track2 24 .15 .15 | xform -t 3 20.5 11.55"
box(xyz(3, 20.5, 11.55), 24, 0.15, 0.15, material=enamel_white_mat)
# South lobby track: "!genbox enamel_white track3 8 .15 .15 | xform -t 31 4.35 11.55"
box(xyz(31, 4.35, 11.55), 8, 0.15, 0.15, material=enamel_white_mat)
# North lobby track: "!genbox enamel_white track4 8 .15 .15 | xform -t 31 20.5 11.55"
box(xyz(31, 20.5, 11.55), 8, 0.15, 0.15, material=enamel_white_mat)

# Track light heads — approximate with spotlight or pointlight at each position
# Each tracklight: cylinder housing + light source pointed downward at various angles
# We use spotlights with approximate beam angles matching the IES data

# Gallery south track lights (from sc2light.rad positions)
# typeS4 = 150W Par38 flood (47.6° beam spread)
# typeS5 = 150W Par38 spot (10.5° beam spread)
# typeS1 = 90W Par38 flood
# typeS2 = 90W Par38 spot

let s4_hotspot = deg2rad(47.6/2),
    s5_hotspot = deg2rad(10.5/2),
    s1_hotspot = deg2rad(40.0/2),
    s2_hotspot = deg2rad(15.0/2)

  # South track — light positions with tilt (rx) and orientation (rz)
  # Each light: position on track, tilt angle from vertical, azimuth rotation
  # We compute direction from tilt and rotation
  function track_light(pos, rx, rz, hotspot)
    let tilt = deg2rad(rx),
        azim = deg2rad(rz),
        # Default light direction is (0, 0, -1), tilted by rx around x-axis
        # then rotated by rz around z-axis
        dir = vxyz(sin(tilt)*sin(azim), -sin(tilt)*cos(azim), -cos(tilt))
      spotlight(loc=pos, dir=dir, hotspot=hotspot, falloff=hotspot*1.5)
    end
  end

  # South gallery track (y=4.35, z=11.1)
  track_light(xyz(7, 4.35, 11.1), 45, 95, s4_hotspot)    # l1s: wave sw
  track_light(xyz(4, 4.35, 11.1), 2, 180, s5_hotspot)     # l2s: lotus
  track_light(xyz(6, 4.35, 11.1), 5, 92, s4_hotspot)      # l3s: lotus
  track_light(xyz(8, 4.35, 11.1), 40, -90, s4_hotspot)    # l4s: feather
  track_light(xyz(14, 4.35, 11.1), 10, -90, s4_hotspot)   # l5s: feather
  track_light(xyz(17, 4.35, 11.1), 15, 90, s4_hotspot)    # l6s: feather
  track_light(xyz(23.5, 4.35, 11.1), 42.5, -90, s4_hotspot) # l7s: wave se
  track_light(xyz(25, 4.35, 11.1), 5, -95, s5_hotspot)    # l8s: champagne

  # North gallery track (y=20.5, z=11.1)
  track_light(xyz(7, 20.5, 11.1), 45, 95, s4_hotspot)     # 21s: wave nw
  track_light(xyz(4, 20.5, 11.1), 35, -90, s4_hotspot)    # 22s: unfurled
  track_light(xyz(8, 20.5, 11.1), 0, 5, s5_hotspot)       # 23s: unfurled
  track_light(xyz(14.5, 20.5, 11.1), 45, 90, s2_hotspot)  # 24s: unfurled
  track_light(xyz(15.2, 20.5, 11.1), 40, -10, s1_hotspot) # 25s: cafe pic
  track_light(xyz(17, 20.5, 11.1), 30, 0, s1_hotspot)     # 26s: cafe pic
  track_light(xyz(19, 20.5, 11.1), 40, 10, s1_hotspot)    # 27s: cafe pic
  track_light(xyz(25.5, 20.5, 11.1), 35, 0, s4_hotspot)   # 28s: wave north

  # South lobby track (y=4.35, z=11.1)
  track_light(xyz(33, 4.35, 11.1), 0, 0, s4_hotspot)      # 31s
  track_light(xyz(37, 4.35, 11.1), 0, 0, s4_hotspot)      # 32s
  track_light(xyz(35, 4.35, 11.1), 40, 90, s4_hotspot)    # 33s
  track_light(xyz(36, 4.35, 11.1), 35, 0, s4_hotspot)     # 34s

  # North lobby track (y=20.5, z=11.1)
  track_light(xyz(33, 20.5, 11.1), 0, 0, s4_hotspot)      # 41s
  track_light(xyz(37, 20.5, 11.1), 0, 0, s4_hotspot)      # 42s
  track_light(xyz(35, 20.5, 11.1), 40, 90, s4_hotspot)    # 43s
end

## ─── Recessed Fluorescents ─────────────────────────────────────────────────
# Direct louvered: "!xform -n rlfd -rz 0 -t 11 12.5 7.99 -a 5 -t 4 0 0 -i 1 lib/typeRLF2.rad"
# 5 fixtures at (11, 12.5), (15, 12.5), (19, 12.5), (23, 12.5), (27, 12.5) at z=7.99
for i in 0:4
  pointlight(loc=xyz(11 + i*4, 12.5, 7.99), intensity=500.0, color=rgb(1, 1, 1))
end

# Indirect prismatic: north and south rows
# "!xform -n rlfin -rx 145 -t 6 15.75 8.35 -a 5 -t 4 0 0 -i 0 lib/typeRLF1.rad"
for i in 0:4
  pointlight(loc=xyz(6 + i*4, 15.75, 8.35), intensity=400.0, color=rgb(1, 1, 1))
end
# "!xform -n rlfis -rx -145 -t 6 9.25 8.35 -a 5 -t 4 0 0 -i 0 lib/typeRLF1.rad"
for i in 0:4
  pointlight(loc=xyz(6 + i*4, 9.25, 8.35), intensity=400.0, color=rgb(1, 1, 1))
end

## ─── Wall Sconces ──────────────────────────────────────────────────────────
# sconce.rad: asymmetric indirect luminaire with prism housing
# 6 instances at various positions
function make_sconce(pos, rot_z)
  let cs = loc_from_o_vx_vy(
        pos,
        vxy(cos(deg2rad(rot_z)), sin(deg2rad(rot_z))),
        vxy(-sin(deg2rad(rot_z)), cos(deg2rad(rot_z))))
    # Housing: triangular prism (simplified as box approximation)
    box(cs + vxyz(-0.5, -0.5, 0), 1, 0.4, 0.25, material=enamel_white_mat)
    # Light source behind housing
    pointlight(loc=pos + vz(0.1), intensity=300.0, color=rgb(1, 0.95, 0.9))
  end
end

# Gallery arch sconces
make_sconce(xyz(29.5, 8.5, 6.25), -90)
make_sconce(xyz(29.5, 16.5, 6.25), -90)
# Lobby arch sconces
make_sconce(xyz(30, 8.5, 6.25), 90)
make_sconce(xyz(30, 16.5, 6.25), 90)
# Front door sconces
make_sconce(xyz(30.7, 0.5, 6.25), 180)
make_sconce(xyz(38.7, 0.5, 6.25), 180)


## ═══════════════════════════════════════════════════════════════════════════
## CAMERA
## ═══════════════════════════════════════════════════════════════════════════

# Default view: archview.vf
# "rview -vtv -vp 37.4965 12.3819 5.51945 -vd -0.999239 -0.0174416 -0.0348995
#   -vu 0 0 1 -vh 60 -vv 45"
let vp = xyz(37.4965, 12.3819, 5.51945),
    vd = vxyz(-0.999239, -0.0174416, -0.0348995)
  set_view(vp, vp + vd, lens=60)
end

# Alternative views (uncomment to use):
# Inside view: vp 31.092 12.8497 4.41341, vd -0.965926 0.258819 0
# let vp = xyz(31.092, 12.8497, 4.41341), vd = vxyz(-0.965926, 0.258819, 0)
#   set_view(vp, vp + vd, lens=45)
# end

# Lotus view: vp 6.09866 6.13479 5.47025, vd -0.589104 -0.610035 -0.529919
# let vp = xyz(6.09866, 6.13479, 5.47025), vd = vxyz(-0.589104, -0.610035, -0.529919)
#   set_view(vp, vp + vd, lens=60)
# end

# Plan view: vp 20 12.5 41, vd 0 0 -1, vu 0 1 0
# set_view(xyz(20, 12.5, 41), xyz(20, 12.5, 40), lens=60)
