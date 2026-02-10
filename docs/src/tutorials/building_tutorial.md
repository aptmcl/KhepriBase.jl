# Building a Complete Building

This tutorial walks through the creation of a 2-story office building using KhepriBase's BIM operations. The code is backend-portable — change the `using` line and backend call to switch between any Khepri backend.

## Setup

```julia
using KhepriThebes   # or KhepriAutoCAD, KhepriRevit, KhepriTikZ, etc.
using KhepriBase

backend(thebes)      # or autocad, revit, tikz, etc.
delete_all_shapes()
```

## Step 1: Define Levels

Every building starts with levels. We define three: ground floor, first floor, and roof.

```julia
ground      = level(0.0)
first_floor = level(3.5)
roof_level  = level(7.0)
```

## Step 2: Define Custom Families

Override default families to match the building's design intent.

```julia
# Exterior walls: 30cm thick
ext_wall = wall_family(thickness=0.3)

# Interior partitions: 15cm thick
int_wall = wall_family(thickness=0.15)

# Custom door and window sizes
main_door    = door_family(width=1.2, height=2.2)
office_door  = door_family(width=0.9, height=2.1)
tall_window  = window_family(width=1.4, height=1.6)
small_window = window_family(width=0.8, height=1.0)

# Structural columns
col_family = column_family(
  profile=rectangular_profile(0.3, 0.3))

# Stair with wider treads
office_stair = stair_family(width=1.2, riser_height=0.175, tread_depth=0.28)
```

## Step 3: Ground Floor Structure

### Floor Slab

```julia
# Building footprint: 16m x 12m
building_region = rectangular_path(xyz(0, 0, 0), 16, 12)
slab(building_region, ground)
```

### Exterior Walls

Use a closed path for the building perimeter. The offset defaults to `1/2` for closed paths, placing wall thickness to the interior.

```julia
exterior = wall(
  closed_polygonal_path([
    xyz(0, 0, 0), xyz(16, 0, 0), xyz(16, 12, 0), xyz(0, 12, 0)]),
  ground, first_floor, ext_wall)
```

### Doors and Windows on Exterior Walls

```julia
# Front facade (south wall, along y=0)
add_door(exterior, xy(2, 0), main_door)       # main entrance
add_window(exterior, xy(5, 1.0), tall_window)
add_window(exterior, xy(8, 1.0), tall_window)
add_window(exterior, xy(11, 1.0), tall_window)

# Right facade (east wall, along x=16 from corner, continues from path length 16)
add_window(exterior, xy(18, 1.0), tall_window)   # 2m from corner on east wall
add_window(exterior, xy(21, 1.0), tall_window)
add_window(exterior, xy(24, 1.0), tall_window)

# Back facade (north wall)
add_window(exterior, xy(30, 1.0), tall_window)
add_window(exterior, xy(33, 1.0), tall_window)
add_window(exterior, xy(36, 1.0), tall_window)

# Left facade (west wall)
add_window(exterior, xy(42, 1.0), tall_window)
add_window(exterior, xy(45, 1.0), tall_window)
```

### Interior Partitions

```julia
# Corridor wall dividing the floor at y=6
corridor_wall = wall(
  open_polygonal_path([xyz(0.3, 6, 0), xyz(15.7, 6, 0)]),
  ground, first_floor, int_wall)

# Office dividers on south side
wall(open_polygonal_path([xyz(8, 0.3, 0), xyz(8, 5.85, 0)]),
     ground, first_floor, int_wall)

# Office doors
add_door(corridor_wall, xy(3, 0), office_door)   # left office
add_door(corridor_wall, xy(10, 0), office_door)  # right office
```

### Columns

```julia
# Columns at 8m spacing along the corridor
for x in [0.15, 8, 15.85]
  for y in [0.15, 6, 11.85]
    column(xyz(x, y, 0), 0, ground, first_floor, col_family)
  end
end
```

### Stairwell

```julia
# Stair in the northeast corner
stair(xyz(13, 8, 0), vy(1), ground, first_floor, office_stair)

# Railings along the stair
railing(open_polygonal_path([xyz(13, 8, 0), xyz(13, 13.5, 3.5)]), ground)
railing(open_polygonal_path([xyz(14.2, 8, 0), xyz(14.2, 13.5, 3.5)]), ground)
```

## Step 4: First Floor

### Floor Slab and Ceiling Below

```julia
slab(building_region, first_floor)
ceiling(building_region, first_floor)
```

### Walls and Openings

The first floor has a similar layout. We repeat the pattern:

```julia
# Exterior walls
exterior_1f = wall(
  closed_polygonal_path([
    xyz(0, 0, 0), xyz(16, 0, 0), xyz(16, 12, 0), xyz(0, 12, 0)]),
  first_floor, roof_level, ext_wall)

# Windows on all facades (same pattern as ground floor)
add_window(exterior_1f, xy(5, 1.0), tall_window)
add_window(exterior_1f, xy(8, 1.0), tall_window)
add_window(exterior_1f, xy(11, 1.0), tall_window)

add_window(exterior_1f, xy(18, 1.0), tall_window)
add_window(exterior_1f, xy(21, 1.0), tall_window)
add_window(exterior_1f, xy(24, 1.0), tall_window)

add_window(exterior_1f, xy(30, 1.0), tall_window)
add_window(exterior_1f, xy(33, 1.0), tall_window)
add_window(exterior_1f, xy(36, 1.0), tall_window)

add_window(exterior_1f, xy(42, 1.0), tall_window)
add_window(exterior_1f, xy(45, 1.0), tall_window)

# Interior partitions
corridor_1f = wall(
  open_polygonal_path([xyz(0.3, 6, 0), xyz(15.7, 6, 0)]),
  first_floor, roof_level, int_wall)
add_door(corridor_1f, xy(3, 0), office_door)
add_door(corridor_1f, xy(10, 0), office_door)

# Columns
for x in [0.15, 8, 15.85]
  for y in [0.15, 6, 11.85]
    column(xyz(x, y, 0), 0, first_floor, roof_level, col_family)
  end
end
```

### Balcony with Railing

```julia
# Balcony slab extending from the south facade
balcony_region = rectangular_path(xyz(4, -2, 0), 8, 2)
slab(balcony_region, first_floor,
     slab_family(thickness=0.15))

# Railing around the balcony edge
railing(open_polygonal_path([
  xyz(4, -2, 0), xyz(12, -2, 0), xyz(12, 0, 0)]),
  first_floor)
railing(open_polygonal_path([xyz(4, 0, 0), xyz(4, -2, 0)]),
        first_floor)
```

## Step 5: Roof

```julia
# Roof slab with slight overhang
roof_region = rectangular_path(xyz(-0.3, -0.3, 0), 16.6, 12.6)
roof(roof_region, roof_level)

# Perimeter railing on the roof
railing(open_polygonal_path([
  xyz(0, 0, 0), xyz(16, 0, 0), xyz(16, 12, 0),
  xyz(0, 12, 0), xyz(0, 0, 0)]),
  roof_level,
  nothing,
  railing_family(height=1.1))
```

## Step 6: Interior Furnishings

```julia
# Ground floor reception: table and chairs
table_and_chairs(xyz(4, 9, 0), 0, ground)

# First floor conference room
conference = table_chair_family(
  table_family=table_family(length=2.4, width=1.0),
  chairs_top=1, chairs_bottom=1,
  chairs_right=3, chairs_left=3)
table_and_chairs(xyz(4, 9, 0), 0, first_floor, conference)

# Office desks
for (x, y) in [(3, 2), (3, 4), (10, 2), (10, 4)]
  table(xyz(x, y, 0), 0, ground, table_family(length=1.4, width=0.7))
  chair(xyz(x, y - 0.6, 0), 0, ground)
end
```

## Step 7: Lighting

```julia
# Ground floor ceiling lights
for x in 4:4:12, y in [3, 9]
  pointlight(xyz(x, y, 3.2), rgb(1, 0.98, 0.95), 800.0, ground)
end

# First floor ceiling lights
for x in 4:4:12, y in [3, 9]
  pointlight(xyz(x, y, 3.2), rgb(1, 0.98, 0.95), 800.0, first_floor)
end
```

## Step 8: Render

```julia
# Set camera position
set_view(xyz(25, -15, 12), xyz(8, 6, 3))

# Render the scene
render_view("office_building")
```

## Summary

This building uses the following BIM elements:

| Element | Count | Purpose |
|---------|-------|---------|
| `level` | 3 | Ground, first floor, roof |
| `slab` | 3+ | Floor plates, balcony |
| `roof` | 1 | Building roof |
| `ceiling` | 1 | Below first floor slab |
| `wall` | 6+ | Exterior and interior walls |
| `door` | 5+ | Entrance and office doors |
| `window` | 22+ | Facade glazing |
| `column` | 18 | Structural grid |
| `stair` | 1 | Vertical circulation |
| `railing` | 6+ | Stair, balcony, and roof |
| `table_and_chairs` | 2 | Reception and conference |
| `table` / `chair` | 8 | Office workstations |
| `pointlight` | 12+ | Interior lighting |

All elements are backend-portable. To render in a different backend, change only the `using` line and `backend()` call at the top.
