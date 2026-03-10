# Vertical Elements

Vertical elements define the enclosure and partitioning of a building. Walls provide structure and separation, doors and windows create openings, and curtain walls provide glazed facades.

## Wall

A wall extrudes a **path** vertically between two levels. The path can be open (a single wall segment or polyline) or closed (forming a room perimeter). Walls carry doors and windows as sub-elements.

### Signature

```julia
wall(path::Path=rectangular_path(),
     bottom_level::Level=default_level(),
     top_level::Level=upper_level(bottom_level),
     family::WallFamily=default_wall_family(),
     offset::Real=is_closed_path(path) ? 1/2 : 0,
     doors::Shapes=Shape[],
     windows::Shapes=Shape[])
```

### WallFamily Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `thickness` | `0.2` | Wall core thickness |
| `left_coating_thickness` | `0.0` | Additional coating on left side |
| `right_coating_thickness` | `0.0` | Additional coating on right side |
| `right_material` | `material_plaster` | Right-side material |
| `left_material` | `material_plaster` | Left-side material |
| `side_material` | `material_plaster` | Top/bottom edge material |

### Offset Semantics

The `offset` parameter controls how the wall thickness is distributed relative to the path:

- **`offset=0`** (default for open paths): wall is centered on the path — half the thickness on each side.
- **`offset=1/2`** (default for closed paths): the full thickness is placed to the left (inside) of the path — appropriate for room perimeters where the path traces the exterior face.

"Left" and "right" are defined relative to an observer looking along the path direction.

### Examples

```julia
ground = level(0)
first_floor = level(3.5)

# Simple straight wall
wall(open_polygonal_path([xy(0, 0), xy(10, 0)]),
     ground, first_floor)

# Room perimeter (closed path — offset defaults to 1/2)
wall(closed_polygonal_path([
  xy(0, 0), xy(10, 0), xy(10, 8), xy(0, 8)]),
  ground, first_floor)

# Custom wall thickness
exterior_wall = wall_family(thickness=0.3)
wall(open_polygonal_path([xy(0, 0), xy(10, 0)]),
     ground, first_floor, exterior_wall)
```

### Joining Walls

Two walls with the same levels, family, and offset can be joined into a single continuous wall:

```julia
w1 = wall(open_polygonal_path([xy(0, 0), xy(5, 0)]),
          ground, first_floor)
w2 = wall(open_polygonal_path([xy(5, 0), xy(5, 8)]),
          ground, first_floor)
joined = join_walls(w1, w2)
```

### Incremental Construction

Use `with_wall` to create a wall and add openings in a transaction:

```julia
with_wall(open_polygonal_path([xy(0, 0), xy(10, 0)]),
          ground, first_floor) do w
  add_door(w, xy(2, 0))
  add_window(w, xy(5, 1.0))
end
```

### Boolean vs. Polygonal Decomposition

Walls with openings can be realized in two ways depending on the backend:

- **Boolean operations** (`HasBooleanOps{true}`): wall solid minus opening solids — used by backends with CSG support (AutoCAD, Revit).
- **Polygonal decomposition** (`HasBooleanOps{false}`, the default): wall geometry is split into polygonal regions around openings — used by backends without CSG (GL, Thebes, Blender).

This is handled automatically; user code is the same for both.

## Door

A door is an opening placed within a wall. Doors are not standalone elements — they exist as children of a wall.

### Signature

```julia
door(wall::Wall, loc::Loc=u0(), flip_x::Bool=false, flip_y::Bool=false,
     angle::Real=0, family::DoorFamily=default_door_family())
```

### DoorFamily Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `width` | `1.0` | Door opening width |
| `height` | `2.0` | Door opening height |
| `thickness` | `0.05` | Door panel thickness |
| `frame` | `default_frame_family()` | Frame profile and material |
| `right_material` | `material_wood` | Right face material |
| `left_material` | `material_wood` | Left face material |
| `side_material` | `material_wood` | Edge material |

### Location Semantics

The `loc` parameter uses a 2D coordinate relative to the wall's base path:
- **`loc.x`**: distance along the path from the start (in meters)
- **`loc.y`**: height above the bottom level

### Adding Doors

```julia
# Add a door to an existing wall
w = wall(open_polygonal_path([xy(0, 0), xy(10, 0)]),
         ground, first_floor)
add_door(w, xy(2, 0))                                    # default family
add_door(w, xy(6, 0), door_family(width=0.9, height=2.1)) # custom size
```

## Window

A window follows the same pattern as a door but with different default dimensions and materials.

### Signature

```julia
window(wall::Wall, loc::Loc=u0(), flip_x::Bool=false, flip_y::Bool=false,
       angle::Real=0, family::WindowFamily=default_window_family())
```

### WindowFamily Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `width` | `1.0` | Window opening width |
| `height` | `1.0` | Window opening height |
| `thickness` | `0.05` | Window pane thickness |
| `frame` | `default_frame_family()` | Frame profile and material |
| `right_material` | `material_glass` | Right face material |
| `left_material` | `material_glass` | Left face material |
| `side_material` | `material_glass` | Edge material |

### Adding Windows

```julia
# Window at 1m above floor, 4m along the wall
add_window(w, xy(4, 1.0))

# Large window
large_window = window_family(width=2.0, height=1.5)
add_window(w, xy(4, 0.8), large_window)
```

### Backend-Specific Door and Window Models

Backends can provide custom realizations for doors and windows. When a family has a backend-specific implementation registered via `set_backend_family`, the backend's `realize` method uses it instead of the default flat-panel geometry.

For example, on any backend that supports OBJ loading (ThreeJS, Blender, Rhino), doors and windows can be rendered as 3D OBJ models that **automatically orient to the wall direction**:

```julia
# Register an OBJ model for a door family
my_door = door_family("Custom Door", width=0.9, height=2.1)
set_backend_family(my_door, THR,
  obj_family("My_Door_Model",
    scale=1.0,
    rotation=0.0,
    offset=vxyz(0, 0, 0)))

# The door mesh automatically rotates to match any wall direction
w1 = wall(open_polygonal_path([xy(0, 0), xy(5, 0)]), ground, first_floor)
add_door(w1, xy(1, 0), my_door)

w2 = wall(open_polygonal_path([xy(0, 0), xy(3, 4)]), ground, first_floor)
add_door(w2, xy(1, 0), my_door)  # same door, aligned to diagonal wall
```

The automatic wall alignment works by computing a local coordinate system from the wall's path: the X-axis follows the wall tangent, the Y-axis follows the wall normal, and the Z-axis points up. The mesh is oriented, scaled, and offset within this coordinate system.

If no backend-specific family is registered, doors and windows fall back to the default realization (a flat panel with frame sweep).

## Curtain Wall

A curtain wall is a glazed facade system composed of panels divided by mullions (vertical) and transoms (horizontal), surrounded by a boundary frame.

### Signature

```julia
curtain_wall(path::Path=rectangular_path(),
             bottom_level::Level=default_level(),
             top_level::Level=upper_level(bottom_level),
             family::CurtainWallFamily=default_curtain_wall_family(),
             offset::Real=0.0)

# Two-point convenience form
curtain_wall(p0::Loc, p1::Loc; bottom_level, top_level, family, offset)
```

### CurtainWallFamily Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_panel_dx` | `1.0` | Maximum panel width (mullion spacing) |
| `max_panel_dy` | `2.0` | Maximum panel height (transom spacing) |
| `panel` | `panel_family(thickness=0.05)` | Glass panel family |
| `boundary_frame` | width=0.1, depth=0.1 | Perimeter frame |
| `mullion_frame` | width=0.08, depth=0.09 | Vertical divider frame |
| `transom_frame` | width=0.06, depth=0.1 | Horizontal divider frame |

Each frame sub-family (`CurtainWallFrameFamily`) has `width`, `depth`, `depth_offset`, and material properties.

### Examples

```julia
ground = level(0)
first_floor = level(3.5)

# Simple curtain wall between two points
curtain_wall(xy(0, 0), xy(10, 0),
             bottom_level=ground, top_level=first_floor)

# Curtain wall with custom panel spacing
fine_grid = curtain_wall_family(max_panel_dx=0.8, max_panel_dy=1.5)
curtain_wall(open_polygonal_path([xy(0, 0), xy(10, 0)]),
             ground, first_floor, fine_grid)
```

## Integration Example

A room with walls, a door, and windows:

```julia
ground = level(0)
first_floor = level(3.0)

# Room perimeter
w = wall(closed_polygonal_path([
  xy(0, 0), xy(8, 0), xy(8, 6), xy(0, 6)]),
  ground, first_floor)

# Front wall openings: door at 1m, windows at 3.5m and 5.5m
add_door(w, xy(1, 0))
add_window(w, xy(3.5, 1.0), window_family(width=1.2, height=1.2))
add_window(w, xy(5.5, 1.0), window_family(width=1.2, height=1.2))

# Floor slab
slab(rectangular_path(xy(0, 0), 8, 6), ground)
```
