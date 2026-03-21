# Horizontal Elements

Horizontal elements form the floors, roofs, and ceilings of a building. They share a common pattern: a **region** (closed path) defines the plan shape, a **level** sets the vertical position, and a **family** controls thickness and materials.

## Slab

A slab is a horizontal plate, typically used for floors. The slab extrudes downward from its level — the top surface sits at the level height.

### Signature

```julia
slab(region::Region=rectangular_path(),
     level::Level=default_level(),
     family::SlabFamily=default_slab_family())
```

### SlabFamily Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `thickness` | `0.2` | Structural thickness (meters) |
| `coating_thickness` | `0.0` | Additional coating on top |
| `bottom_material` | `material_concrete` | Underside material |
| `top_material` | `material_concrete` | Top surface material |
| `side_material` | `material_concrete` | Edge material |

The slab's elevation relative to the level is `coating_thickness - thickness` (i.e., it hangs below), and its total thickness is `coating_thickness + thickness`.

### Examples

```julia
# Rectangular slab at ground level
ground = level(0)
slab(surface_rectangular_path(xy(0, 0), 10, 8), ground)

# L-shaped slab using a polygonal path
l_shape = closed_polygonal_path([
  xy(0, 0), xy(10, 0), xy(10, 5),
  xy(6, 5), xy(6, 8), xy(0, 8)])
slab(l_shape, ground)

# Custom thickness
thick_floor = slab_family(thickness=0.35)
slab(rectangular_path(xy(0, 0), 10, 8), ground, thick_floor)
```

## Roof

A roof is structurally similar to a slab but sits on top of its level — the bottom surface is at the level height, and the roof extrudes upward.

### Signature

```julia
roof(region::Region=rectangular_path(),
     level::Level=default_level(),
     family::RoofFamily=default_roof_family())
```

### RoofFamily Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `thickness` | `0.2` | Structural thickness |
| `coating_thickness` | `0.0` | Additional coating |
| `bottom_material` | `material_concrete` | Underside material |
| `top_material` | `material_concrete` | Top surface material |
| `side_material` | `material_concrete` | Edge material |

### Examples

```julia
# Flat roof at the top of the building
roof_level = level(6.0)
roof(rectangular_path(xy(0, 0), 12, 10), roof_level)

# Roof with overhang (larger region than the floor below)
roof(rectangular_path(xy(-0.5, -0.5), 13, 11), roof_level)
```

## Ceiling

A ceiling is a thin finish element that hangs below its level. It represents the plaster or drywall surface seen from below.

### Signature

```julia
ceiling(region::Region=rectangular_path(),
        level::Level=default_level(),
        family::CeilingFamily=default_ceiling_family())
```

### CeilingFamily Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `thickness` | `0.02` | Ceiling thickness (thin by default) |
| `coating_thickness` | `0.0` | Additional coating |
| `bottom_material` | `material_plaster` | Visible surface from below |
| `top_material` | `material_concrete` | Hidden top surface |
| `side_material` | `material_plaster` | Edge material |

### Examples

```julia
# Ceiling below the first floor slab
first_floor = level(3.0)
ceiling(rectangular_path(xy(0, 0), 10, 8), first_floor)

# Custom ceiling material
painted_ceiling = ceiling_family(
  bottom_material=material(base_color=rgba(0.95, 0.95, 0.95, 1)))
ceiling(rectangular_path(xy(0, 0), 10, 8), first_floor, painted_ceiling)
```

## Panel

A panel is a thin surface element with no level dependency. Unlike slabs and roofs, a panel extrudes along its surface normal, making it suitable for glazed facades, partitions, or non-horizontal surfaces.

### Signature

```julia
panel(region::Region=rectangular_path(),
      family::PanelFamily=default_panel_family())
```

### PanelFamily Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `thickness` | `0.02` | Panel thickness |
| `right_material` | `material_glass` | Material on the right side |
| `left_material` | `material_glass` | Material on the left side |
| `side_material` | `material_glass` | Edge material |

### Examples

```julia
# Vertical glass panel (not on XY plane -- vertices span different heights)
panel(closed_polygonal_path([
  xyz(0, 0, 0), xyz(3, 0, 0), xyz(3, 0, 2.5), xyz(0, 0, 2.5)]))

# Opaque partition panel
partition = panel_family(
  thickness=0.05,
  right_material=material_plaster,
  left_material=material_plaster)
panel(rectangular_path(xy(5, 0), 4, 3), partition)
```

## Composing Horizontal Elements

A typical floor assembly combines a structural slab with a ceiling finish below:

```julia
ground = level(0)
first_floor = level(3.5)
roof_level = level(7.0)

floor_region = rectangular_path(xy(0, 0), 12, 10)

# Ground floor: just a slab
slab(floor_region, ground)

# First floor: slab + ceiling below it (visible from ground floor)
slab(floor_region, first_floor)
ceiling(floor_region, first_floor)

# Roof: roof element on top
roof(rectangular_path(xy(-0.3, -0.3), 12.6, 10.6), roof_level)
```
