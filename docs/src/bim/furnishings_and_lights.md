# Furnishings and Lights

Furnishings populate interior spaces with furniture and fixtures. Lights define illumination sources for rendering.

## Table

A table is a simple four-legged surface placed at a location.

### Signature

```julia
table(loc::Loc=u0(), level::Level=default_level(),
      family::TableFamily=default_table_family())

# With rotation angle
table(loc::Loc, angle::Real, level::Level=default_level(),
      family::TableFamily=default_table_family())
```

### TableFamily Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `length` | `1.6` | Table length |
| `width` | `0.9` | Table width |
| `height` | `0.75` | Table height |
| `top_thickness` | `0.05` | Tabletop thickness |
| `leg_thickness` | `0.05` | Leg cross-section size |
| `material` | `material_wood` | Material for top and legs |

### Examples

```julia
ground = level(0)

# Default table at origin
table(xy(3, 3), ground)

# Rotated table (angle in radians)
table(xy(3, 3), pi/4, ground)

# Custom desk
desk = table_family(length=1.8, width=0.8, height=0.72)
table(xy(5, 2), 0, ground, desk)
```

## Chair

A chair is a simple seated furniture element with back support.

### Signature

```julia
chair(loc::Loc=u0(), level::Level=default_level(),
      family::ChairFamily=default_chair_family())

# With rotation angle
chair(loc::Loc, angle::Real, level::Level=default_level(),
      family::ChairFamily=default_chair_family())
```

### ChairFamily Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `length` | `0.4` | Chair depth (front to back) |
| `width` | `0.4` | Chair width |
| `height` | `1.0` | Total height (including back) |
| `seat_height` | `0.5` | Height of the seat surface |
| `thickness` | `0.05` | Material thickness |
| `material` | `material_wood` | Chair material |

### Examples

```julia
ground = level(0)

# Chair facing forward
chair(xy(3, 2), ground)

# Chair rotated to face a table
chair(xy(3, 1.5), pi, ground)
```

## Table and Chairs

A composite element that places a table surrounded by chairs. The chairs are arranged around the table with configurable counts on each side.

### Signature

```julia
table_and_chairs(loc::Loc=u0(), level::Level=default_level(),
                 family::TableChairFamily=default_table_chair_family())

# With rotation angle
table_and_chairs(loc::Loc, angle::Real, level::Level=default_level(),
                 family::TableChairFamily=default_table_chair_family())
```

### TableChairFamily Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `table_family` | `default_table_family()` | Table configuration |
| `chair_family` | `default_chair_family()` | Chair configuration |
| `chairs_top` | `1` | Chairs on the far side |
| `chairs_bottom` | `1` | Chairs on the near side |
| `chairs_right` | `2` | Chairs on the right side |
| `chairs_left` | `2` | Chairs on the left side |
| `spacing` | `0.7` | Distance between chair centers |

### Examples

```julia
ground = level(0)

# Default table with 6 chairs (1+1+2+2)
table_and_chairs(xy(5, 5), ground)

# Conference table with more chairs
conference = table_chair_family(
  table_family=table_family(length=3.0, width=1.2),
  chairs_top=2,
  chairs_bottom=2,
  chairs_right=4,
  chairs_left=4,
  spacing=0.65)
table_and_chairs(xy(10, 5), 0, ground, conference)

# Rotated dining set
table_and_chairs(xy(3, 3), pi/6, ground)
```

## Fixtures

Fixtures are bathroom and utility elements that require a host slab (the floor they sit on).

### Toilet

```julia
toilet(cb::Loc=u0(), host::BIMShape=missing_slab(),
       family::ToiletFamily=default_toilet_family())

# With rotation angle
toilet(cb::Loc, angle::Real, host::BIMShape, family::ToiletFamily)
```

### Sink

```julia
sink(cb::Loc=u0(), host::BIMShape=missing_slab(),
     family::SinkFamily=default_sink_family())

# With rotation angle
sink(cb::Loc, angle::Real, host::BIMShape, family::SinkFamily)
```

### Closet

```julia
closet(cb::Loc=u0(), host::BIMShape=slab(),
       family::ClosetFamily=default_closet_family())

# With rotation angle
closet(cb::Loc, angle::Real, host::BIMShape, family::ClosetFamily)
```

`ToiletFamily`, `SinkFamily`, and `ClosetFamily` currently have no additional parameters beyond the name inherited from `Family`. Their geometry is determined by the backend implementation.

### Examples

```julia
ground = level(0)
floor = slab(rectangular_path(xy(0, 0), 3, 4), ground)

# Bathroom fixtures placed on the floor slab
toilet(xy(0.5, 3), 0, floor)
sink(xy(2, 3), 0, floor)
closet(xy(0.5, 0.5), pi/2, floor)
```

## Lights

Lights define illumination sources for rendering. They are BIM elements with position and intensity parameters.

### Point Light

A point light emits light uniformly in all directions with inverse-square attenuation. Intensity is specified in **candela**.

```julia
pointlight(loc::Loc=z(3), color::RGB=rgb(1,1,1),
           intensity::Real=1500.0, level::Level=default_level())
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loc` | `z(3)` | Position (relative to level) |
| `color` | `rgb(1,1,1)` | Light color |
| `intensity` | `1500.0` | Luminous intensity in candela |
| `level` | `default_level()` | Reference level |

```julia
# Ceiling light
pointlight(xyz(5, 5, 2.8), rgb(1, 0.95, 0.9), 800.0, level(0))

# Multiple lights in a grid
for x in 2:4:14, y in 2:4:10
  pointlight(xyz(x, y, 2.8), rgb(1, 1, 1), 600.0, level(0))
end
```

### Spotlight

A spotlight emits a cone of light defined by hotspot and falloff angles.

```julia
spotlight(loc::Loc=z(3), dir::Vec=vz(-1),
          hotspot::Real=pi/4, falloff::Real=pi/3)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loc` | `z(3)` | Light position |
| `dir` | `vz(-1)` | Direction vector |
| `hotspot` | `pi/4` | Inner cone angle (full intensity) |
| `falloff` | `pi/3` | Outer cone angle (light fades to zero) |

```julia
# Downlight aimed at the floor
spotlight(xyz(5, 5, 3), vz(-1), pi/6, pi/4)

# Angled spotlight
spotlight(xyz(0, 0, 3), vxyz(1, 1, -2), pi/8, pi/4)
```

### IES Light

An IES light uses a photometric data file (.ies) for physically accurate light distribution.

```julia
ieslight(file::String, loc::Loc=z(3), dir::Vec=vz(-1),
         alpha::Real=0, beta::Real=0, gamma::Real=0)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `file` | (required) | Path to the .ies photometric file |
| `loc` | `z(3)` | Light position |
| `dir` | `vz(-1)` | Aim direction |
| `alpha`, `beta`, `gamma` | `0` | Euler rotation angles for orientation |

```julia
# Photometric ceiling fixture
ieslight("fixtures/downlight.ies", xyz(5, 5, 2.8))

# Rotated wall sconce
ieslight("fixtures/sconce.ies", xyz(0, 3, 2.0), vy(-1), 0, pi/2, 0)
```
