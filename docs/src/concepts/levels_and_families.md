# Levels, Families, and Materials

KhepriBase organizes BIM elements around three foundational abstractions: **levels** define vertical positions in a building, **families** define the parametric properties of element types, and **materials** define visual appearance. Understanding these is essential before working with any BIM element.

## Levels

A **level** represents a horizontal reference plane at a given height, corresponding to a floor or story in a building. Most BIM elements are placed relative to levels rather than absolute Z coordinates, making it easy to adjust floor heights without modifying every element.

```julia
# Create levels at specific heights
ground = level(0)
first_floor = level(3.5)
second_floor = level(7.0)

# Use default_level() and upper_level() for quick setup
default_level(level(0))
floor1 = upper_level()                    # 3.0m above default (using default_level_to_level_height)
floor2 = upper_level(floor1)              # 3.0m above floor1
floor3 = upper_level(floor1, 4.0)         # 4.0m above floor1

# Numeric values auto-convert to levels
slab(rectangular_path(), 3.5)             # equivalent to slab(..., level(3.5))
```

### Key Functions

| Function | Description |
|----------|-------------|
| `level(height)` | Create a level at the given height |
| `default_level()` | Get/set the current default level |
| `upper_level(lvl, h)` | Create a level `h` meters above `lvl` (default: `default_level_to_level_height()` = 3.0) |
| `default_level_to_level_height()` | Get/set the default floor-to-floor height |
| `convert(Level, h)` | Convert a number to a level |

## Families

A **family** defines the parametric properties shared by all instances of a BIM element type. For example, a `SlabFamily` specifies thickness and materials; every `slab(...)` created with that family inherits those properties.

### The `@deffamily` Macro

Families are defined with the `@deffamily` macro, which generates five names from each family definition:

| Generated Name | Purpose | Example for `slab_family` |
|---------------|---------|---------------------------|
| `slab_family(...)` | Constructor — create a new family | `slab_family(thickness=0.3)` |
| `slab_family_element(family; ...)` | Instance — derive a variant from an existing family | `slab_family_element(default_slab_family(), thickness=0.25)` |
| `default_slab_family` | Parameter — holds the current default family | `default_slab_family()` |
| `SlabFamily` | Struct type | Used for dispatch and type annotations |
| `with_slab_family(f; ...)` | Scoped override — temporarily change the default | `with_slab_family(thickness=0.3) do ... end` |

### Creating and Using Families

```julia
# Use defaults
slab(rectangular_path())  # uses default_slab_family()

# Create a custom family
thick_slab = slab_family(thickness=0.4)
slab(rectangular_path(), family=thick_slab)

# Derive a variant from an existing family
thinner = slab_family_element(thick_slab, thickness=0.25)

# Temporarily override the default
with_slab_family(thickness=0.5) do
  slab(rectangular_path())    # uses 0.5m thick slab
end
```

### Backend-Specific Families

Khepri families are backend-portable by default. When a backend needs to map a Khepri family to a native implementation, use `set_backend_family`:

```julia
# Map Khepri's default wall family to a Revit-specific wall type
set_backend_family(default_wall_family(), revit,
  revit_wall_family("Generic - 200mm"))

# Map the default toilet to an OBJ model (works on any backend with OBJ support)
set_backend_family(default_toilet_family(), THR,
  obj_family("My_Toilet_Model"))
```

This keeps user code backend-agnostic while allowing each backend to use its native element types (Revit `.rfa` files, OBJ/MTL meshes, etc.).

`set_backend_family` also accepts a backend **type** instead of an instance, which is useful for registering families at module load time before any connection exists:

```julia
# Register using the backend type (no instance needed)
set_backend_family(default_door_family(), THR,
  obj_family("My_Door"))
```

### OBJ/MTL File Families

KhepriBase provides a generic `OBJFileFamily` for mapping any Khepri family to an OBJ/MTL 3D model. This works on **any backend** that implements `b_mesh_obj_fmt`: ThreeJS (native OBJ loader with materials), Blender (native import with full MTL support), Rhino (native import with full MTL support), and any future backend that adds the implementation.

```julia
# Create an OBJ backend family
obj_family(obj_name;     # relative subpath under resources/models/obj/
  scale=1.0,             # uniform scale factor
  rotation=0.0,          # rotation around vertical axis (radians)
  offset=vxyz(0,0,0),   # local offset in model coordinates
  y_is_up=false)         # true if OBJ uses Y-up convention (default: Z-up)
```

The `obj_name` is a relative subpath under `resources/models/obj/` (without the `.obj` extension). This supports both subfolder and flat layouts:

```julia
obj_family("My_Toilet/My_Toilet")  # → resources/models/obj/My_Toilet/My_Toilet.obj
obj_family("My_Toilet")            # → resources/models/obj/My_Toilet.obj
```

When an OBJ family is registered for a BIM element (toilet, sink, closet, table, chair, door, window), the element's `realize` function automatically loads the OBJ model with the correct position and orientation — including automatic wall alignment for doors and windows.

```julia
# Register OBJ models for fixtures (subfolder layout)
set_backend_family(default_toilet_family(), THR,
  obj_family("My_Toilet/My_Toilet", y_is_up=true))

set_backend_family(default_sink_family(), THR,
  obj_family("My_Sink/My_Sink"))

# Standalone OBJ placement (no BIM element)
cabinet = obj_family("Kitchen_Cabinet/Kitchen_Cabinet")
place_obj(cabinet, xy(3, 2))
place_obj_oriented(cabinet, xy(5, 2), vx(1))
place_obj_at_wall(cabinet, my_wall, 3.0, 0.0)
```

### Family Dispatch Chain

When a BIM element is realized, Khepri resolves the family through this chain:

1. **`backend_family(b, family)`** — looks up `family.implemented_as[typeof(b)]`. If not found, follows the `based_on` chain until a match is found or returns `family` itself.
2. **`backend_get_family_ref(b, family, backend_family)`** — gives the backend a chance to load resources (e.g., Revit loads `.rfa` files, OBJ families return themselves). Default: returns `backend_family` unchanged.
3. **`family_ref(b, family)`** — caches the result of step 2 in `family.ref[b]` so resources are loaded only once.

This means user code never needs to know which backend is active:

```julia
# Same code works on any backend
toilet(xy(1, 1), 0, floor_slab)

# The backend resolves to its own implementation:
#   Revit  → loads M_Toilet-Domestic-3D.rfa
#   ThreeJS → loads OBJ/MTL mesh
#   Others → default box geometry
```

### Overriding Default Families

Each `@deffamily` generates a `default_xxx_family` parameter. Override it per-backend to change the default for the entire project:

```julia
# All doors on the ThreeJS backend use this OBJ model
set_backend_family(default_door_family(), THR,
  obj_family("Custom_Door"))

# All toilets on Revit use this .rfa file
set_backend_family(default_toilet_family(), revit,
  revit_file_family("path/to/toilet.rfa"))
```

Project-specific variants can coexist with the defaults:

```julia
# Custom family that inherits from the default
sliding_door = door_family("Sliding Door", width=1.5, height=2.0)
set_backend_family(sliding_door, THR,
  obj_family("Sliding_Panel_Model"))

# Use it alongside default doors
add_door(wall1, xy(1, 0))                  # uses default
add_door(wall2, xy(1, 0), sliding_door)    # uses sliding
```

## Materials

Materials define the visual appearance of BIM elements. KhepriBase provides two systems: **named materials** for BIM element defaults, and **standard materials** for fine-grained appearance control.

### Predefined Named Materials

These are used as defaults in family definitions:

| Material | Typical Use |
|----------|-------------|
| `material_concrete` | Slabs, columns, structural elements |
| `material_plaster` | Wall surfaces, ceilings |
| `material_wood` | Doors, furniture |
| `material_metal` | Beams, railings, frames |
| `material_glass` | Windows, panels |

### Standard Materials

For precise appearance control, use `standard_material` based on the Filament PBR model:

```julia
# Simple colored material
red_mat = standard_material(base_color=rgba(1, 0, 0, 1))

# Metallic material
gold = standard_material(
  base_color=rgba(1, 0.84, 0, 1),
  metallic=1.0,
  roughness=0.3)

# Glass-like material
glass = standard_material(
  base_color=rgba(0.9, 0.9, 1.0, 0.3),
  transmission=0.8,
  roughness=0.0)

# Use with any shape
sphere(xyz(0, 0, 0), 5, material=red_mat)
```

Key `standard_material` properties:

| Property | Range | Description |
|----------|-------|-------------|
| `name` | `String` | Human-readable material name (used as identifier by some backends) |
| `base_color` | RGBA [0..1] | Diffuse albedo (dielectric) or specular color (metallic) |
| `metallic` | [0..1] | 0 = dielectric, 1 = conductor |
| `roughness` | [0..1] | 0 = smooth/mirror, 1 = rough/matte |
| `reflectance` | [0..1] | Fresnel reflectance at normal incidence |
| `transmission` | [0..1] | Transparency (0 = opaque, 1 = fully transparent) |
| `ior` | [1..n] | Index of refraction |
| `emissive` | RGBA | Emissive light (for glowing surfaces) |
| `data` | `BackendParameter` | Per-backend material overrides (see below) |

Each backend interprets these properties using the best available shading model in its renderer.

### Backend-Specific Material Overrides

The `data` field allows passing backend-specific material definitions that bypass
the PBR pipeline. When a material has a `data` entry for the active backend,
Khepri calls `b_get_material(backend, spec)` with the backend-specific value
instead of the default `b_standard_material` PBR path. The PBR properties
(`base_color`, etc.) serve as fallback for backends without an override.

```julia
# Radiance-specific material with PBR fallback
chrome = standard_material(
  name="chrome",
  base_color=rgba(0.8, 0.8, 0.8, 1.0),
  metallic=1.0,
  roughness=0.0,
  data=BackendParameter(
    RAD => radiance_metal_material("chrome", gray=0.8, specularity=0.9, roughness=0)))
```

`BackendParameter` maps backend types to arbitrary values. Each backend package
provides helper constructors (e.g., `radiance_plastic_material`,
`radiance_glass_material`) for its native material types.

## The BIM Proxy Pattern

All BIM elements in KhepriBase are defined using the `@defproxy` macro. A proxy is a lightweight Julia struct that stores the element's parameters without immediately creating geometry. Geometry is only generated ("realized") when the element is sent to a backend. For a detailed treatment of shapes and the `@defshape` macro (which extends `@defproxy` with material defaults), see [Shapes](shapes.md).

The realization pipeline:

1. **User calls** `slab(region, level, family)` -- creates a `Slab` proxy struct
2. **Backend dispatch** calls `realize(backend, slab)` -- invoked lazily
3. **`realize` calls** `b_slab(backend, region, level, family)` -- the backend-specific operation
4. **Backend returns** a reference (or set of references) to the created geometry

This two-phase approach means:
- The same `Slab` object can be realized in multiple backends
- Elements can be modified before realization
- Backend-specific details are isolated in `b_*` implementations

For the full details of the realize/ref protocol, including reference types, transaction modes, and storage, see [Realize & Ref Protocol](../reference/realize_and_ref.md).

## See Also

- [Shapes](shapes.md) -- The shape catalog and `@defshape` macro
- [Parameters](parameters.md) -- The parameter system used by `default_level`, `default_wall_family`, and other BIM defaults
- [Backends](backends.md) -- How backends implement `b_*` operations and the fallback chain
