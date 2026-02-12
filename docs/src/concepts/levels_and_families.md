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

Khepri families are backend-portable by default. When a backend (e.g., Revit) needs to map a Khepri family to a native BIM family, use `set_backend_family`:

```julia
# Map Khepri's default wall family to a Revit-specific wall type
set_backend_family(default_wall_family(), revit,
  revit_wall_family("Generic - 200mm"))
```

This keeps user code backend-agnostic while allowing each backend to use its native element types.

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
| `base_color` | RGBA [0..1] | Diffuse albedo (dielectric) or specular color (metallic) |
| `metallic` | [0..1] | 0 = dielectric, 1 = conductor |
| `roughness` | [0..1] | 0 = smooth/mirror, 1 = rough/matte |
| `reflectance` | [0..1] | Fresnel reflectance at normal incidence |
| `transmission` | [0..1] | Transparency (0 = opaque, 1 = fully transparent) |
| `ior` | [1..n] | Index of refraction |
| `emissive` | RGBA | Emissive light (for glowing surfaces) |

Each backend interprets these properties using the best available shading model in its renderer.

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
