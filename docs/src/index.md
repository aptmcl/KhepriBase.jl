```@meta
CurrentModule = KhepriBase
```

# KhepriBase

KhepriBase is the core library of the [Khepri](https://github.com/aptmcl/Khepri) algorithmic design framework. It defines the portable abstractions — shapes, BIM elements, materials, levels, families, and backend operations — that allow the same design script to produce equivalent geometry across AutoCAD, Revit, Blender, Rhino, TikZ, and many other backends.

## Architecture

KhepriBase is organized around four layers:

1. **Proxies** (`@defproxy`): Lightweight structs that store element parameters without creating geometry. Every shape and BIM element is a proxy.
2. **Families** (`@deffamily`): Parametric type definitions that capture the properties shared by all instances of an element type (thickness, profile, materials).
3. **Backend operations** (`b_*`): The contract between KhepriBase and backends. Each backend implements a subset of ~136 operations like `b_slab`, `b_wall`, `b_sphere`.
4. **Realization**: When a proxy is sent to a backend, `realize(backend, proxy)` dispatches to the appropriate `b_*` operation, producing geometry.

## Quick Start

Create a simple room with a floor slab and four walls:

```julia
using KhepriThebes   # or any Khepri backend
using KhepriBase

backend(thebes)

ground = level(0)
first_floor = level(3.0)

# Floor slab
slab(rectangular_path(xyz(0, 0, 0), 8, 6), ground)

# Walls (closed path = room perimeter)
w = wall(
  closed_polygonal_path([
    xyz(0, 0, 0), xyz(8, 0, 0), xyz(8, 6, 0), xyz(0, 6, 0)]),
  ground, first_floor)

# Add a door and window
add_door(w, xy(1, 0))
add_window(w, xy(4, 1.0), window_family(width=1.5, height=1.2))

render_view("simple_room")
```

## Documentation Guide

### [Levels, Families, and Materials](concepts/levels_and_families.md)
Foundational abstractions — levels, the family system, materials, and the proxy pattern.

### BIM Elements
- **[Horizontal Elements](bim/horizontal_elements.md)** — Slab, Roof, Ceiling, Panel
- **[Vertical Elements](bim/vertical_elements.md)** — Wall, Door, Window, Curtain Wall
- **[Structural Elements](bim/structural_elements.md)** — Beam, Column, Free Column, Trusses
- **[Circulation](bim/circulation.md)** — Stair, Spiral Stair, Stair Landing, Ramp, Railing
- **[Furnishings and Lights](bim/furnishings_and_lights.md)** — Table, Chair, Fixtures, Point/Spot/IES Lights

### [Building a Complete Building](tutorials/building_tutorial.md)
End-to-end tutorial: a 2-story office building from levels to render.

### [Backend Operations Matrix](reference/backend_operations.md)
Complete reference of all `b_*` operations and which backends implement them.

## API Index

```@index
```
