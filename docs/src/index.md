```@meta
CurrentModule = KhepriBase
```

# KhepriBase

KhepriBase is the core library of the [Khepri](https://github.com/aptmcl/Khepri) algorithmic design framework. It defines the portable abstractions -- shapes, BIM elements, materials, levels, families, and backend operations -- that allow the same design script to produce equivalent geometry across AutoCAD, Revit, Blender, Rhino, TikZ, and many other backends.

## What is Khepri?

Khepri is a Julia-based algorithmic design system built around one principle: **the same script produces equivalent designs in every backend**. You write your design once using KhepriBase's abstractions, then switch between 16+ backends by changing a single `using` line. KhepriBase provides the shapes, coordinates, paths, BIM elements, materials, and rendering operations; each backend translates those into native geometry.

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

## Reading Roadmap

### New to Khepri?

Start here for a ground-up introduction:

1. [Installation and Setup](getting_started/installation.md) -- install KhepriBase and a backend, run your first shape
2. [Coordinates and Vectors](getting_started/coordinates.md) -- locations, vectors, coordinate systems
3. [Paths](getting_started/paths.md) -- open/closed paths, regions, profiles
4. [Shapes](concepts/shapes.md) -- the shape catalog (points, curves, surfaces, solids, CSG)
5. [Levels, Families, and Materials](concepts/levels_and_families.md) -- BIM foundations

### BIM User?

Jump into building design:

1. [Levels, Families, and Materials](concepts/levels_and_families.md) -- levels, families, standard materials
2. [Horizontal Elements](bim/horizontal_elements.md) -- slabs, roofs, ceilings, panels
3. [Vertical Elements](bim/vertical_elements.md) -- walls, doors, windows, curtain walls
4. [Structural Elements](bim/structural_elements.md) -- beams, columns, trusses
5. [Circulation](bim/circulation.md) -- stairs, ramps, railings
6. [Furnishings and Lights](bim/furnishings_and_lights.md) -- tables, chairs, lights
7. [Spaces](bim/spaces.md) -- space-first layout design
8. [Building a Complete Building](tutorials/building_tutorial.md) -- end-to-end tutorial

### Backend Developer?

Implement a new Khepri backend:

1. [Backends](concepts/backends.md) -- `Backend{K,T}`, dispatch, fallback chain
2. [Parameters](concepts/parameters.md) -- the parameter system
3. [Realize & Ref Protocol](reference/realize_and_ref.md) -- the lazy proxy realization system
4. [Implementing a Backend](reference/implementing_backend.md) -- step-by-step backend guide
5. [Backend Operations Matrix](reference/backend_operations.md) -- which `b_*` operations to implement

## Documentation Guide

### Getting Started
- **[Installation and Setup](getting_started/installation.md)** -- Install, configure, hello-sphere
- **[Coordinates and Vectors](getting_started/coordinates.md)** -- `Loc` vs `Vec`, constructors, coordinate spaces, arithmetic
- **[Paths](getting_started/paths.md)** -- Open/closed paths, regions, profiles, path operations

### Concepts
- **[Levels, Families, and Materials](concepts/levels_and_families.md)** -- Levels, the family system, standard materials, the proxy pattern
- **[Shapes](concepts/shapes.md)** -- Shape proxy system, dimensionality hierarchy, CSG operations
- **[Parameters](concepts/parameters.md)** -- `Parameter`, `GlobalParameter`, `OptionParameter`, `LazyParameter`, `with` scoping
- **[Backends](concepts/backends.md)** -- `Backend{K,T}`, backend types, multi-backend mode, `b_*` dispatch

### BIM Elements
- **[Horizontal Elements](bim/horizontal_elements.md)** -- Slab, Roof, Ceiling, Panel
- **[Vertical Elements](bim/vertical_elements.md)** -- Wall, Door, Window, Curtain Wall
- **[Structural Elements](bim/structural_elements.md)** -- Beam, Column, Free Column, Trusses
- **[Circulation](bim/circulation.md)** -- Stair, Spiral Stair, Stair Landing, Ramp, Railing
- **[Furnishings and Lights](bim/furnishings_and_lights.md)** -- Table, Chair, Fixtures, Point/Spot/IES Lights
- **[Spaces](bim/spaces.md)** -- Space, FloorPlan, BuildResult, validation rules

### Tutorials
- **[Building a Complete Building](tutorials/building_tutorial.md)** -- End-to-end: a 2-story office building from levels to render
- **[Space-First Layout Design](tutorials/spaces_tutorial.md)** -- Define rooms and connections, auto-generate walls
- **[Rendering and Animation](tutorials/rendering_tutorial.md)** -- Camera, render settings, film workflow
- **[Algorithmic Design](tutorials/algorithmic_tutorial.md)** -- Parametric patterns, subdivision, recursive structures

### Reference
- **[Shapes and Geometry](reference/shapes_geometry.md)** -- Complete API tables for all shapes, CSG, geometric utilities
- **[Utilities](reference/utilities.md)** -- `division`, `map_division`, Grasshopper-compat functions, random, color
- **[Camera and Rendering](reference/camera_rendering.md)** -- Render parameters, film parameters, camera functions
- **[Backend Operations Matrix](reference/backend_operations.md)** -- All `b_*` operations and which backends implement them
- **[Realize and Ref Protocol](reference/realize_and_ref.md)** -- The lazy proxy realization system in detail
- **[Implementing a Backend](reference/implementing_backend.md)** -- Backend developer guide: struct, void_ref, operations, checklist

## API Index

```@index
```
