```@meta
CurrentModule = KhepriBase
```

# API Reference -- Infrastructure

Auto-generated documentation for the modules that sit under the
architectural layer: core types, coordinate systems, path algebra,
the backend protocol, and rendering scaffolding. Modules that have
their own dedicated narrative page — `WallGraph`, `Spaces`,
`Adjacencies`, the `Designs` submodule, `Constraints`,
`ConstraintLibrary`, `DesignLayout` — are *not* duplicated here;
follow the cross-references below.

Section grouping mirrors the `include(...)` order in
`src/KhepriBase.jl`.

## Core

### Types

```@autodocs
Modules = [KhepriBase]
Pages = ["Types.jl"]
```

### Parameters

```@autodocs
Modules = [KhepriBase]
Pages = ["Parameters.jl"]
```

### Utilities

See also the standalone [Utilities](utilities.md) page for narrative.

```@autodocs
Modules = [KhepriBase]
Pages = ["Utils.jl"]
```

### Coordinates

```@autodocs
Modules = [KhepriBase]
Pages = ["Coords.jl"]
```

### Regions

```@autodocs
Modules = [KhepriBase]
Pages = ["Regions.jl"]
```

### Paths

```@autodocs
Modules = [KhepriBase]
Pages = ["Paths.jl"]
```

### Geometry

```@autodocs
Modules = [KhepriBase]
Pages = ["Geometry.jl"]
```

## Backend Infrastructure

### Backend

```@autodocs
Modules = [KhepriBase]
Pages = ["Backend.jl"]
```

### Frontend

Documented on the [API -- Shapes](api.md) page (its *Frontend
Operations* section covers every `Frontend.jl` symbol).

### Backends

```@autodocs
Modules = [KhepriBase]
Pages = ["Backends.jl"]
```

### Primitives

```@autodocs
Modules = [KhepriBase]
Pages = ["Primitives.jl"]
```

### Camera

See [Camera & Rendering](camera_rendering.md) for the narrative treatment.

```@autodocs
Modules = [KhepriBase]
Pages = ["Camera.jl"]
```

### Simulation

```@autodocs
Modules = [KhepriBase]
Pages = ["Simulation.jl"]
```

### Plugin Management

```@autodocs
Modules = [KhepriBase]
Pages = ["PluginManagement.jl"]
```

### Backend Developer API

`KhepriBase.@import_backend_api` is the entry point for backend
packages: it splices every public-but-not-exported `KhepriBase`
name into the calling module so backends can extend `b_sphere`,
`b_trig`, etc., and otherwise see the full developer surface
without having to maintain an import list. See
[Implementing a Backend](implementing_backend.md) for usage context.

## Architectural Layer

### Architectural Materials

The canonical cross-backend material catalogue. Individual
`PbrMaterial` instances (`material_basic`, `material_metal`,
`material_glass`, …) are defined in `Shapes.jl` and documented under
[API -- Shapes](api.md); this module exposes an iterable view over
them for backend integrations and regression tests.

```@autodocs
Modules = [KhepriBase]
Pages = ["ArchMaterials.jl"]
```

### BIM Elements

Auto-documented in its own page: see [API -- BIM](api_bim.md).
Narrative walkthroughs live under the
[BIM Elements](../bim/horizontal_elements.md) section.

### Wall Graph

The wall-graph data model and its operations are documented in
[Wall Graph](../bim/wall_graph.md), which combines a conceptual
walkthrough with the full symbol reference.

### Space Layout

The multi-storey `Layout`, its `Storey`s and `Space`s, and every
accessor used by the constraint library and the declarative engine
live here. Narrative and examples are in the [Spaces](../bim/spaces.md)
BIM chapter; the imperative and declarative compilation paths are
covered in the [Layout Engine](layout-engine.md) reference.

```@autodocs
Modules = [KhepriBase]
Pages = ["Spaces.jl"]
```

### Adjacencies

Documented on a dedicated page: [Adjacencies](adjacencies.md).

### Design Tree (Level 2)

The declarative `SpaceDesc` tree and its compilation engine are
split across the design reference pages:

- [Design Types](design-types.md) — every `SpaceDesc` struct
- [Leaf Constructors](design-leaves.md) — `room`, `void`, `envelope`, `polar_envelope`
- [Combinators](design-combinators.md) — `beside`, `above`, `grid`, transforms, infix operators
- [Subdivision](design-subdivision.md) — top-down carving
- [Annotations](design-annotations.md) — connections, disconnects, no-window markers
- [Tree Queries](design-queries.md) — introspection
- [Layout Engine](layout-engine.md) — `layout(desc)` and `rectangular_boundary`

### Constraints & Validation

Documented on dedicated pages:
[Constraints (reference)](constraints.md) and
[Constraints (concept)](../concepts/constraints.md).

## Module

```@autodocs
Modules = [KhepriBase]
Pages = ["KhepriBase.jl"]
```
