# Implementing a Backend

This guide walks through the steps required to create a new Khepri backend.  A
backend is a Julia type that maps abstract Khepri operations to a concrete
representation -- a CAD application, a file format, or a rendering engine.

## 1. Choosing a Backend Base Type

KhepriBase provides several abstract base types.  Pick the one closest to your
communication model:

| Base Type | Typical Use | Examples |
|-----------|-------------|---------|
| `SocketBackend{K,T}` | TCP socket to a C#/C++/Python plugin running inside a CAD application | AutoCAD, Revit, Rhino, Unity |
| `WebSocketBackend{K,T}` | Browser-based or WebSocket-connected backends | Three.js, Xeokit, MeshCat |
| `IOBackend{K,T,E}` (via `LocalBackend`) | File-output backends; shapes are collected locally and written to a file | TikZ, POVRay, Radiance |
| Custom `Backend{K,T}` | Local Julia backends that render or display directly | Makie, Thebes |

`K` is a singleton key type that identifies the backend (e.g., `Val{:ACAD}`)
and `T` is the native reference type used by the backend (e.g., `Int32`,
`UInt128`, or a custom handle type).

## 2. Backend Struct Definition

Every backend struct **must** include a `refs::References{K,T}` field.
`References` holds six dictionaries mapping proxy types to their backend
references:

```julia
struct References{K,T}
  shapes::Dict{Shape, GenericRef{K,T}}
  materials::Dict{Material, GenericRef{K,T}}
  layers::Dict{Layer, GenericRef{K,T}}
  annotations::Dict{Annotation, GenericRef{K,T}}
  families::Dict{Family, GenericRef{K,T}}
  levels::Dict{Level, GenericRef{K,T}}
end
```

The standard backend structs (`SocketBackend`, `WebSocketBackend`, `IOBackend`)
already include this field.  If you define a fully custom struct, add it
explicitly:

```julia
mutable struct MyBackend <: Backend{Val{:MY}, Int}
  refs::References{Val{:MY}, Int}
  # ... other fields ...
end
```

## 3. `void_ref`

**Critical:** `void_ref(b)` must return a raw value of type `T`, **not** wrapped
in `NativeRef{K,T}`.  It serves as the "null reference" sentinel.

```julia
KhepriBase.void_ref(b::MyBackend) = -1        # T = Int
KhepriBase.void_ref(b::MyBackend) = UInt128(0) # T = UInt128
```

## 4. `view_type` -- FrontendView vs BackendView

Backends that manage camera state locally (file-output, Julia-only renderers)
should declare:

```julia
KhepriBase.view_type(::Type{MyBackend}) = FrontendView()
```

This makes `b_set_view` / `b_get_view` store camera parameters in the
backend's `view::View` field rather than sending them over the wire.

Socket and WebSocket backends that delegate view management to the remote
application use the default `BackendView()` (no override needed).

## 5. Connection Protocol (SocketBackend)

`SocketBackend` provides a complete connection lifecycle:

1. `start_connection(b)` -- attempts `connect(b.port)` with up to 10 retries.
2. `before_connecting(b)` -- hook called before connecting (default: no-op).
3. `after_connecting(b)` -- hook called after connecting (default: no-op).
4. `failed_connecting(b)` -- called after all retries fail.
5. `retry_connecting(b)` -- called between retries (default: prints message,
   sleeps 8 seconds).

Override these hooks to customize the connection flow, e.g., to launch the CAD
application automatically.

## 6. Remote Procedure Calls

### `@remote_api` and `@remote`

For socket-based backends, remote functions are declared with `@remote_api`,
which parses C#/C++/Python/JavaScript signatures and generates encode/decode
logic:

```julia
const my_api = @remote_api :CS """
  int Sphere(Point3d c, double r)
  void DeleteAll()
"""
```

The result is a named tuple of `RemoteFunctionInfo` structs.  At backend
construction time, call `remote_functions(my_api)` to create the live
`RemoteFunction` objects (which cache opcodes after the first call).

Inside `b_*` implementations, use the `@remote` macro to invoke them:

```julia
b_sphere(b::MySocketBackend, c, r, mat) =
  @remote(b, Sphere(c, r))
```

### `encode` / `decode`

The RPC system serializes arguments and deserializes results using
`encode(namespace, type_val, io, value)` and `decode(namespace, type_val, io)`.
Built-in type support includes:

| Type Tag | Julia Type | Wire Format |
|----------|-----------|-------------|
| `:int` | `Int32` | 4 bytes LE |
| `:long` | `Int64` | 8 bytes LE |
| `:float` | `Float32` / `Float64` | 4 or 8 bytes LE (language-dependent) |
| `:double` | `Float64` | 8 bytes LE |
| `:string` | `String` | Length-prefixed (7-bit encoded length, C# format) |
| `:bool` | `Bool` | 1 byte |
| `:byte` | `UInt8` | 1 byte |
| `:void` / `:None` | -- | 1 byte status |
| `:RGB` / `:RGBA` | `RGB` / `RGBA` | 3 or 4 floats |
| `Vector{T}` | arrays | Size-prefixed |

For custom types, define `encode` and `decode` methods specialized on your
namespace `Val{:XX}`.

## 7. Which `b_*` Operations to Implement

Backend operations are layered: higher-level operations have default
implementations that decompose into lower-level ones.  Implement the level
appropriate for your backend's capabilities.

### Tier 0 -- Essential (curves)

These are the most fundamental operations. Without them, nothing works.

| Operation | Signature | Fallback |
|-----------|-----------|----------|
| `b_point` | `b_point(b, p, mat)` | None -- must implement |
| `b_line` | `b_line(b, ps, mat)` | None -- must implement |
| `b_polygon` | `b_polygon(b, ps, mat)` | `b_line(b, [ps..., ps[1]], mat)` |
| `b_spline` | `b_spline(b, ps, mat)` | Falls back to NURBS curve, then polyline |
| `b_closed_spline` | `b_closed_spline(b, ps, mat)` | Falls back to NURBS curve |
| `b_circle` | `b_circle(b, c, r, mat)` | `b_closed_spline` with 32-gon |
| `b_arc` | `b_arc(b, c, r, a, da, mat)` | `b_spline` with sampled points |
| `b_ellipse` | `b_ellipse(b, c, rx, ry, mat)` | `b_closed_spline` with 64 samples |
| `b_rectangle` | `b_rectangle(b, c, dx, dy, mat)` | `b_polygon` with 4 corners |

### Tier 1 -- Triangles (foundations for surfaces and solids)

| Operation | Signature | Fallback |
|-----------|-----------|----------|
| `b_trig` | `b_trig(b, p1, p2, p3)` | None -- must implement for surface/solid support |
| `b_quad` | `b_quad(b, p1, p2, p3, p4, mat)` | Two `b_trig` calls |
| `b_ngon` | `b_ngon(b, ps, pivot, smooth, mat)` | Fan of `b_trig` calls |

### Tier 2 -- Surfaces

| Operation | Signature | Fallback |
|-----------|-----------|----------|
| `b_surface_polygon` | `b_surface_polygon(b, ps, mat)` | Ear-clipping triangulation into `b_trig` |
| `b_surface_polygon_with_holes` | `b_surface_polygon_with_holes(b, ps, qss, mat)` | Polygon subtraction then `b_surface_polygon` |
| `b_surface_circle` | `b_surface_circle(b, c, r, mat)` | `b_surface_regular_polygon` with 32 sides |
| `b_surface_arc` | `b_surface_arc(b, c, r, a, da, mat)` | `b_ngon` with sampled points |
| `b_surface_grid` | `b_surface_grid(b, ptss, cu, cv, su, sv, mat)` | Quad strips with optional interpolation |
| `b_surface_mesh` | `b_surface_mesh(b, vertices, faces, mat)` | Per-face `b_trig`/`b_quad`/`b_surface_polygon` |

### Tier 3 -- Solids

All solid operations have default decompositions.

| Operation | Signature | Fallback chain |
|-----------|-----------|---------------|
| `b_box` | `b_box(b, c, dx, dy, dz, mat)` | `b_cuboid` |
| `b_sphere` | `b_sphere(b, c, r, mat)` | Latitude/longitude strips of `b_ngon` + `b_quad_strip_closed` |
| `b_cylinder` | `b_cylinder(b, cb, r, h, mat)` | `b_generic_prism` with 32-gon |
| `b_cone` | `b_cone(b, cb, r, h, mat)` | `b_generic_pyramid` with 32-gon |
| `b_cone_frustum` | `b_cone_frustum(b, cb, rb, h, rt, mat)` | `b_generic_pyramid_frustum` with 32-gon |
| `b_torus` | `b_torus(b, c, ra, rb, mat)` | `b_surface_grid` with spherical sampling |
| `b_cuboid` | `b_cuboid(b, b0..b3, t0..t3, mat)` | `b_surface_polygon` + `b_quad_strip_closed` |

`b_solidify(b, refs)` is called at the end of each solid construction.  By
default it returns `refs` unchanged.  Override it if your backend needs to
convert a set of surface primitives into a proper solid.

### Tier 4 -- Extrusions, sweeps, revolves, lofts

| Operation | Default implementation |
|-----------|----------------------|
| `b_extruded_point` | `b_line` |
| `b_extruded_curve` | `b_quad_strip` or `b_cylinder` (for circular profiles along Z) |
| `b_extruded_surface` | `b_extruded_curve` for each boundary + `b_surface` for caps |
| `b_sweep` | `b_surface_grid` over rotation-minimizing frames |
| `b_loft` | `b_surface_grid` |
| `b_revolved_curve` | `b_surface_grid` over revolution frames |
| `b_revolved_surface` | `b_revolved_curve` for each boundary + caps |

### Tier 5 -- BIM

| Operation | Default implementation |
|-----------|----------------------|
| `b_slab` | `b_extruded_surface` |
| `b_roof` | `b_slab` |
| `b_beam` | `b_extruded_surface` with profile from family |
| `b_column` | `b_beam` oriented vertically |
| `b_wall` | Decomposed into `b_extruded_surface` for each wall segment |

### Tier 6 -- Rendering

| Operation | Description |
|-----------|-------------|
| `b_render_view(b, name)` | Render and save the current view. Default: calls `b_render_and_save_view`. |
| `b_render_and_save_view(b, path)` | Must be implemented for rendering support. |
| `b_render_initial_setup(b, kind)` | Called by `render_setup`. Default: no-op. |
| `b_render_final_setup(b, kind)` | Called just before rendering. Default: no-op. |
| `b_render_pathname(b, name)` | Returns output file path. Default: `render_default_pathname(name)`. |
| `b_set_view(b, camera, target, lens, aperture)` | Set the camera. Dispatches on `view_type`. |
| `b_get_view(b)` | Retrieve camera state. |

### Tier 7 -- Lighting

| Operation | Signature | Default Fallback |
|-----------|-----------|-----------------|
| `b_pointlight` | `b_pointlight(b, loc, energy, color)` | `missing_specialization` |
| `b_spotlight` | `b_spotlight(b, loc, dir, hotspot, falloff)` | `missing_specialization` |
| `b_arealight` | `b_arealight(b, loc, dir, size, energy, color)` | `b_pointlight(b, loc, energy, color)` |
| `b_ieslight` | `b_ieslight(b, file, loc, dir, alpha, beta, gamma)` | `b_spotlight(b, loc, dir, pi/4, pi/3)` |

`b_pointlight` and `b_spotlight` are required for rendering backends.
`b_arealight` and `b_ieslight` have built-in fallbacks that approximate them
using simpler light types, so they work on any backend that implements
point and spot lights.

## 8. The Fallback Chain

The fallback hierarchy follows this general pattern:

```
BIM elements (wall, slab, beam, ...)
  --> Solids (box, cylinder, extruded_surface, ...)
    --> Surfaces (surface_polygon, surface_grid, ...)
      --> Triangles (b_trig, b_quad, ...)
        --> Lines (b_line, b_polygon, ...)
```

Each level decomposes into the level below.  A backend only needs to implement
the lowest level it can support natively.  Implementing higher levels is
optional but produces better results (e.g., native cylinders instead of
32-sided prisms).

## 9. Material and Layer Operations

Backends must handle material references.  The key functions:

| Function | Description |
|----------|-------------|
| `b_get_material(b, spec)` | Resolve a material specification to a backend reference. Default: returns `void_ref(b)` for `nothing`, or `spec` itself otherwise. |
| `b_layer(b, name, active, color)` | Create a layer. Default: returns a `BasicLayer` struct. |
| `b_current_layer_ref(b)` | Get the current layer reference. Default: reads `b.current_layer`. |
| `b_current_layer_ref(b, layer)` | Set the current layer reference. Default: sets `b.current_layer`. |

For the `standard_material` proxy system, override `realize` for your backend to
interpret the material's PBR properties (base_color, metallic, roughness,
transmission, etc.) appropriately.

## 10. Minimal vs Full Backend Checklist

### Minimal backend (wireframe only)

- [ ] Define the backend struct with `refs::References{K,T}`
- [ ] Implement `void_ref(b)` returning raw `T`
- [ ] Implement `b_point(b, p, mat)`
- [ ] Implement `b_line(b, ps, mat)`
- [ ] Implement `b_delete_ref(b, r)` (or `b_delete_all_shape_refs`)

### Surface-capable backend

Everything above, plus:

- [ ] Implement `b_trig(b, p1, p2, p3)` (or `b_trig(b, p1, p2, p3, mat)`)
- [ ] Optionally implement `b_surface_polygon`, `b_surface_grid`,
      `b_surface_mesh` for better quality

### Solid-capable backend

Everything above, plus:

- [ ] Override `b_solidify(b, refs)` if the backend has real solids
- [ ] Optionally implement native `b_sphere`, `b_box`, `b_cylinder`, etc.
- [ ] Implement `b_subtract_ref`, `b_intersect_ref` for CSG support

### Rendering backend

Everything above, plus:

- [ ] Implement `b_render_and_save_view(b, path)`
- [ ] Implement `b_render_initial_setup(b, kind)` and
      `b_render_final_setup(b, kind)` for material/environment switching
- [ ] Implement `b_get_material(b, spec)` for `standard_material` support
- [ ] Set `view_type` to `FrontendView()` if storing camera locally
- [ ] Implement `b_pointlight` and `b_spotlight` for scene lighting
- [ ] Optionally implement `b_arealight` and `b_ieslight` for advanced lighting
      (default fallbacks approximate them as point/spot lights)

### BIM backend

Everything above, plus:

- [ ] Override `b_slab`, `b_wall`, `b_beam`, `b_column` with native
      implementations
- [ ] Implement family resolution functions for wall, slab, beam families
- [ ] Implement `b_level` for level management
