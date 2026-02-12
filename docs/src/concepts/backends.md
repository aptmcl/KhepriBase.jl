# Backends

Khepri separates *what* you design from *where* it is rendered. Every CAD application, rendering engine, or file format is represented by a **backend** -- an object that knows how to turn Khepri operations into concrete geometry in that target. The backend abstraction is the mechanism that makes Khepri scripts portable: the same code produces equivalent results in AutoCAD, Blender, Revit, TikZ, or any other supported target.

## The `Backend{K,T}` Type

All backends derive from the abstract type `Backend{K,T}`, parameterized by two types:

- **`K`** -- A key type that uniquely identifies the backend kind (e.g., `ACADKey`, `BLRKey`). This enables dispatch without ambiguity when multiple backends coexist.
- **`T`** -- The reference type the backend uses to track objects it has created (e.g., `Int64` for an integer handle, `String` for a name-based identifier).

```julia
abstract type Backend{K,T} end
```

When Khepri creates a sphere in AutoCAD, the AutoCAD backend returns an `Int64` handle that refers to the sphere inside the AutoCAD process. When it creates a sphere in a TikZ backend, the TikZ backend returns its own reference type. The user never sees these references directly; they are managed by the [Realize & Ref Protocol](../reference/realize_and_ref.md).

## Selecting a Backend

Activate a backend with the `backend` function:

```julia
backend(autocad)       # set AutoCAD as the current backend
```

Query the current state:

```julia
top_backend()          # the primary (first) active backend
current_backend()      # alias -- returns all active backends as a tuple
current_backends()     # all active backends (for multi-backend mode)
has_current_backend()  # true if at least one backend is active
```

If no backend has been set when an operation requires one, Khepri throws an `UndefinedBackendException`. The `top_backend()` function will wait briefly for a backend to become available (useful when a plugin connects asynchronously) before raising the exception.

## Backend Type Hierarchy

Backends are organized into a hierarchy that reflects their communication model:

```
Backend{K,T} (abstract)
|
+-- RemoteBackend{K,T} (abstract)
|   |
|   +-- SocketBackend{K,T}
|   |     TCP connections to C#/Python/C++ plugins running inside
|   |     CAD applications (AutoCAD, Revit, Rhino, Unity, Unreal).
|   |     Fields: name, port, connection, static_remote, remote,
|   |             transaction, refs
|   |
|   +-- WebSocketBackend{K,T}
|         Browser-based backends (Three.js, Xeokit, MeshCat).
|         Fields: name, websocket, buffer, static_remote, remote,
|                 transaction, refs, handlers
|
+-- LocalBackend{K,T} (abstract)
|   |
|   +-- IOBackend{K,T,E}
|         File-output backends (TikZ, POVRay, Radiance). Shapes are
|         collected in memory and serialized to a file or stream on
|         demand.
|         Fields: shapes, transaction, refs, current_layer, layers,
|                 date, place, render_env, ground_level,
|                 ground_material, view, io, extra
|
+-- LazyBackend{K,T} (abstract)
      Batch-processing backends that store shapes for later analysis
      (e.g., structural analysis with Frame3DD). Shapes are saved
      locally with save_shape! instead of being realized immediately.
```

Every concrete backend struct must include a `refs::References{K,T}` field. The `References` struct holds dictionaries that map Khepri proxies (shapes, materials, layers, annotations, families, levels) to their backend-specific references. See [Realize & Ref Protocol](../reference/realize_and_ref.md) for details.

## The Portability Guarantee

The central design principle of Khepri is that **the same script must work across all backends**. Only the import and the `backend(...)` call change:

```julia
using KhepriAutoCAD          # swap this line...
backend(autocad)              # ...and this line

# Everything below is backend-agnostic
delete_all_shapes()
sphere(xyz(0, 0, 0), 5)
box(xyz(10, 0, 0), 5, 5, 5)
set_view(xyz(50, 50, 30), xyz(0, 0, 0))
render_view("my_design")
```

Switching to Blender requires only two changes:

```julia
using KhepriBlender
backend(blender)
# ... identical design code ...
```

This guarantee is enforced by the operation dispatch and fallback system described below.

## Multi-Backend Mode

Khepri can send geometry to several backends simultaneously. This is useful for live previewing a design in a lightweight viewer while also building it in a full CAD application:

```julia
current_backend((autocad, blender))   # activate both
```

Operations defined with `@defcbs` (see below) are dispatched to **all** active backends. Operations defined with `@defcb` are dispatched only to the **primary** backend (`top_backend()`, which is the first element of the tuple).

You can also scope a block of code to a specific backend with the `@backend` and `@backends` macros:

```julia
@backend autocad begin
  # everything here runs only on AutoCAD
  sphere(xyz(0, 0, 0), 5)
end
```

## Operation Dispatch: The `b_*` Protocol

When you call a user-facing function like `sphere(center, radius)`, Khepri creates a shape proxy and, when that proxy is realized, calls the corresponding backend operation `b_sphere(backend, center, radius, material)`.

If a backend has not implemented `b_sphere`, KhepriBase provides a chain of **fallbacks** that emulate the operation using progressively simpler primitives:

1. **BIM level** -- High-level building elements (columns, walls, slabs).
2. **Solid level** -- 3D solids (spheres, cylinders, boxes, extrusions).
3. **Surface level** -- Surfaces and surface grids.
4. **Triangle level** -- Individual triangles and quad strips.
5. **Error** -- If no fallback exists, an `UnimplementedBackendOperationException` is raised.

For example, if a backend does not implement `b_circle`, the default fallback approximates it with `b_closed_spline` using 32 regularly-spaced vertices. If `b_closed_spline` is also missing, it falls through to `b_nurbs_curve`, then to `b_polygon` (a polyline), and so on.

Backend operations are declared in `Backend.jl` using the `@bdef` macro:

```julia
@bdef(b_sphere(center, radius, mat))
```

This generates a default method that raises `UnimplementedBackendOperationException`. Each backend package then provides its own specialization:

```julia
b_sphere(b::MyBackend, center, radius, mat) =
  # backend-specific implementation
```

## Frontend Macros

The Frontend module (`Frontend.jl`) provides macros that connect user-facing functions to backend operations.

### `@defcb` -- Single-Backend Dispatch

Defines a function that dispatches to `top_backend()` (the primary backend):

```julia
@defcb disable_update()
# Generates:
#   disable_update(; backend=top_backend()) = b_disable_update(backend)
```

### `@defcbs` -- All-Backends Dispatch

Defines a function that dispatches to **every** active backend:

```julia
@defcbs delete_all_shapes()
# Generates:
#   delete_all_shapes(; backends=current_backends()) =
#     for backend in backends
#       b_delete_all_shapes(backend)
#     end
```

### `@bcall` -- Explicit Backend Call

Calls a `b_*` operation on a specific backend:

```julia
@bcall(my_backend, sphere(center, radius, mat))
# Expands to: b_sphere(my_backend, center, radius, mat)
```

There is also `@bscall` for broadcasting to a collection of backends, and `@cbcall`/`@cbscall` for calling on the current backend(s) without passing them explicitly.

## Layers

Layers group shapes and control visibility. The layer API is backend-agnostic:

```julia
# Query and set the current layer
current_layer()                           # get the active layer
current_layer(my_layer)                   # set the active layer

# Layer operations
set_layer_active(layer, true)             # show/hide a layer
switch_to_layer(layer)                    # change the active layer
delete_all_shapes_in_layer(layer)         # clear a layer's shapes
```

Backends that manage layers locally (like `IOBackend`) store a `current_layer` field and a `layers` dictionary. Remote backends delegate layer management to the CAD application.

## Deletion Hierarchy

Khepri provides three levels of cleanup, each broader than the last:

| Function | What it removes |
|----------|----------------|
| `delete_all_annotations()` | Annotations only (labels, dimensions). Shapes and resources are preserved. |
| `delete_all_shapes()` | All shapes and annotations. Resources like materials, layers, and families are preserved. |
| `delete_all()` | Everything -- shapes, annotations, materials, layers, families, and levels. |

All three are defined with `@defcbs` and dispatch to every active backend. At the backend level, `b_delete_all_shape_refs` removes the backend-side references (e.g., deleting objects in AutoCAD), while the Julia-side dictionaries in `References` are emptied separately.

## Available Backends

Each backend follows a naming convention: a short type alias and a lowercase singleton instance.

| Package | Type Alias | Instance | Kind |
|---------|-----------|----------|------|
| KhepriAutoCAD | `ACAD` | `autocad` | SocketBackend |
| KhepriBlender | `BLR` | `blender` | SocketBackend |
| KhepriRevit | `RVT` | `revit` | SocketBackend |
| KhepriRhino | `RH` | `rhino` | SocketBackend |
| KhepriFreeCAD | `FRCAD` | `freecad` | SocketBackend |
| KhepriUnity | `Unity` | `unity` | SocketBackend |
| KhepriUnreal | `UE` | `unreal` | SocketBackend |
| KhepriThreejs | `THR` | `threejs` | WebSocketBackend |
| KhepriXeokit | `XKT` | `xeokit` | WebSocketBackend |
| KhepriMeshCat | -- | `meshcat` | WebSocketBackend |
| KhepriMakie | `MKE` | `makie` | Custom Backend |
| KhepriTikZ | -- | `tikz` | IOBackend |
| KhepriPOVRay | -- | `povray` | IOBackend |
| KhepriRadiance | -- | `radiance` | IOBackend |

For a complete matrix of which operations each backend supports, see the [Backend Operations](../reference/backend_operations.md) reference.

## Connection Lifecycle (Remote Backends)

Remote backends follow a connection protocol:

1. `connection(b)` is called when an operation needs to communicate with the external process.
2. If `b.connection` is `missing`, `start_connection(b)` establishes the link (TCP or WebSocket).
3. `before_connecting(b)` and `after_connecting(b)` hooks run around the connection setup.
4. If the connection fails, `retry_connecting(b)` waits and retries up to 10 times.
5. `reset_backend(b)` closes the connection and clears remote function opcodes.

Socket backends can also operate in **server mode**: Julia listens on a port (default 12345) and CAD plugins connect as clients. When a plugin connects, it sends its backend name, Khepri instantiates the appropriate backend, and calls `main(b)` to execute the user's design function.

## See Also

- [Parameters](parameters.md) -- The `Parameter{T}` system used by `current_backends`, `transaction`, and other dynamic state.
- [Realize & Ref Protocol](../reference/realize_and_ref.md) -- How shape proxies are lazily realized into backend references.
- [Implementing a Backend](../reference/implementing_backend.md) -- Step-by-step guide for backend developers.
