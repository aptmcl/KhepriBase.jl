# Parameters

KhepriBase uses a parameter system to manage mutable configuration values -- coordinate systems, render dimensions, BIM defaults, and more. Parameters behave like callable objects: `p()` reads the current value and `p(newvalue)` sets it. The system offers five parameter types, each with different scoping and initialization semantics.

## Parameter Types

### `Parameter{T}` -- Task-Local Mutable Parameter

The most common type. Each Julia task gets its own value via `task_local_storage`, preventing cross-task contamination. The `value` field serves as the global default, returned when the current task has not set a task-local override.

```julia
mutable struct Parameter{T}
  value::T
end
(p::Parameter{T})() where T = get(task_local_storage(), p, p.value)::T
(p::Parameter{T})(newvalue::T) where T = task_local_storage(p, newvalue)::T
```

```julia
const current_cs = Parameter(world_cs)
current_cs()            # read: returns task-local value, or world_cs if unset
current_cs(my_cs)       # write: sets task-local override for this task only
```

### `GlobalParameter{T}` -- Shared Mutable Parameter

A single mutable value shared across all tasks. Use this for infrastructure state that is set once (server host/port) or that must be visible across tasks.

```julia
mutable struct GlobalParameter{T}
  value::T
end
(p::GlobalParameter{T})() where T = p.value::T
(p::GlobalParameter{T})(newvalue::T) where T = p.value = newvalue::T
```

### `OptionParameter{T}` -- Task-Local, Possibly Missing

Like `Parameter{T}` but the value can be `missing`. Reading a missing `OptionParameter` throws an error, ensuring that required configuration is explicitly set before use. Uses task-local storage like `Parameter`.

```julia
mutable struct OptionParameter{T}
  value::Union{Missing,T}
end
(p::OptionParameter{T})() where T =
  let v = get(task_local_storage(), p, p.value)
    ismissing(v) ? error("Parameter was not initialized with value of type $T") : v::T
  end
```

### `LazyParameter{T,F}` -- Global, Lazily Initialized

Initialized only on first access by calling the provided initializer function. The result is cached. `reset(p)` clears the cached value so the next read re-runs the initializer. Global (not task-local), typically used for singletons like server instances.

```julia
mutable struct LazyParameter{T,F<:Function}
  initializer::F
  value::Union{T,Nothing}
end
(p::LazyParameter{T,F})() where {T,F} =
  isnothing(p.value) ? (p.value = p.initializer()::T) : p.value
Base.reset(p::LazyParameter{T,F}) where {T,F} = p.value = nothing
```

```julia
const khepri_websocket_server = LazyParameter(run_khepri_websocket_server)
khepri_websocket_server()          # starts server on first call, returns cached instance after
reset(khepri_websocket_server)     # forces re-initialization on next access
```

### `ThreadLocalParameter` -- Backwards Compatibility Alias

An alias for `Parameter`. Older code may reference `ThreadLocalParameter`; new code should use `Parameter` directly.

## The `with` Function

`with` temporarily overrides one or more parameters for the duration of a block, restoring the original values when the block exits (even on error).

```julia
with(current_cs, my_local_cs) do
  # inside this block, current_cs() returns my_local_cs
  sphere(xyz(0, 0, 0), 1)
end
# current_cs() is restored here
```

Multiple parameters -- pass alternating parameter/value pairs:

```julia
with(render_width, 1920, render_height, 1080) do
  set_view(xyz(30, 30, 20), xyz(0, 0, 0))
  render_view("high_res_output")
end
```

The multi-parameter form nests automatically: `with(f, p1, v1, p2, v2)` is equivalent to `with(p1, v1) do; with(f, p2, v2); end`.

## Built-in Parameters

KhepriBase defines many parameters across its modules. Below are representative examples.

### Coordinates and Geometry

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `current_cs` | `Parameter` | `world_cs` | Active coordinate system |
| `epsilon` | `Parameter` | `1e-8` | Geometric comparison tolerance |
| `path_tolerance` | `Parameter` | `1e-10` | Path operations tolerance |

### Rendering and Camera

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `render_width` | `Parameter` | `1024` | Output width in pixels |
| `render_height` | `Parameter` | `768` | Output height in pixels |
| `render_dir` | `Parameter` | `homedir()` | Base directory for output |
| `render_kind` | `Parameter` | `:realistic` | Style (`:realistic`, `:white`, `:black`) |

### Materials

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_material` | `Parameter` | `material_basic` | Default for 3D shapes |
| `default_curve_material` | `Parameter` | `material_curve` | Default for curves |
| `default_surface_material` | `Parameter` | `material_surface` | Default for surfaces |

See [Levels & Families](levels_and_families.md) for material details and `material()`.

### BIM

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_level` | `OptionParameter` | `level()` | Current default BIM level |
| `default_level_to_level_height` | `Parameter` | `3` | Floor-to-floor height (meters) |

Each family defined via `@deffamily` also generates a `default_*_family` parameter (e.g., `default_wall_family`, `default_slab_family`). See [Levels & Families](levels_and_families.md).

### Backend Infrastructure

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `current_backends` | `Parameter` | `()` | Active backend(s) |
| `default_khepri_socket_server_host` | `GlobalParameter` | `ip"127.0.0.1"` | Socket server bind address |
| `default_khepri_socket_server_port` | `GlobalParameter` | `12345` | Socket server port |
| `khepri_websocket_server` | `LazyParameter` | (lazy) | WebSocket server singleton |

See [Backends](backends.md) for backend configuration details.
