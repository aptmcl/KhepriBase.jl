# Installation and Setup

KhepriBase is the core library of the Khepri algorithmic design framework. You do not typically use KhepriBase alone -- instead, you install a **backend package** that re-exports everything from KhepriBase and adds the ability to produce geometry in a specific target application or format.

## Prerequisites

- Julia 1.12 or later (KhepriBase requirement)
- For socket backends (AutoCAD, Revit, etc.): the corresponding CAD application with the Khepri plugin installed
- For browser-based backends (Three.js, Xeokit): a modern web browser

## Installing a Backend

The simplest backend to start with is **KhepriThebes**, a lightweight local renderer that requires no external applications:

```julia
using Pkg
Pkg.add(url="https://github.com/aptmcl/KhepriBase.jl")
Pkg.add(url="https://github.com/aptmcl/KhepriThebes.jl")
```

Other backends follow the same pattern. For example, to use AutoCAD:

```julia
Pkg.add(url="https://github.com/aptmcl/KhepriBase.jl")
Pkg.add(url="https://github.com/aptmcl/KhepriAutoCAD.jl")
```

Available backend packages include:

| Package | Target | Type |
|---------|--------|------|
| KhepriThebes | Built-in Julia renderer | Local |
| KhepriAutoCAD | Autodesk AutoCAD | Socket |
| KhepriRevit | Autodesk Revit | Socket |
| KhepriBlender | Blender | Socket |
| KhepriRhino | McNeel Rhinoceros | Socket |
| KhepriUnity | Unity | Socket |
| KhepriUnreal | Unreal Engine | Socket |
| KhepriMakie | Makie.jl visualization | Local |
| KhepriThreejs | Three.js (browser) | WebSocket |
| KhepriXeokit | Xeokit (browser) | WebSocket |
| KhepriMeshCat | MeshCat (browser) | WebSocket |
| KhepriTikZ | LaTeX TikZ output | File output |
| KhepriPOVRay | POV-Ray rendering | File output |
| KhepriRadiance | Radiance lighting analysis | File output |

## Hello Sphere

The minimum code to create geometry in Khepri:

```julia
using KhepriThebes

backend(thebes)
delete_all_shapes()
sphere(xyz(0, 0, 0), 5)
```

This creates a sphere of radius 5 centered at the origin. The four lines are all you need: load the backend, activate it, clear the scene, and create a shape.

## Switching Backends

One of Khepri's core guarantees is that **the same design code works across all backends**. Only two lines change -- the `using` declaration and the `backend()` call:

```julia
# Using Thebes (local renderer)
using KhepriThebes
backend(thebes)

# Using AutoCAD
using KhepriAutoCAD
backend(autocad)

# Using Blender
using KhepriBlender
backend(blender)

# Using Revit
using KhepriRevit
backend(revit)

# Using Rhino
using KhepriRhino
backend(rhino)
```

Everything else -- shape creation, materials, views, rendering -- remains identical regardless of the backend.

## Backend Types

Khepri backends fall into four categories, each with a different communication model:

### Socket Backends

Socket backends communicate with CAD application plugins over TCP. Julia acts as the server and the CAD plugin connects as a client. Each plugin has a fixed port assignment:

| Backend | Port |
|---------|------|
| AutoCAD | 11000 |
| Revit | 11001 |
| Unity | 11002 |
| Rhino | dynamic |

To use a socket backend, start the CAD application with the Khepri plugin loaded, then activate the backend in Julia. The plugin connects automatically.

### WebSocket Backends

WebSocket backends (Three.js, Xeokit, MeshCat) communicate with browser-based viewers. Activating the backend starts a local server and opens a browser window for real-time 3D visualization. No CAD application is required.

### File-Output Backends (IO)

IO backends (TikZ, POV-Ray, Radiance) collect geometry in memory and serialize it to a file on demand. They produce source files that are then processed by their respective tools -- LaTeX for TikZ, POV-Ray for raytracing, Radiance for lighting simulation.

### Local Julia Backends

Local backends (Thebes, Makie) run entirely within the Julia process. They are the simplest to set up since they have no external dependencies beyond their Julia packages. Thebes is the recommended starting point for learning Khepri.

## Verifying the Installation

After installing a backend, verify that everything works:

```julia
using KhepriThebes

backend(thebes)
delete_all_shapes()

# Create a few shapes
sphere(xyz(0, 0, 0), 3)
box(xyz(8, 0, 0), 4, 4, 4)
cylinder(xyz(-8, 0, 0), 2, 6)

# Set a viewpoint and render
set_view(xyz(20, 20, 15), xyz(0, 0, 0))
render_view("test_scene")
```

If the render completes without errors, the installation is working correctly.

## Next Steps

- [Coordinates](coordinates.md) -- Learn about Khepri's coordinate system: Cartesian, cylindrical, and spherical locations, vectors, and coordinate frames.
- [Shapes](../concepts/shapes.md) -- The full catalog of geometric primitives, surfaces, solids, and CSG operations.
- [Backends](../concepts/backends.md) -- Detailed coverage of the backend abstraction, multi-backend mode, operation dispatch, and fallback chains.
