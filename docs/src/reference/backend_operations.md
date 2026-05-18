```@meta
CurrentModule = KhepriBase
```

# Backend Operations Matrix

Every Khepri backend implements a subset of the `b_*` operations defined in KhepriBase's `Backend.jl`. These operations form the backend portability contract: when a user writes `slab(...)`, KhepriBase dispatches to `b_slab(backend, ...)`, and each backend provides its own implementation.

Operations are organized into tiers by complexity. A backend only needs to implement the tiers relevant to its purpose — a ray tracer (POVRay) skips selection operations, while a BIM tool (Revit) implements BIM operations that a game engine (Unity) does not.

This document provides a comprehensive overview of which `b_` operations each Khepri backend implements.

## Operations Inventory

KhepriBase defines **136+ operations** organized into functional tiers.

## Tier Definitions

### Tier 0: Core & Reference Management (6 operations)
Essential operations for backend functionality.

| Operation | Description |
|-----------|-------------|
| `void_ref` | Returns a null/void reference |
| `b_delete_all_shape_refs` | Delete all shape references |
| `b_delete_refs` | Delete specific references |
| `b_delete_ref` | Delete a single reference |
| `b_all_shape_refs` | Get all shape references |
| `b_delete_all_annotations` | Delete all annotations |

### Tier 1: Basic Curves
2D and 3D curve primitives.

| Operation | Description |
|-----------|-------------|
| `b_point` | Create a point |
| `b_line` | Create a line from points |
| `b_polygon` | Create a closed polygon |
| `b_regular_polygon` | Create a regular n-gon |
| `b_interpolating_spline_curve` | Create an interpolating spline while preserving fit-point semantics |
| `b_bezier_curve` | Create a Bezier curve from Bezier segments |
| `b_bspline_curve` | Create a non-rational B-spline curve from control data |
| `b_nurbs_curve` | Create a NURBS curve |
| `b_polycurve` | Create or join a mixed exact segment curve |
| `b_spline` | Create a spline |
| `b_closed_spline` | Create a closed spline |
| `b_circle` | Create a circle |
| `b_arc` | Create an arc |
| `b_ellipse` | Create an ellipse |
| `b_rectangle` | Create a rectangle |
| `b_trig` | Create a triangle |
| `b_quad` | Create a quadrilateral |
| `b_ngon` | Create an n-sided polygon |
| `b_quad_strip` | Create a quad strip |
| `b_quad_strip_closed` | Create a closed quad strip |
| `b_closed_line` | Legacy alias for b_polygon |

### Tier 2: Surfaces
Surface primitives.

| Operation | Description |
|-----------|-------------|
| `b_strip` | Create a strip surface |
| `b_surface_polygon` | Create a polygonal surface |
| `b_surface_polygon_with_holes` | Surface with holes |
| `b_surface_rectangle` | Rectangular surface |
| `b_surface_regular_polygon` | Regular polygon surface |
| `b_surface_circle` | Circular surface |
| `b_surface_ring` | Annular ring surface |
| `b_surface_arc` | Arc surface |
| `b_surface_ellipse` | Elliptical surface |
| `b_surface_closed_spline` | Closed spline surface |
| `b_surface_grid` | Grid surface |
| `b_smooth_surface_grid` | Smooth grid surface |
| `b_surface_mesh` | Mesh surface |
| `b_sampled_surface` | Tessellated fallback for any surface geometry |
| `b_bezier_surface` | Create a Bezier tensor-product surface |
| `b_bspline_surface` | Create a non-rational B-spline tensor-product surface |
| `b_nurbs_surface` | Create a rational B-spline / NURBS surface |
| `b_trimmed_surface` | Create a surface with trim boundaries |

### Tier 3: Basic Solids (15 operations)
3D solid primitives.

| Operation | Description |
|-----------|-------------|
| `b_generic_pyramid_frustum` | Generic pyramid frustum |
| `b_generic_pyramid` | Generic pyramid |
| `b_generic_prism` | Generic prism |
| `b_generic_prism_with_holes` | Prism with holes |
| `b_pyramid_frustum` | Pyramid frustum |
| `b_pyramid` | Pyramid |
| `b_prism` | Prism |
| `b_cuboid` | Cuboid |
| `b_box` | Box |
| `b_sphere` | Sphere |
| `b_cone` | Cone |
| `b_cone_frustum` | Cone frustum |
| `b_cylinder` | Cylinder |
| `b_torus` | Torus |
| `b_solidify` | Convert to solid |

### Tier 4: Advanced Modeling (16 operations)
Complex geometry operations.

| Operation | Description |
|-----------|-------------|
| `b_surface` | Create surface from frontier |
| `b_extruded_point` | Extrude a point |
| `b_extruded_curve` | Extrude a curve |
| `b_extruded_surface` | Extrude a surface |
| `b_extrusion` | General extrusion |
| `b_swept_curve` | Sweep a curve |
| `b_swept_surface` | Sweep a surface |
| `b_sweep` | General sweep |
| `b_revolved_point` | Revolve a point |
| `b_revolved_curve` | Revolve a curve |
| `b_revolved_surface` | Revolve a surface |
| `b_loft` | General loft |
| `b_loft_points` | Loft through points |
| `b_loft_curve_point` | Loft curve to point |
| `b_loft_curves` | Loft through curves |
| `b_loft_surfaces` | Loft through surfaces |

### Tier 5: Boolean Operations (13 operations)
CSG operations.

| Operation | Description |
|-----------|-------------|
| `b_unite_ref` | Unite two references |
| `b_unite_refs` | Unite multiple references |
| `b_subtract_ref` | Subtract two references |
| `b_intersect_ref` | Intersect two references |
| `b_subtracted_surfaces` | Subtract surfaces |
| `b_subtracted_solids` | Subtract solids |
| `b_subtracted` | General subtraction |
| `b_intersected_surfaces` | Intersect surfaces |
| `b_intersected_solids` | Intersect solids |
| `b_intersected` | General intersection |
| `b_united_surfaces` | Unite surfaces |
| `b_united_solids` | Unite solids |
| `b_united` | General union |
| `b_slice` | Slice geometry |
| `b_slice_ref` | Slice reference |

### Tier 6: BIM Operations (20 operations)
Building Information Modeling.

| Operation | Description |
|-----------|-------------|
| `b_slab` | Create a slab/floor |
| `b_roof` | Create a roof |
| `b_ceiling` | Create a ceiling |
| `b_beam` | Create a beam |
| `b_column` | Create a column |
| `b_free_column` | Create a free-standing column |
| `b_wall` | Create a wall |
| `b_curtain_wall` | Create a curtain wall |
| `b_curtain_wall_element` | Curtain wall element |
| `b_railing` | Create a railing |
| `b_ramp` | Create a ramp |
| `b_stair` | Create a straight stair |
| `b_spiral_stair` | Create a spiral stair |
| `b_stair_landing` | Create a stair landing |
| `b_toilet` | Place a toilet |
| `b_sink` | Place a sink |
| `b_closet` | Place a closet |
| `b_truss_node` | Create truss node |
| `b_truss_node_support` | Create truss support |
| `b_truss_bar` | Create truss bar |

### Tier 7: Annotations & Graphics (9 operations)
Drawing annotations.

| Operation | Description |
|-----------|-------------|
| `b_dimension` | Create dimension |
| `b_ext_line` | Extension line |
| `b_dim_line` | Dimension line |
| `b_text` | Create text |
| `b_text_size` | Get/set text size |
| `b_arc_dimension` | Arc dimension |
| `b_stroke` | Stroke path |
| `b_fill` | Fill path |
| `b_realize_path` | Realize path |

### Tier 8: Selection Operations (13 operations)
Interactive selection.

| Operation | Description |
|-----------|-------------|
| `b_select_position` | Select a position |
| `b_select_positions` | Select multiple positions |
| `b_select_point` | Select a point |
| `b_select_points` | Select multiple points |
| `b_select_curve` | Select a curve |
| `b_select_curves` | Select multiple curves |
| `b_select_surface` | Select a surface |
| `b_select_surfaces` | Select multiple surfaces |
| `b_select_solid` | Select a solid |
| `b_select_solids` | Select multiple solids |
| `b_select_shape` | Select a shape |
| `b_select_shapes` | Select multiple shapes |
| `b_all_shapes_in_layer` | Get shapes in layer |

### Tier 9: Rendering & Environment (13 operations)
Rendering and view control.

| Operation | Description |
|-----------|-------------|
| `b_render_pathname` | Get render output path |
| `b_render_view` | Render current view |
| `b_render_initial_setup` | Initial render setup |
| `b_render_final_setup` | Final render setup |
| `b_render_and_save_view` | Render and save |
| `b_setup_render` | Setup rendering |
| `b_set_environment` | Set environment |
| `b_realistic_sky` | Set realistic sky |
| `b_set_ground` | Set ground plane |
| `b_set_view` | Set camera view |
| `b_set_view_top` | Set top view |
| `b_get_view` | Get current view |
| `b_zoom_extents` | Zoom to extents |

### Tier 10: Materials & Layers (18 operations)
Material and layer management.

| Operation | Description |
|-----------|-------------|
| `b_get_material` | Get material |
| `b_material` | Create material (cascading tiers: full PBR → standard → basic → color) |
| `b_plastic_material` | Create plastic material |
| `b_metal_material` | Create metal material |
| `b_glass_material` | Create glass material |
| `b_mirror_material` | Create mirror material |
| `b_layer` | Create/get layer |
| `b_current_layer_ref` | Get/set current layer |
| `b_delete_all_shapes_in_layer` | Delete layer shapes |
| `b_highlight_ref` | Highlight reference |
| `b_highlight_refs` | Highlight references |
| `b_highlight_shapes` | Highlight shapes |
| `b_highlight_shape` | Highlight shape |
| `b_unhighlight_ref` | Unhighlight reference |
| `b_unhighlight_refs` | Unhighlight references |
| `b_unhighlight_shapes` | Unhighlight shapes |
| `b_unhighlight_shape` | Unhighlight shape |
| `b_unhighlight_all_refs` | Unhighlight all |

### Tier 11: Decorative & Utilities (8 operations)
Furniture and illustrations.

| Operation | Description |
|-----------|-------------|
| `b_table` | Create table |
| `b_chair` | Create chair |
| `b_table_and_chairs` | Create table with chairs |
| `b_mesh_obj_fmt` | Load OBJ/MTL mesh with transform |
| `b_labels` | Create labels |
| `b_radii_illustration` | Radii illustration |
| `b_vectors_illustration` | Vectors illustration |
| `b_angles_illustration` | Angles illustration |
| `b_arcs_illustration` | Arcs illustration |

### Tier 12: Lighting (4 operations)
Light sources. Operations with default fallbacks degrade gracefully on backends without native support.

| Operation | Description | Default Fallback |
|-----------|-------------|-----------------|
| `b_pointlight` | Point light | `missing_specialization` (must implement) |
| `b_spotlight` | Spot light | `missing_specialization` (must implement) |
| `b_arealight` | Area light | `b_pointlight` (same location/energy/color) |
| `b_ieslight` | IES photometric light | `b_spotlight` (45°/60° angles) |

---

## Exact Curve and Surface Hooks

The spline and surface hierarchy now separates the geometric intent from the
fallback representation sent to a backend. If a backend does not implement one
of these hooks, KhepriBase keeps the model portable by sampling curves to
polylines and surfaces to meshes. Interpolating splines keep their legacy
fit-point semantics through `b_spline`/`b_closed_spline`, and planar trimmed
surfaces still use polygon-with-holes fallbacks.

Backends declare these direct mappings with `curve_geometry_capabilities` and
`surface_geometry_capabilities`; user code can query the individual
`supports_exact_*` helpers when exact native geometry matters.

| Khepri geometry | Direct hook | Best native backend mapping | Implemented direct mappings |
|-----------------|-------------|-----------------------------|-----------------------------|
| `InterpolatingSplinePath` | `b_interpolating_spline_curve` | Fit/interpolated spline curve | Rhino and AutoCAD through existing interpolating spline operations |
| `BezierPath` | `b_bezier_curve` | Bezier curve or Bezier-derived spline | Rhino, AutoCAD, FreeCAD |
| `BSplinePath{*,false}` | `b_bspline_curve` | Non-rational B-spline with explicit degree and knots | Rhino, AutoCAD, FreeCAD |
| `NurbsPath` | `b_nurbs_curve` | Rational B-spline / NURBS curve with explicit degree, knots, and weights | Rhino, AutoCAD, FreeCAD |
| `BezierSurface` | `b_bezier_surface` | Tensor-product Bezier surface | Rhino, AutoCAD, FreeCAD |
| `BSplineSurface{*,*,false}` | `b_bspline_surface` | Non-rational tensor-product B-spline surface | Rhino, AutoCAD, FreeCAD |
| `NurbsSurface` | `b_nurbs_surface` | Rational tensor-product B-spline / NURBS surface | Rhino, AutoCAD, FreeCAD |
| `TrimmedSurface` | `b_trimmed_surface` | Native trimmed face or trimmed surface | Planar trims use polygon-with-holes operations; non-planar trims tessellate from UV-space boundaries unless a backend provides a native override |

## Geometry Computation Hooks

The result-valued geometry operations in KhepriBase have backend hooks separate
from the shape-realization hooks. They are used when the caller needs explicit
computed geometry, not a lazy proxy. A backend should advertise support with
`supports_geometry_operation(backend, operation, args...)` and return KhepriBase
result containers such as `IntersectionSet`, `ProjectionResult`,
`ClosestPointsResult`, `Region`, `MultiRegion`, or `GeometryCollection`.

| Hook | Public operation | Expected result |
|------|------------------|-----------------|
| `b_intersections` | `intersections(a, b)` | `IntersectionSet` with point, curve, or region elements |
| `b_section` | `section(a, b)` | `IntersectionSet` containing curve intersections |
| `b_boolean_geometry` | `boolean(op, a, b)` | explicit `Region`, `MultiRegion`, `GeometryCollection`, or backend-specific geometry value |
| `b_project_geometry` | `project(a, target)` | `ProjectionResult` |
| `b_split_geometry` | `split(a, cutters...)` | geometry pieces, usually `PathSet`, `MultiRegion`, or `GeometryCollection` |
| `b_trim_geometry` | `trim(a, cutters...)` | trimmed geometry |
| `b_closest_points` | `closest_points(a, b)` | `ClosestPointsResult` |
| `b_classify_geometry` | `classify_geometry(container, item)` | classification symbol such as `:inside`, `:boundary`, `:outside`, `:on`, `:positive`, or `:negative` |

These hooks are intended to map directly to native kernel capabilities such as
RhinoCommon intersection routines, AutoCAD/AcGe intersectors, or Open CASCADE
section and projection algorithms. Backends without a native operation can
decline support and let KhepriBase use its local analytic implementation when
one exists.

Current direct mappings:

| Backend | Native kernel/API | Advertised geometry operations |
|---------|-------------------|--------------------------------|
| Rhino | RhinoCommon `Intersection.CurveCurve`, `CurveBrep`, and `BrepBrep` | `intersections` and `section` for path/path, path/finite-surface, and finite-surface/finite-surface operands; straight sampled section curves collapse to `LinePath` |
| AutoCAD | Entity `IntersectWith` | point-valued `intersections` for path/path and path/finite-surface operands |
| FreeCAD | Open CASCADE section results through `Shape.section` | `intersections` and `section` for path/path, path/finite-surface, and finite-surface/finite-surface operands; straight sampled section curves collapse to `LinePath` |

## Backend Geometry Mapping Report

`backend_geometry_mapping(backend)` returns a compact machine-readable summary
of the backend's exact creation capabilities, import path, and advertised
result-valued operations for representative operands. Use it to keep backend
coverage tables honest:

```julia
report = backend_geometry_mapping(top_backend())

report.curve_capabilities.nurbs
report.surface_capabilities.trimmed
report.import_mapping.all_shapes
report.operations.section_surface_surface
```

The import fields distinguish local shape storage from remote-reference
storage. A remote backend should provide both `b_existing_shape_refs` and
`b_create_shape_from_ref_value` so `all_shapes`, layer queries, and selected
backend objects can be mapped back into Khepri proxies. Rhino, AutoCAD, and
FreeCAD now use this pathway for basic curve imports; FreeCAD currently imports
lines, polygons, circles, arcs, sampled open curves, sampled closed curves, and
opaque surface placeholders.

## Backend Coverage Summary

| Backend | Type | Curves | Surfaces | Solids | Boolean | BIM | Selection | Rendering | Materials | Lighting | OBJ/MTL |
|---------|------|--------|----------|--------|---------|-----|-----------|-----------|-----------|----------|---------|
| **Rhino** | Socket/C# | Full | Full | Full | Partial | No | Full | Yes | Yes | Point, Spot, IES, Area* | Yes |
| **AutoCAD** | Socket/C# | Full | Full | Full | Full | Yes | Full | Yes | Yes | Point, Spot, IES, Area* | No |
| **Blender** | Socket/Python | Good | Good | Good | Good | No | No | Yes | Yes | Point, Spot, IES, Area | Yes |
| **Mitsuba** | IO/file | No | Good | Good | No | No | No | Yes | Yes | Point, Spot, IES, Area | No |
| **GL** | Local | Full | Full | Full | No-op | No | No | Yes | Yes | Point, Spot, IES*, Area* | No |
| **Thebes** | Local | Full | Full | Full | No-op | No | No | Yes | Yes | (fallbacks) | No |
| **Makie** | Local | Good | Good | Good | No | No | No | Yes | Yes | (fallbacks) | No |
| **MeshCat** | WebSocket | Good | Good | Good | No | No | No | Partial | Yes | (fallbacks) | No |
| **Unity** | Socket/C# | Good | Good | Good | Yes | No | No | Yes | Yes | Point, IES*, Area* | No |
| **Revit** | Socket/C# | No | Partial | Partial | Yes | Full | No | Yes | Read-only | (fallbacks) | No |
| **Three.js** | WebSocket | Good | Good | Good | No | No | No | Partial | Yes | (fallbacks) | Yes |
| **Xeokit** | WebSocket | Good | Good | Good | No | No | No | Partial | Yes | (fallbacks) | No |
| **TikZ** | IO/file | Good | Good | No | No | No | No | No | No | None | No |
| **POVRay** | IO/file | No | Partial | Good | No | No | No | Yes | Yes | Point, IES*, Area* | No |
| **Radiance** | IO/file | No | Partial | Partial | No | No | No | Yes | Yes | Point, Spot, IES, Area | No |
| **FreeCAD** | Socket/Python | Good | Good | Good | Good | No | No | No | Yes | (fallbacks) | No |
| **Unreal** | Socket/C++ | Good | Good | Good | No | No | No | Yes | Yes | Point, Spot, IES*, Area* | No |
| **3dsMax** | Socket | Good | Good | Good | No | No | No | No | Yes | (fallbacks) | No |

*\* = approximated via simpler light type (IES→spotlight, Area→pointlight)*

---

## Backend Profiles

### Rhino (RH) - General Purpose CAD
**Best for:** General 3D modeling, interactive selection, curve/surface work
- Socket backend (C#), right-handed Z-up
- Most comprehensive implementation with full selection API

### AutoCAD (ACAD) - CAD/Drafting
**Best for:** 2D/3D CAD, technical drawings, BIM, interactive workflows
- Socket backend (C#), right-handed Z-up
- Strong curve/solid support, full selection, BIM operations

### Blender (BLR) - 3D Graphics
**Best for:** Rendering, visualization, material-heavy scenes
- Socket backend (Python)
- Full lighting, materials, and rendering pipeline

### GL - OpenGL Visualization
**Best for:** Local 3D visualization, rapid prototyping
- Local backend (Julia), right-handed Z-up
- Immediate-mode rendering, good for quick preview

### Thebes - Software 3D Renderer
**Best for:** Offline rendering without external dependencies
- Local backend (Julia), right-handed Z-up
- Pure Julia renderer with material support

### Makie - Julia Visualization
**Best for:** Scientific visualization, Julia ecosystem integration
- Local backend (Julia)
- Leverages Makie.jl for interactive plotting

### MeshCat - Web Visualization
**Best for:** Browser-based 3D preview
- WebSocket backend
- Lightweight web viewer via MeshCat.jl

### Unity - Game Engine
**Best for:** Real-time visualization, game development
- Socket backend (C#), **left-handed Y-up** (Y-Z axis swap)
- Real-time rendering and game integration

### Revit (RVT) - BIM
**Best for:** Building design, BIM workflows
- Socket backend (C#), right-handed Z-up
- Best BIM support (walls, slabs, roofs, columns, beams)

### Three.js (THR) - Web Graphics
**Best for:** Web-based 3D visualization
- WebSocket backend (TypeScript)
- Browser-based, lightweight

### Xeokit (XEO) - Web BIM Viewer
**Best for:** Web-based BIM visualization
- WebSocket backend
- IFC/BIM model viewing in browser

### TikZ - 2D/3D Vector Graphics
**Best for:** Technical documentation, LaTeX integration
- IO backend (file output)
- Excellent annotation support, 2D/3D graphics to LaTeX

### POVRay - Ray Tracing
**Best for:** High-quality ray-traced renders
- IO backend (file output)
- Scene description for POV-Ray renderer

### Radiance - Lighting Simulation
**Best for:** Lighting analysis, daylighting studies
- IO backend (file output)
- Accurate lighting simulation, specialized geometry

### FreeCAD - Open Source CAD
**Best for:** Open-source 3D modeling
- Socket backend (Python)
- Parametric modeling with boolean operations

### Unreal - Game Engine
**Best for:** High-fidelity real-time visualization
- Socket backend (C++)
- Unreal Engine integration

---

## Choosing a Backend

| Use Case | Recommended Backend |
|----------|---------------------|
| General 3D modeling | Rhino, AutoCAD |
| Architectural BIM | Revit |
| High-quality renders | Blender, POVRay |
| Web visualization | Three.js, Xeokit, MeshCat |
| Real-time / Game | Unity, Unreal |
| Local preview (no deps) | GL, Thebes |
| Julia ecosystem | Makie |
| Documentation figures | TikZ |
| Lighting analysis | Radiance |
| Interactive selection | Rhino, AutoCAD |
| Open-source CAD | FreeCAD |

---

## Notes

### Axis Conventions
- Most backends use right-handed Z-up coordinate system
- **Unity** swaps Y and Z axes to match its Y-up convention

### Reference Types
Different backends use different reference types (`T` in `Backend{K,T}`):
- `Int64`: AutoCAD, GL, Revit, Robot, Frame3DD
- `Int32`: Blender, Unity, Three.js, FreeCAD, 3dsMax, Unreal
- `UInt128` (GUID): Rhino
- `String`: MeshCat, Xeokit
- `Nothing`: Makie, Thebes, TikZ (local backends use `nothing` as void)
- `Int`: POVRay, Radiance

The `void_ref` function returns a raw value of type `T` (not wrapped in `NativeRef`).
The `ensure_ref` function in KhepriBase wraps raw `T` values into `NativeRef{K,T}`.

### Encoding Protocols
- `CS`: C# backends (AutoCAD, Revit, Rhino, Unity)
- `PY`: Python backends (Blender, FreeCAD)
- `CPP`: C++ backends (Unreal)
- `TS`/`WS`: TypeScript/WebSocket backends (Three.js, Xeokit, MeshCat)
- None: Local backends (GL, Makie, Thebes, TikZ, POVRay, Radiance)
