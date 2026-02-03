# Khepri Backend Operations Matrix

This document provides a comprehensive overview of which `b_` operations each Khepri backend implements.

## Operations Inventory

KhepriBase defines **130+ operations** organized into functional tiers.

## Tier Definitions

### Tier 0: Core & Reference Management (6 operations)
Essential operations for backend functionality.

| Operation | Description |
|-----------|-------------|
| `void_ref` | Returns a null/void reference |
| `b_delete_all_shape_refs` | Delete all shape references |
| `b_delete_refs` | Delete specific references |
| `b_delete_ref` | Delete a single reference |
| `b_all_shapes` | Get all shapes |
| `b_delete_all_annotations` | Delete all annotations |

### Tier 1: Basic Curves (17 operations)
2D and 3D curve primitives.

| Operation | Description |
|-----------|-------------|
| `b_point` | Create a point |
| `b_line` | Create a line from points |
| `b_polygon` | Create a closed polygon |
| `b_regular_polygon` | Create a regular n-gon |
| `b_nurbs_curve` | Create a NURBS curve |
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

### Tier 2: Surfaces (12 operations)
Surface primitives.

| Operation | Description |
|-----------|-------------|
| `b_strip` | Create a strip surface |
| `b_surface_polygon` | Create a polygonal surface |
| `b_surface_polygon_with_holes` | Surface with holes |
| `b_surface_rectangle` | Rectangular surface |
| `b_surface_regular_polygon` | Regular polygon surface |
| `b_surface_circle` | Circular surface |
| `b_surface_arc` | Arc surface |
| `b_surface_ellipse` | Elliptical surface |
| `b_surface_closed_spline` | Closed spline surface |
| `b_surface_grid` | Grid surface |
| `b_smooth_surface_grid` | Smooth grid surface |
| `b_surface_mesh` | Mesh surface |

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
| `b_unite_ref` | Unite references |
| `b_unite_refs` | Unite multiple references |
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

### Tier 6: BIM Operations (14 operations)
Building Information Modeling.

| Operation | Description |
|-----------|-------------|
| `b_slab` | Create a slab/floor |
| `b_roof` | Create a roof |
| `b_beam` | Create a beam |
| `b_column` | Create a column |
| `b_free_column` | Create a free-standing column |
| `b_wall` | Create a wall |
| `b_curtain_wall` | Create a curtain wall |
| `b_curtain_wall_element` | Curtain wall element |
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
| `b_new_material` | Create material |
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
| `b_labels` | Create labels |
| `b_radii_illustration` | Radii illustration |
| `b_vectors_illustration` | Vectors illustration |
| `b_angles_illustration` | Angles illustration |
| `b_arcs_illustration` | Arcs illustration |

### Tier 12: Lighting (4 operations)
Light sources.

| Operation | Description |
|-----------|-------------|
| `b_pointlight` | Point light |
| `b_arealight` | Area light |
| `b_spotlight` | Spot light |
| `b_ieslight` | IES light |

---

## Backend Coverage Summary

| Backend | Total Ops | Coverage | Curves | Surfaces | Solids | Boolean | BIM | Selection | Rendering |
|---------|-----------|----------|--------|----------|--------|---------|-----|-----------|-----------|
| **Rhino** | 84 | 65% | 94% | 83% | 93% | 38% | 0% | 100% | 62% |
| **AutoCAD** | 58 | 45% | 94% | 75% | 87% | 62% | 0% | 100% | 46% |
| **Blender** | 44 | 34% | 65% | 42% | 53% | 23% | 0% | 0% | 46% |
| **Unity** | 33 | 25% | 65% | 33% | 40% | 0% | 0% | 0% | 0% |
| **Three.js** | 29 | 22% | 65% | 42% | 47% | 0% | 0% | 0% | 15% |
| **Revit** | 24 | 18% | 0% | 0% | 40% | 0% | 71% | 0% | 31% |
| **TikZ** | 24 | 18% | 53% | 33% | 0% | 0% | 0% | 0% | 0% |
| **POVRay** | 20 | 15% | 0% | 0% | 40% | 0% | 0% | 0% | 15% |
| **Radiance** | 8 | 6% | 0% | 8% | 20% | 0% | 0% | 0% | 8% |

---

## Backend Profiles

### Rhino (RH) - General Purpose CAD
**Best for:** General 3D modeling, interactive selection, curve/surface work

- **Strengths:** Most comprehensive implementation, excellent curve/surface support, full selection
- **Weaknesses:** No BIM operations
- **Total:** ~84 operations

### AutoCAD (ACAD) - CAD/Drafting
**Best for:** 2D/3D CAD, technical drawings, interactive workflows

- **Strengths:** Strong curve/solid support, full selection, good boolean ops
- **Weaknesses:** Limited advanced modeling (sweep/loft)
- **Total:** ~58 operations

### Blender (BLR) - 3D Graphics
**Best for:** Rendering, visualization, material-heavy scenes

- **Strengths:** Full lighting support, good material system, rendering
- **Weaknesses:** No selection, limited boolean ops
- **Total:** ~44 operations

### Unity - Game Engine
**Best for:** Real-time visualization, game development

- **Strengths:** Real-time rendering, game integration
- **Weaknesses:** Limited geometry ops, no selection/BIM
- **Note:** Y-Z axis swap for Unity coordinate system
- **Total:** ~33 operations

### Three.js (THR) - Web Graphics
**Best for:** Web-based 3D visualization

- **Strengths:** Browser-based, lightweight
- **Weaknesses:** Limited advanced modeling
- **Total:** ~29 operations

### Revit (RVT) - BIM
**Best for:** Building design, BIM workflows

- **Strengths:** Best BIM support (71%), architectural elements
- **Weaknesses:** Limited basic geometry, no selection
- **Total:** ~24 operations

### TikZ - 2D Vector Graphics
**Best for:** Technical documentation, LaTeX integration

- **Strengths:** Excellent annotation support (89%), 2D graphics
- **Weaknesses:** No 3D support
- **Total:** ~24 operations

### POVRay - Ray Tracing
**Best for:** High-quality renders, ray-traced images

- **Strengths:** Ray-tracing quality
- **Weaknesses:** Highly specialized, limited geometry
- **Total:** ~20 operations

### Radiance - Lighting Simulation
**Best for:** Lighting analysis, daylighting studies

- **Strengths:** Accurate lighting simulation
- **Weaknesses:** Very specialized, minimal geometry
- **Total:** ~8 operations

---

## Choosing a Backend

| Use Case | Recommended Backend |
|----------|---------------------|
| General 3D modeling | Rhino |
| Technical CAD drawings | AutoCAD |
| Architectural BIM | Revit |
| High-quality renders | Blender, POVRay |
| Web visualization | Three.js |
| Game integration | Unity |
| Documentation figures | TikZ |
| Lighting analysis | Radiance |
| Interactive selection | Rhino, AutoCAD |

---

## Notes

### Axis Conventions
- Most backends use right-handed Z-up coordinate system
- **Unity** swaps Y and Z axes to match its Y-up convention

### Reference Types
Different backends use different reference types based on their native APIs:
- `Int64`: AutoCAD
- `Int32`: Blender, Unity, Three.js, FreeCAD, 3dsMax
- `UInt128` (GUID): Rhino
- `Any`: Makie, TikZ (local backends)

### Encoding Protocols
- `CS`: C# backends (AutoCAD, Revit, Rhino, Unity)
- `PY`: Python backends (Blender, FreeCAD)
- `TS`: TypeScript backends (Three.js)
- None: Local backends (Makie, TikZ, POVRay, Radiance)
