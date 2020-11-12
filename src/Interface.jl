using Reexport
@reexport using KhepriBase
using Dates
using ColorTypes

# resolve conflicts
using KhepriBase:
  XYZ,
  Text

import Base:
 show

import KhepriBase:
  backend_all_shapes,
  backend_all_shapes_in_layer,
  backend_bounding_box,
  backend_captured_shape,
  backend_captured_shapes,
  backend_chair,
  backend_changed_shape,
  backend_create_block,
  backend_create_layer,
  backend_current_layer,
  backend_current_material,
  backend_cylinder,
  backend_delete_all_shapes,
  backend_delete_all_shapes_in_layer,
  backend_delete_shapes,
  backend_dimension,
  backend_dimension,
  backend_disable_update,
  backend_enable_update,
  backend_extrusion,
  backend_fill,
  #backend_fill_curves,
  #backend_frame_at,
  #backend_generate_captured_shape,
  #backend_generate_captured_shapes,
  backend_get_family_ref,
  backend_get_material,
  backend_get_view,
  #backend_highlight_shape,
  #backend_highlight_shapes,
  backend_ieslight,
  #backend_loft_curve_point,
  #backend_loft_curves,
  #backend_loft_surface_point,
  #backend_loft_surfaces,
  #backend_map_division,
  backend_name,
  backend_node_displacement_function,
  #backend_panel,
  backend_pointlight,
  backend_polygon,
  #backend_pre_selected_shapes_from_set,
  backend_pyramid,
  backend_pyramid_frustum,
  backend_realistic_sky,
  backend_realize_beam_profile,
  backend_rectangular_table,
  backend_rectangular_table_and_chairs,
  backend_regular_pyramid,
  backend_render_view,
  backend_revolve_curve,
  backend_revolve_point,
  backend_revolve_surface,
  backend_right_cuboid,
  #backend_rotation_minimizing_frames,
  backend_select_curve,
  backend_select_curves,
  backend_select_point,
  backend_select_points,
  backend_select_position,
  backend_select_positions,
  backend_select_shape,
  backend_select_shapes,
  backend_select_solid,
  backend_select_solids,
  backend_select_surface,
  backend_select_surfaces,
  #backend_set_length_unit,
  backend_set_view,
  #backend_shape_from_ref,
  backend_show_truss_deformation,
  backend_slab,
  backend_sphere,
  #backend_spotlight,
  backend_stroke,
  backend_stroke_arc,
  #backend_stroke_color,
  backend_stroke_unite,
  #backend_surface_boundary,
  #backend_surface_domain,
  backend_surface_grid,
  backend_surface_polygon,
  backend_truss_analysis,
  #backend_sweep,
  backend_view_top,
  #backend_wall,
  backend_zoom_extents,
  decode,
  encode,
  intersect_ref,
  parse_signature,
  realize,
  #slice_ref,
  maybe_merged_node,
  maybe_merged_bar,
  save_shape!,
  subtract_ref,
  switch_to_layer,
  unite_ref
  #unite_refs,
