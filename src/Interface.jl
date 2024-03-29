using Base: @kwdef
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
  backend,
  backend_bounding_box,
  #backend_captured_shape,
  #backend_captured_shapes,
  #backend_changed_shape,
  #backend_create_block,
  #backend_disable_update,
  #backend_enable_update,
  backend_fill_curves,
  backend_frame_at,
  #backend_generate_captured_shape,
  #backend_generate_captured_shapes,
  backend_get_family_ref,
  #backend_highlight_shape,
  #backend_highlight_shapes,
  #backend_loft_curve_point,
  #backend_loft_curves,
  #backend_loft_surface_point,
  #backend_loft_surfaces,
  #backend_map_division,
  backend_name,
  #backend_panel,
  #backend_pre_selected_shapes_from_set,
  #backend_right_cuboid,
  #backend_rotation_minimizing_frames,
  #backend_set_length_unit,
  #backend_shape_from_ref,
  #backend_spotlight,
  #backend_stroke_arc,
  #backend_stroke_unite,
  #backend_surface_boundary,
  #backend_surface_domain,
  #backend_wall,
  decode,
  encode,
  parse_signature,
  realize,
  #slice_ref,
  maybe_merged_node,
  maybe_merged_bar,
  save_shape!,
  switch_to_layer,
  truss_bar_family_cross_section_area,
  with_material_as_layer,
  use_material_as_layer
