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
  backend_frame_at,
  backend_get_family_ref,
  backend_name,
  decode,
  encode,
  parse_signature,
  realize,
  maybe_merged_node,
  maybe_merged_bar,
  save_shape!,
  switch_to_layer,
  truss_bar_family_cross_section_area,
  with_material_as_layer,
  use_material_as_layer
