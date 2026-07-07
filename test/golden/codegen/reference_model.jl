# Auto-generated Khepri code from Revit model
# Generated on: <timestamp>
using KhepriRevit

level_0 = level(0.0)
level_1 = level(3.0)
level_2 = level(6.0)

wall_basic_wall_generic_200mm = wall_family("wall_family", 0.2, 0.0, 0.0)
set_backend_family(wall_basic_wall_generic_200mm, revit, revit_system_family())
door_single_flush_0915_x_2134mm = door_family("Single-Flush:0915 x 2134mm", 0.915, 2.134, 0.05, frame_family("frame_family", rectangular_path(xy(-0.08, -0.08), 0.16, 0.16)))
set_backend_family(door_single_flush_0915_x_2134mm,
  revit,
  revit_file_family(raw"C:\Families\Doors\Single-Flush.rfa"))
window_fixed_0915_x_1220mm = window_family("Fixed:0915 x 1220mm", 0.915, 1.22, 0.05, frame_family("frame_family", rectangular_path(xy(-0.08, -0.08), 0.16, 0.16)))
set_backend_family(window_fixed_0915_x_1220mm,
  revit,
  revit_file_family(raw"C:\Families\Windows\Fixed.rfa"))
slab_floor_generic_300mm = slab_family("slab_family", 0.2, 0.0)
set_backend_family(slab_floor_generic_300mm, revit, revit_system_family())
column_m_concrete_rectangular_300_x_450mm = column_family("column_family", rectangular_path(xy(-0.1, -0.1), 0.2, 0.2))
set_backend_family(column_m_concrete_rectangular_300_x_450mm,
  revit,
  revit_file_family(raw"C:\Families\Columns\Concrete-Rectangular.rfa"))

let __wall = wall(open_polygonal_path([xy(0, 0), xy(6, 0)]), bottom_level = level_0, top_level = level_1, family = wall_basic_wall_generic_200mm)
    add_door(__wall, xy(1.0, 0.0), door_single_flush_0915_x_2134mm)
    add_window(__wall, xy(3.0, 0.9), window_fixed_0915_x_1220mm)
    __wall
end

wall(open_polygonal_path([xy(6, 0), xy(6, 4)]),
  bottom_level=level_0,
  top_level=level_1,
  family=wall_basic_wall_generic_200mm)
wall(open_polygonal_path([xy(6, 4), xy(0, 4)]),
  bottom_level=level_0,
  top_level=level_1,
  family=wall_basic_wall_generic_200mm)
wall(open_polygonal_path([xy(0, 4), xy(0, 0)]),
  bottom_level=level_0,
  top_level=level_1,
  family=wall_basic_wall_generic_200mm)
slab(region(closed_polygonal_path([xy(0, 0), xy(6, 0), xy(6, 4), xy(0, 4)])),
  level_0,
  slab_floor_generic_300mm)
column(xy(0.0, 0.0),
  0,
  level_0,
  level_1,
  column_m_concrete_rectangular_300_x_450mm)
column(xy(0.0, 4.0),
  0,
  level_0,
  level_1,
  column_m_concrete_rectangular_300_x_450mm)
column(xy(0.0, 8.0),
  0,
  level_0,
  level_1,
  column_m_concrete_rectangular_300_x_450mm)
column(xy(5.0, 0.0),
  0,
  level_0,
  level_1,
  column_m_concrete_rectangular_300_x_450mm)
column(xy(5.0, 4.0),
  0,
  level_0,
  level_1,
  column_m_concrete_rectangular_300_x_450mm)
column(xy(5.0, 8.0),
  0,
  level_0,
  level_1,
  column_m_concrete_rectangular_300_x_450mm)
column(xy(10.0, 0.0),
  0,
  level_0,
  level_1,
  column_m_concrete_rectangular_300_x_450mm)
column(xy(10.0, 4.0),
  0,
  level_0,
  level_1,
  column_m_concrete_rectangular_300_x_450mm)
column(xy(10.0, 8.0),
  0,
  level_0,
  level_1,
  column_m_concrete_rectangular_300_x_450mm)
column(xy(15.0, 0.0),
  0,
  level_0,
  level_1,
  column_m_concrete_rectangular_300_x_450mm)
column(xy(15.0, 4.0),
  0,
  level_0,
  level_1,
  column_m_concrete_rectangular_300_x_450mm)
column(xy(15.0, 8.0),
  0,
  level_0,
  level_1,
  column_m_concrete_rectangular_300_x_450mm)
