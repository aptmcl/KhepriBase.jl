# Auto-generated Khepri code from Revit model
# Generated on: <timestamp>
using KhepriRevit

columns_y_spacing = 4.0
n_columns_y = 3
columns_x_spacing = 5.0
n_columns_x = 4
floor_height = 3.0
n_levels = 3

level_0 = level(0.0)
level_1 = level(floor_height)
level_2 = level(2 * floor_height)

opening_frame = frame_family("frame_family", rectangular_path(xy(-0.08, -0.08), 0.16, 0.16))

wall_basic_wall_generic_200mm = wall_family("wall_family", 0.2, 0.0, 0.0)
@isdefined(revit) && set_backend_family(wall_basic_wall_generic_200mm, revit, revit_system_family(type_name="Basic Wall:Generic 200mm"))
door_single_flush_0915_x_2134mm = door_family("Single-Flush:0915 x 2134mm", 0.915, 2.134, 0.05, opening_frame)
@isdefined(revit) && set_backend_family(door_single_flush_0915_x_2134mm, revit, revit_opening_file_family(raw"C:\Families\Doors\Single-Flush.rfa"))
window_fixed_0915_x_1220mm = window_family("Fixed:0915 x 1220mm", 0.915, 1.22, 0.05, opening_frame)
@isdefined(revit) && set_backend_family(window_fixed_0915_x_1220mm, revit, revit_opening_file_family(raw"C:\Families\Windows\Fixed.rfa"))
slab_floor_generic_300mm = slab_family("slab_family", 0.2, 0.0)
@isdefined(revit) && set_backend_family(slab_floor_generic_300mm, revit, revit_system_family(type_name="Floor:Generic 300mm"))
column_m_concrete_rectangular_300_x_450mm = column_family("column_family", rectangular_path(xy(-0.1, -0.1), 0.2, 0.2))
@isdefined(revit) && set_backend_family(column_m_concrete_rectangular_300_x_450mm, revit, revit_file_family(raw"C:\Families\Columns\Concrete-Rectangular.rfa"))

# Storey storey_0 (level_0 = 0.0)

function storey_0(p0=xy(0.0, 0.0))
  let __wall = wall(open_polygonal_path([p0 + vxy(0.0, 0.0), p0 + vxy(6.0, 0.0)]), bottom_level = level_0, top_level = level_1, family = wall_basic_wall_generic_200mm)
    add_door(__wall, xy(1.0, 0.0), door_single_flush_0915_x_2134mm)
    add_window(__wall, xy(3.0, 0.9), window_fixed_0915_x_1220mm)
    __wall
end

  wall(open_polygonal_path([p0 + vxy(0.0, 4.0), p0 + vxy(0.0, 0.0)]),
    bottom_level=level_0,
    top_level=level_1,
    family=wall_basic_wall_generic_200mm)
  wall(open_polygonal_path([p0 + vxy(6.0, 0.0), p0 + vxy(6.0, 4.0)]),
    bottom_level=level_0,
    top_level=level_1,
    family=wall_basic_wall_generic_200mm)
  wall(open_polygonal_path([p0 + vxy(6.0, 4.0), p0 + vxy(0.0, 4.0)]),
    bottom_level=level_0,
    top_level=level_1,
    family=wall_basic_wall_generic_200mm)
  slab(region(closed_polygonal_path([p0 + vxy(0.0, 0.0), p0 + vxy(6.0, 0.0), p0 + vxy(6.0, 4.0), p0 + vxy(0.0, 4.0)])),
    level_0,
    slab_floor_generic_300mm)
  for x = range(0.0, step=columns_x_spacing, length=n_columns_x)
    for y = range(0.0, step=columns_y_spacing, length=n_columns_y)
      column(xy(x, y),
        0,
        level_0,
        level_1,
        column_m_concrete_rectangular_300_x_450mm)
    end
  end
end

# Build the storeys, bottom-up

storey_0()
