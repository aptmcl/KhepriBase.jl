# Auto-generated Khepri code from Revit model
# Generated on: <timestamp>
using KhepriRevit

wall_top_offset = -0.163
columns_y_spacing = 2.0
n_columns_y = 2
columns_x_spacing = 4.0
n_columns_x = 2
floor_height = 3.26
n_levels = 3

levels = [level(i * floor_height) for i = 0:n_levels - 1]
level_0 = levels[1]
level_1 = levels[2]
level_2 = levels[3]
level_3 = unconnected_level(2 * floor_height + 0.5)

wall_basic_wall_generic_200mm = wall_family("wall_family", 0.2, 0.0, 0.0)
@isdefined(revit) && set_backend_family(wall_basic_wall_generic_200mm, revit, revit_system_family(type_name="Basic Wall:Generic 200mm"))
slab_floor_generic_300mm = slab_family("slab_family", 0.2, 0.0)
@isdefined(revit) && set_backend_family(slab_floor_generic_300mm, revit, revit_system_family(type_name="Floor:Generic 300mm"))
column_m_concrete_rectangular_300_x_450mm = column_family("column_family", rectangular_path(xy(-0.1, -0.1), 0.2, 0.2))
@isdefined(revit) && set_backend_family(column_m_concrete_rectangular_300_x_450mm, revit, revit_file_family(raw"C:\Families\Columns\Concrete-Rectangular.rfa"))
family_element_desk_lamp_std = family_element_family("family_element_family")
@isdefined(revit) && set_backend_family(family_element_desk_lamp_std, revit, revit_file_family(raw"C:\Families\Fixtures\Lamp.rfa"))

building_origin = xy(0.0, 0.0)

# Storey storey_0 (level_0 = 0.0)

function storey_0(p0=building_origin)
  wall(open_polygonal_path([add_xy(p0, 0.0, 0.0), add_xy(p0, 6.0, 0.0)]),
    bottom_level=level_0,
    top_level=level_1,
    family=wall_basic_wall_generic_200mm)
  wall(open_polygonal_path([add_xy(p0, 0.0, 4.0), add_xy(p0, 0.0, 0.0)]),
    bottom_level=level_0,
    top_level=level_1,
    family=wall_basic_wall_generic_200mm)
  wall(open_polygonal_path([add_xy(p0, 6.0, 0.0), add_xy(p0, 6.0, 4.0)]),
    bottom_level=level_0,
    top_level=level_1,
    family=wall_basic_wall_generic_200mm)
  wall(open_polygonal_path([add_xy(p0, 6.0, 4.0), add_xy(p0, 0.0, 4.0)]),
    bottom_level=level_0,
    top_level=level_1,
    family=wall_basic_wall_generic_200mm)
  slab(region(closed_polygonal_path([add_xy(p0, 0.0, 0.0), add_xy(p0, 6.0, 0.0), add_xy(p0, 6.0, 4.0), add_xy(p0, 0.0, 4.0)])),
    level_0,
    slab_floor_generic_300mm)
  for x = range(1.0, step=columns_x_spacing, length=n_columns_x)
    for y = range(1.0, step=columns_y_spacing, length=n_columns_y)
      column(add_xy(p0, x, y),
        0,
        level_0,
        level_1,
        column_m_concrete_rectangular_300_x_450mm)
    end
  end
  family_element(add_xy(p0, 1.5, 3.0),
    pi / 2,
    level_0,
    family_element_desk_lamp_std)
  family_element(add_xy(p0, 3.0, 3.0),
    pi,
    level_0,
    family_element_desk_lamp_std)
  family_element(add_xy(p0, 4.5, 3.0),
    -(pi / 2),
    level_0,
    family_element_desk_lamp_std)

  group_instance(group_desk, add_xyz(p0, 2.0, 1.0, 0.0))
  group_instance(group_desk, add_xyz(p0, 8.0, 1.0, 0.0))
end

# Storey storey_1 (level_1 = 3.26)

function storey_1(p0=building_origin)
  wall(open_polygonal_path([add_xy(p0, 0.0, 0.0), add_xy(p0, 6.0, 0.0)]),
    bottom_level=level_1,
    top_level=level_2,
    family=wall_basic_wall_generic_200mm,
    top_offset=wall_top_offset)
  wall(open_polygonal_path([add_xy(p0, 0.0, 4.0), add_xy(p0, 0.0, 0.0)]),
    bottom_level=level_1,
    top_level=level_2,
    family=wall_basic_wall_generic_200mm,
    top_offset=wall_top_offset)
  wall(open_polygonal_path([add_xy(p0, 6.0, 0.0), add_xy(p0, 6.0, 4.0)]),
    bottom_level=level_1,
    top_level=level_2,
    family=wall_basic_wall_generic_200mm,
    top_offset=wall_top_offset)
  wall(open_polygonal_path([add_xy(p0, 6.0, 4.0), add_xy(p0, 0.0, 4.0)]),
    bottom_level=level_1,
    top_level=level_2,
    family=wall_basic_wall_generic_200mm,
    top_offset=wall_top_offset)
end

# Storey storey_2 (level_2 = 6.52)

function storey_2(p0=building_origin)
  wall(open_polygonal_path([add_xy(p0, 0.0, 4.0), add_xy(p0, 6.0, 4.0)]),
    bottom_level=level_2,
    top_level=level_3,
    family=wall_basic_wall_generic_200mm)
end
function group_desk_factory()
  column(xy(0.2, 0.2),
    0,
    level_0,
    level_1,
    column_m_concrete_rectangular_300_x_450mm)
  slab(region(closed_polygonal_path([xy(0.0, 0.0), xy(1.0, 0.0), xy(1.0, 0.5), xy(0.0, 0.5)])),
    level_0,
    slab_floor_generic_300mm)
end

group_desk = group("group_desk", factory=group_desk_factory)

# Build the storeys, bottom-up

storey_0()
storey_1()
storey_2()
finalize_groups()
