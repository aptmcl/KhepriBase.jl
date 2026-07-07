# Auto-generated Khepri code from Revit model
# Generated on: <timestamp>
using KhepriRevit

level_0 = level(0.0)
level_1 = level(3.0)

wall_w_t = wall_family("wall_family", 0.2, 0.0, 0.0)
set_backend_family(wall_w_t, revit, revit_system_family())
door_single_flush_0915 = door_family("Single-Flush:0915", 1.0, 2.0, 0.05, frame_family("frame_family", rectangular_path(xy(-0.08, -0.08), 0.16, 0.16)))
set_backend_family(door_single_flush_0915,
  revit,
  revit_file_family(raw"C:\A.rfa"))
door_double_flush_1830 = door_family("Double-Flush:1830", 1.0, 2.0, 0.05, frame_family("frame_family", rectangular_path(xy(-0.08, -0.08), 0.16, 0.16)))
set_backend_family(door_double_flush_1830,
  revit,
  revit_file_family(raw"C:\B.rfa"))

let __wall = wall(open_polygonal_path([xy(0, 0), xy(6, 0)]), bottom_level = level_0, top_level = level_1, family = wall_w_t)
    add_door(__wall, xy(1.0, 0.0), door_single_flush_0915)
    add_door(__wall, xy(4.0, 0.0), door_double_flush_1830)
    __wall
end
