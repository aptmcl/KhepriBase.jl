# test_roundtrip.jl — Headless round-trip stress tests for the shape → Khepri-source → shape pipeline
# that underlies both the KhepriAutoCAD geometric round-trip and the KhepriRevit BIM introspection.
#
# The codegen emit path (meta_program → _expr_str) and its inverse (parse → constructor) are backend-
# agnostic, so we exercise them on the headless MockBackend under `with_introspection` (no external app).
# Each shape is checked three ways:
#   • reparses     — the emitted source is valid Julia (guards the printer, e.g. the a:step:b range bug).
#   • roundtrips   — emitting the reconstruction reproduces byte-identical source (a fixed point; guards
#                    representation stability + non-reproducible numeric noise).
#   • cb-frame     — for cs-based solids only: the reconstructed base loc reproduces position AND
#                    orientation within tolerance. Idempotence alone CANNOT catch a *consistent*
#                    coordinate-system drop (both emits would drop identically), so this geometric check
#                    is what actually guards the cylinder/cone/pyramid cs-preservation fixes.

using Test
using KhepriBase
include(joinpath(@__DIR__, "TestMockBackend.jl"))

const KB = KhepriBase

emit(s) = KB._expr_str(KB.meta_program(s))
reconstruct(s) = Core.eval(@__MODULE__, Meta.parse(emit(s)))

function roundtrips(s)
  e1 = emit(s)
  e2 = emit(reconstruct(s))
  e1 == e2 || @info "round-trip DIFF" original=e1 reconstructed=e2
  e1 == e2
end
reparses(s) = (Meta.parse(emit(s)); true)

# Position + full orientation of a cs-based shape's base loc must survive the round-trip. Compare four
# world probe points (origin + the three unit axes) between original and reconstruction.
function cb_frame_roundtrips(s; atol=1e-6)
  cb1, cb2 = s.cb, reconstruct(s).cb
  all(distance(in_world(p(cb1)), in_world(p(cb2))) < atol
      for p in (identity, cb -> add_x(cb, 1.0), cb -> add_y(cb, 1.0), cb -> add_z(cb, 1.0)))
end

# Every shape: reparse + idempotence; cs-based shapes (those carrying a `.cb`) also get the geometric check.
function check(s)
  @test reparses(s)
  @test roundtrips(s)
  if hasfield(typeof(s), :cb)
    @test cb_frame_roundtrips(s)
  end
end

const custom_mat = pbr_material("TestMat", base_color=rgba(0.2, 0.6, 0.9, 1.0), roughness=0.3, metallic=0.8)
tilt(o, v) = KB.loc_from_o_vz(o, v)   # a non-world cs (base loc with a tilted +z axis)

@testset "round-trip stress tests" begin
  with_mock_backend() do b
    with_introspection(b) do

      @testset "2D curves + points" begin
        for (s, l) in [
            (point(xyz(1.0, 2.0, 3.0)), "point"),
            (line(xyz(0.0, 0.0, 0.0), xyz(5.0, 5.0, 5.0)), "line-3pt"),
            (line(xy(0.0, 0.0), xy(3.0, 0.0), xy(3.0, 4.0)), "polyline"),
            (polygon(xy(0.0, 0.0), xy(4.0, 0.0), xy(4.0, 3.0), xy(0.0, 3.0)), "polygon"),
            (circle(xyz(2.0, 3.0, 1.0), 5.0), "circle"),
            (arc(xy(1.0, 1.0), 3.0, 0.0, pi/2), "arc"),
            (arc(xyz(1.0, 1.0, 2.0), 3.0, pi/4, pi), "arc-3d"),
            (ellipse(xy(2.0, 2.0), 4.0, 2.0), "ellipse"),
            (spline([xy(0.0, 0.0), xy(2.0, 3.0), xy(5.0, 1.0), xy(7.0, 4.0)]), "spline"),
            (regular_polygon(6, xyz(3.0, 3.0, 0.0), 2.0, 0.0, true), "regular_polygon"),
          ]
          @testset "$l" begin; check(s); end
        end
      end

      @testset "surfaces" begin
        for (s, l) in [
            (surface_circle(xyz(1.0, 2.0, 0.0), 4.0), "surface_circle"),
            (surface_polygon(xy(0.0, 0.0), xy(5.0, 0.0), xy(5.0, 5.0)), "surface_polygon"),
            (surface_rectangle(xyz(0.0, 0.0, 1.0), 4.0, 3.0), "surface_rectangle"),
            (surface_arc(xy(0.0, 0.0), 3.0, 0.0, pi/2), "surface_arc"),
          ]
          @testset "$l" begin; check(s); end
        end
      end

      @testset "3D primitives (axis-aligned)" begin
        for (s, l) in [
            (sphere(xyz(1.0, 2.0, 3.0), 2.5), "sphere"),
            (box(xyz(1.0, 1.0, 1.0), 2.0, 3.0, 4.0), "box"),
            (right_cuboid(xyz(5.0, 0.0, 0.0), 2.0, 3.0, 5.0), "right_cuboid"),
            (cylinder(xyz(0.0, 0.0, 0.0), 1.0, xyz(0.0, 0.0, 5.0)), "cylinder"),
            (cone(xyz(2.0, 0.0, 0.0), 2.0, xyz(2.0, 0.0, 5.0)), "cone"),
            (cone_frustum(xyz(4.0, 0.0, 0.0), 3.0, xyz(4.0, 0.0, 4.0), 1.0), "cone_frustum"),
            (torus(xyz(0.0, 0.0, 3.0), 4.0, 1.0), "torus"),
            (regular_pyramid(4, xyz(6.0, 0.0, 0.0), 2.0, 0.0, 3.0), "regular_pyramid"),
            (regular_prism(6, xyz(8.0, 0.0, 0.0), 1.5, 0.0, 4.0), "regular_prism"),
            (regular_pyramid_frustum(5, xyz(10.0, 0.0, 0.0), 2.0, 0.0, 3.0, 1.0), "regular_pyramid_frustum"),
          ]
          @testset "$l" begin; check(s); end
        end
      end

      @testset "cs-based solids (TILTED — cs-drop regression guard)" begin
        for (s, l) in [
            (cylinder(xyz(1.0, 0.0, 0.0), 0.5, xyz(4.0, 3.0, 2.0)), "cylinder-tilted"),
            (cone(tilt(xyz(2.0, 0.0, 0.0), vxyz(1.0, 1.0, 1.0)), 2.0, 4.0), "cone-tilted"),
            (cone_frustum(xyz(4.0, 0.0, 0.0), 3.0, xyz(6.0, 3.0, 2.0), 1.0), "cone_frustum-tilted"),
            (regular_pyramid(4, tilt(xyz(0.0, 0.0, 0.0), vxyz(1.0, 1.0, 1.0)), 2.0, 0.0, 3.0), "pyramid-tilted"),
            (regular_prism(6, tilt(xyz(5.0, 0.0, 0.0), vxyz(0.0, 1.0, 1.0)), 1.5, 0.0, 4.0), "prism-tilted"),
            (regular_pyramid_frustum(5, tilt(xyz(9.0, 0.0, 0.0), vxyz(1.0, 0.0, 1.0)), 2.0, 0.0, 3.0, 1.0), "pyr-frustum-tilted"),
          ]
          @testset "$l" begin; check(s); end
        end
      end

      @testset "generated solids/surfaces" begin
        for (s, l) in [
            (extrusion(surface_circle(xy(0.0, 0.0), 2.0), vz(5.0)), "extrusion"),
            (sweep(arc(xy(0.0, 0.0), 5.0, 0.0, pi/2), surface_circle(xy(5.0, 0.0), 0.5)), "sweep"),
          ]
          @testset "$l" begin; check(s); end
        end
      end

      @testset "materials round-trip on shapes" begin
        for (s, l) in [
            (sphere(xyz(0.0, 0.0, 0.0), 1.0, material_metal), "canonical-metal"),
            (box(xyz(2.0, 0.0, 0.0), 1.0, 1.0, 1.0, material_glass), "canonical-glass"),
            (cylinder(xyz(4.0, 0.0, 0.0), 1.0, xyz(4.0, 0.0, 3.0), custom_mat), "custom-pbr"),
            (surface_circle(xy(6.0, 0.0), 1.0, custom_mat), "surface-custom"),
          ]
          @testset "$l" begin; check(s); end
        end
      end

      @testset "BIM elements" begin
        l0, l1 = level(0.0), level(3.0)
        for (s, l) in [
            (wall(open_polygonal_path([xyz(0.0, 0.0, 0.0), xyz(6.0, 0.0, 0.0)]), bottom_level=l0, top_level=l1), "wall"),
            (slab(closed_polygonal_path([xy(0.0, 0.0), xy(6.0, 0.0), xy(6.0, 4.0), xy(0.0, 4.0)]), level=l0), "slab"),
            (column(xyz(0.0, 0.0, 0.0), bottom_level=l0, top_level=l1), "column"),
            (beam(xyz(0.0, 0.0, 3.0), xyz(6.0, 0.0, 3.0)), "beam-horizontal"),
            (beam(xyz(0.0, 0.0, 0.0), xyz(3.0, 4.0, 5.0)), "beam-tilted"),
            (ceiling(closed_polygonal_path([xy(0.0, 0.0), xy(4.0, 0.0), xy(4.0, 4.0)]), level=l1), "ceiling"),
            (roof(closed_polygonal_path([xy(0.0, 0.0), xy(4.0, 0.0), xy(4.0, 4.0)]), level=l1), "roof"),
          ]
          @testset "$l" begin; check(s); end
        end
      end

    end
  end
end
