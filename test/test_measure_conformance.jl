# test_measure_conformance.jl — headless acceptance regressions for the
# realization-layer bug classes found on the Revit round-trip corpus, measured
# on the MeasureBackend. Each testset reconstructs one historical bug scenario
# and asserts the geometry that USED to be wrong:
#
#   1. OBJ fixture instance angle dropped (cabinets through walls)
#   2. cs-baked angle dropped by in_world (Table/Chair via loc_from_o_phi)
#   3. mirrored placement rendered as pure rotation (door-handle class):
#      chirality via signed volume
#   4. grouped slab double-translated (current_cs baked into both profile and
#      height cb)
#   5. group instance translation applied exactly once to members
#
# These are the mutation tests that keep the measurement layer honest: if a
# future refactor reintroduces any of these, the corresponding assertion names
# the class directly.

using Test
using KhepriBase

# An asymmetric Z-up OBJ fixture: 1.0 long in +x, 0.4 deep in -y (like a
# cabinet whose body extends to one side), 0.5 tall. Asymmetry is what makes
# orientation measurable.
const CONF_OBJ_DIR = mktempdir()
write(joinpath(CONF_OBJ_DIR, "conf_cabinet.obj"), """
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 -0.4 0.0
v 0.0 -0.4 0.0
v 0.0 0.0 0.5
v 1.0 0.0 0.5
v 1.0 -0.4 0.5
v 0.0 -0.4 0.5
f 1 4 3 2
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
""")
add_resource_folder!(CONF_OBJ_DIR)

conf_measure(f) =
  let b = measure_backend(),
      prev = KhepriBase.current_backends()
    KhepriBase.current_backend(b)
    try
      f()
    finally
      KhepriBase.current_backends(prev)
    end
    measured_shapes(b)
  end

approx3(p, x, y, z; tol=1e-6) =
  abs(cx(p) - x) <= tol && abs(cy(p) - y) <= tol && abs(cz(p) - z) <= tol

@testset "measure conformance (historical bug classes)" begin

  @testset "OBJ fixture honors the instance angle" begin
    # Unrotated: body spans x 0..1, y -0.4..0 from the origin.
    let fam = family_element_family("conf"),
        _ = set_backend_family(fam, KhepriBase.MeasureBackend, obj_family("conf_cabinet")),
        rows = conf_measure() do
          family_element(xyz(10.0, 20.0, 0.0), 0.0, level(0.0), fam)
        end,
        m = only([r for r in rows if r.shape isa KhepriBase.FamilyElement]).measurement
      @test approx3(m.aabb_min, 10.0, 19.6, 0.0)
      @test approx3(m.aabb_max, 11.0, 20.0, 0.5)
    end
    # Rotated 90°: local +x → world +y, local -y → world -x wait: R(π/2): x→y, y→-x;
    # body local y ∈ [-0.4, 0] → world x ∈ [0, 0.4]. The bug class rendered this
    # IDENTICAL to the unrotated case.
    let fam = family_element_family("conf2"),
        _ = set_backend_family(fam, KhepriBase.MeasureBackend, obj_family("conf_cabinet")),
        rows = conf_measure() do
          family_element(xyz(10.0, 20.0, 0.0), pi / 2, level(0.0), fam)
        end,
        m = only([r for r in rows if r.shape isa KhepriBase.FamilyElement]).measurement
      @test approx3(m.aabb_min, 10.0, 20.0, 0.0)
      @test approx3(m.aabb_max, 10.4, 21.0, 0.5)
    end
  end

  @testset "cs-baked angle honored (loc_from_o_phi)" begin
    let fam = family_element_family("conf3"),
        _ = set_backend_family(fam, KhepriBase.MeasureBackend, obj_family("conf_cabinet")),
        rows = conf_measure() do
          family_element(loc_from_o_phi(xyz(0.0, 0.0, 0.0), pi / 2), 0.0, level(0.0), fam)
        end,
        m = only([r for r in rows if r.shape isa KhepriBase.FamilyElement]).measurement
      @test approx3(m.aabb_min, 0.0, 0.0, 0.0)
      @test approx3(m.aabb_max, 0.4, 1.0, 0.5)
    end
  end

  @testset "mirrored placement flips chirality (door-handle class)" begin
    let fam = family_element_family("conf4"),
        _ = set_backend_family(fam, KhepriBase.MeasureBackend, obj_family("conf_cabinet")),
        plain = conf_measure() do
          family_element(xyz(0.0, 0.0, 0.0), 0.0, level(0.0), fam)
        end,
        mirrored = conf_measure() do
          family_element(loc_from_o_vx_vy(xyz(0.0, 0.0, 0.0), vxyz(1, 0, 0), vxyz(0, -1, 0)),
                         0.0, level(0.0), fam)
        end,
        fe = rs -> only([r for r in rs if r.shape isa KhepriBase.FamilyElement]),
        sv0 = fe(plain).measurement.signed_volume,
        sv1 = fe(mirrored).measurement.signed_volume
      @test abs(sv0) > 1e-6               # closed shell: volume magnitude is real
      @test sign(sv0) != sign(sv1)        # mirroring flips orientation
      # And the mirrored body extends +y instead of -y:
      @test cy(fe(mirrored).measurement.aabb_max) > 0.39
    end
  end

  @testset "grouped slab lands at the group offset exactly once" begin
    let rows = conf_measure() do
          let g = group("g", factory=() ->
                    slab(region(closed_polygonal_path(
                           [xyz(0, 0, 0), xyz(2, 0, 0), xyz(2, 1, 0), xyz(0, 1, 0)])),
                         level(0.0), default_slab_family()))
            group_instance(g, xyz(10.0, 20.0, 0.0))
          end
        end,
        slabs = [r for r in rows if r.shape isa KhepriBase.Slab]
      @test length(slabs) == 1
      let m = slabs[1].measurement
        # The double-translation bug put this at (20, 40).
        @test abs(cx(m.centroid) - 11.0) <= 1e-6
        @test abs(cy(m.centroid) - 20.5) <= 1e-6
      end
    end
  end

  @testset "group members translated exactly once (walls too)" begin
    let rows = conf_measure() do
          let g = group("g2", factory=() ->
                    wall(open_polygonal_path([xy(0, 0), xy(3, 0)]),
                         bottom_level=level(0.0), top_level=level(3.0)))
            group_instance(g, xyz(-5.0, 7.0, 0.0))
          end
        end,
        walls = [r for r in rows if r.shape isa KhepriBase.Wall]
      @test length(walls) == 1
      let m = walls[1].measurement
        @test abs(cx(m.centroid) - (-3.5)) <= 1e-6
        @test abs(cy(m.centroid) - 7.0) <= 0.15   # wall thickness centering
      end
    end
  end
end

@testset "door flips are measurable (hinge + facing)" begin
  # An OBJ-mapped door on a wall: flip_x (hinge) mirrors the asymmetric mesh about
  # the opening center in the wall plane; flip_y (facing) mirrors it across the
  # wall plane and flips chirality. Neither may be a no-op.
  let dfam = door_family("confdoor", 1.0, 2.0),
      _ = set_backend_family(dfam, KhepriBase.MeasureBackend, obj_family("conf_cabinet")),
      run = (fx, fy) -> let rows = conf_measure() do
              let w = wall(open_polygonal_path([xy(0, 0), xy(6, 0)]),
                           bottom_level=level(0.0), top_level=level(3.0))
                add_door(w, xy(2.0, 0.0), dfam, flip_x=fx, flip_y=fy)
              end
            end
        only([r for r in rows if r.shape isa KhepriBase.Door]).measurement
      end,
      base = run(false, false),
      hinge = run(true, false),
      facing = run(false, true)
    # Hinge flip: the mesh's asymmetric x-extent mirrors about the opening
    # center (x=2.5): base occupies one side, flipped the other.
    @test abs((cx(base.centroid) - 2.5) + (cx(hinge.centroid) - 2.5)) <= 1e-6
    @test abs(cx(base.centroid) - cx(hinge.centroid)) > 1e-3
    # Facing flip: body crosses to the other side of the wall plane (y sign),
    # and chirality flips.
    @test sign(cy(base.centroid)) != sign(cy(facing.centroid))
    @test sign(base.signed_volume) != sign(facing.signed_volume)
  end
end
