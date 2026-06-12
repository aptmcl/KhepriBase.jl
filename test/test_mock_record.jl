# test_mock_record.jl — Tier 0 app-free correctness of the front-end lowering.
#
# The MockBackend records the PARAMETERS of every b_* call into type-keyed
# vectors. Running a high-level script against it and asserting the recorded
# parameters equal the construction parameters verifies that the front-end
# lowers shapes correctly — independent of any external application. This is the
# "recorded geometry == analytic oracle" upgrade over conformance's weaker
# "the op returned a non-void ref". See Docs/TestingStrategy.md.

using KhepriBase
using Test
isdefined(Main, :MockBackend) || include("TestMockBackend.jl")
isdefined(Main, :Oracles) || include("Oracles.jl")
using .Oracles

# Location p is at world coordinates t (a 3-tuple)?  /  same world location as q?
at_world(p, t; atol=1e-9) = all(isapprox.(world_xyz(p), t; atol=atol))
same_world(p, q; atol=1e-9) = all(isapprox.(world_xyz(p), world_xyz(q); atol=atol))

@testset "Mock-backend recording == construction (Tier 0)" begin

  @testset "sphere records center + radius" begin
    with_mock_backend() do b
      sphere(xyz(1, 2, 3), 4)
      @test length(b.spheres) == 1
      s = only(b.spheres)
      @test at_world(s.center, (1.0, 2.0, 3.0))
      @test s.radius == 4
    end
  end

  @testset "box records corner + extents" begin
    with_mock_backend() do b
      box(xyz(2, -1, 0), 5, 6, 7)
      @test length(b.boxes) == 1
      bx = only(b.boxes)
      @test at_world(bx.corner, (2.0, -1.0, 0.0))
      @test (bx.dx, bx.dy, bx.dz) == (5, 6, 7)
    end
  end

  @testset "cylinder records center + radius + height" begin
    with_mock_backend() do b
      cylinder(xyz(0, 0, 1), 3, 10)
      @test length(b.cylinders) == 1
      c = only(b.cylinders)
      @test at_world(c.center, (0.0, 0.0, 1.0))
      @test c.radius == 3
      @test c.height == 10
    end
  end

  @testset "circle records center + radius" begin
    with_mock_backend() do b
      circle(xy(4, 5), 2)
      @test length(b.circles) == 1
      c = only(b.circles)
      @test at_world(c.center, (4.0, 5.0, 0.0))
      @test c.radius == 2
    end
  end

  @testset "arc records center, radius, angles" begin
    with_mock_backend() do b
      arc(u0(), 5, π/6, π/3)
      @test length(b.arcs) == 1
      a = only(b.arcs)
      @test at_world(a.center, (0.0, 0.0, 0.0))
      @test a.radius == 5
      @test a.start_angle ≈ π/6
      @test a.amplitude ≈ π/3
    end
  end

  @testset "point + line + polygon + rectangle record their vertices" begin
    with_mock_backend() do b
      point(xyz(7, 8, 9))
      @test at_world(only(b.points).position, (7.0, 8.0, 9.0))
    end
    with_mock_backend() do b
      line(xy(0, 0), xy(3, 0), xy(3, 4))
      vs = only(b.lines).vertices
      @test length(vs) == 3
      @test at_world(vs[end], (3.0, 4.0, 0.0))
    end
    with_mock_backend() do b
      polygon(xy(0, 0), xy(2, 0), xy(1, 2))   # mock closes the loop
      vs = only(b.lines).vertices
      @test length(vs) == 4
      @test same_world(vs[1], vs[end])
    end
    with_mock_backend() do b
      rectangle(xy(0, 0), 4, 3)               # mock records 5 corner points
      vs = only(b.lines).vertices
      @test length(vs) == 5
      @test at_world(vs[3], (4.0, 3.0, 0.0))  # far corner
    end
  end

  @testset "mock_geometry_stats counts match the script" begin
    with_mock_backend() do b
      sphere(u0(), 1)
      sphere(x(5), 2)
      box(x(10), 1, 1, 1)
      circle(x(15), 3)
      stats = mock_geometry_stats(b)
      @test stats.spheres == 2
      @test stats.boxes == 1
      @test stats.circles == 1
      @test stats.total_refs == 4
    end
  end

  @testset "a surface the mock lacks natively decomposes to recorded triangles" begin
    # surface_polygon has no native mock op → it must fall back to b_trig, which
    # the mock records. A square triangulates to exactly 2 triangles.
    with_mock_backend() do b
      surface_polygon(u0(), ux(), uxy(), uy())
      @test length(b.triangles) == 2
      @test isempty(b.spheres) && isempty(b.boxes)   # nothing mis-routed
    end
  end
end
