# test_metamorphic.jl — Tier 0 metamorphic & analytic-anchor correctness.
#
# Metamorphic invariants need NO golden baseline: a correct implementation must
# satisfy them whatever the absolute values are. They catch transform-composition,
# unit, and handedness bugs that a golden silently embeds (because the golden
# would contain the same bug). See Docs/TestingStrategy.md.

using KhepriBase
using Test
isdefined(Main, :Oracles) || include("Oracles.jl")
using .Oracles

@testset "Metamorphic & analytic oracles (Tier 0)" begin

  @testset "Analytic length anchors (vs hand-derived literals)" begin
    # The oracle FUNCTION is checked against a constant, not just against itself.
    @test KhepriBase.path_length(circular_path(u0(), 3)) ≈ 2π*3            atol=MEASURE_ATOL
    @test KhepriBase.path_length(circular_path(u0(), 3)) ≈ 18.84955592153876 atol=1e-10
    @test KhepriBase.path_length(rectangular_path(xy(0, 0), 4, 5)) ≈ 18.0   atol=MEASURE_ATOL
    @test KhepriBase.path_length(line_path(xyz(0, 0, 0), xyz(3, 4, 0))) ≈ 5.0 atol=MEASURE_ATOL
    @test KhepriBase.path_length(arc_path(u0(), 2, 0, π/2)) ≈ π             atol=MEASURE_ATOL
  end

  @testset "Length is translation-invariant and scale-equivariant" begin
    paths = (circular_path(u0(), 3),
             rectangular_path(xy(0, 0), 4, 5),
             line_path(xyz(0, 0, 0), xyz(3, 4, 0)),
             arc_path(u0(), 2, 0, π/2),
             open_polygonal_path([xy(0, 0), xy(2, 0), xy(2, 3), xy(5, 3)]))
    for p in paths
      L = KhepriBase.path_length(p)
      # Translating a curve cannot change its length.
      @test KhepriBase.path_length(KhepriBase.translate(p, vxyz(10, -7, 4))) ≈ L atol=MEASURE_ATOL
      # Uniform scale about the origin multiplies length by k.
      for k in (0.5, 2.0, 10.0)
        @test KhepriBase.path_length(KhepriBase.scale(p, k)) ≈ k*L atol=MEASURE_ATOL*max(1, k)
      end
    end
  end

  @testset "AABB shifts by exactly v under translation" begin
    v = vxyz(3.5, -2.0, 1.25)
    paths = (circular_path(u0(), 3),
             rectangular_path(xy(0, 0), 4, 5),
             arc_path(u0(), 2, 0, π/2),
             open_polygonal_path([xy(0, 0), xy(2, 0), xy(2, 3), xy(5, 3)]))
    for p in paths
      base  = sampled_world_aabb(p)
      moved = sampled_world_aabb(KhepriBase.translate(p, v))
      @test aabb_isapprox(moved, aabb_shift(base, v))
    end
  end

  @testset "Signed area: winding, scale (k²), reflection, translation" begin
    ccw = [xy(0, 0), xy(4, 0), xy(4, 3), xy(0, 3)]   # CCW, area 12
    cw  = reverse(ccw)
    @test signed_area(ccw) ≈  12.0 atol=MEASURE_ATOL
    @test signed_area(cw)  ≈ -12.0 atol=MEASURE_ATOL
    # Scaling about the origin by k multiplies signed area by k² (sign kept).
    for k in (0.5, 2.0, 3.0)
      @test signed_area(scale_points(ccw, k)) ≈ k^2*12.0 atol=MEASURE_ATOL*max(1, k^2)
    end
    # Reflection flips handedness → sign negates, magnitude preserved.
    @test signed_area(reflect_x(ccw)) ≈ -12.0 atol=MEASURE_ATOL
    # Translation leaves signed area unchanged.
    @test signed_area([p + vxy(100, -50) for p in ccw]) ≈ 12.0 atol=MEASURE_ATOL
  end
end
