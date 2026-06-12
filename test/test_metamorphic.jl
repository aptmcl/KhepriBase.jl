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

  @testset "Intersection classification is scale-invariant" begin
    # Scaling a configuration by k must scale intersection points by k and
    # leave point counts and kinds unchanged: the kernels classify through
    # the dimensionless sin²θ parallelism test, metre-space root deduping,
    # and radius-derived angular slack, none of which depends on absolute
    # scale. (The old raw thresholds returned EMPTY for mm-scale crossings.)
    for k in (1e-3, 1.0, 1e3)
      # Perpendicular crossing segments: one transversal point at k·(1, 0).
      let r = intersections(line_path(xy(0, 0), xy(2k, 0)),
                            line_path(xy(k, -k), xy(k, k)))
        @test length(r) == 1
        @test r[1].kind == :transversal
        @test r[1].point.x ≈ k atol=MEASURE_ATOL*max(1, k)
        @test r[1].point.y ≈ 0 atol=MEASURE_ATOL*max(1, k)
      end
      # Tangent line/circle: exactly one point of kind :tangent at any scale.
      let hits = intersections(line_path(xy(-2k, k), xy(2k, k)),
                               circular_path(xy(0, 0), k))
        @test length(hits) == 1
        @test hits[1].kind == :tangent
        @test hits[1].point.y ≈ k atol=MEASURE_ATOL*max(1, k)
      end
      # Line through an arc endpoint: the endpoint hit survives the
      # metre→radian conversion of the angular slack at any radius.
      let hits = intersections(line_path(xy(0, k/2), xy(0, 3k/2)),
                               arc_path(u0(), k, 0, π/2))
        @test length(hits) == 1
        @test hits[1].point.y ≈ k atol=MEASURE_ATOL*max(1, k)
      end
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
