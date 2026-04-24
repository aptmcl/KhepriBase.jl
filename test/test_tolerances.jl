# Tests for the <property>_tolerance family defined across Coords.jl,
# Paths.jl, Geometry.jl, and BIM.jl. Each block exercises the round-trip
# of the Parameter and the classifier flip it controls. See
# docs/src/concepts/parameters.md for the full naming convention.

using KhepriBase
using Test

@testset "Geometric tolerances" begin

  @testset "coincidence_tolerance — Paths.jl:75" begin
    @test coincidence_tolerance() == 1e-10
    with(coincidence_tolerance, 1e-6) do
      @test coincidence_tolerance() == 1e-6
      @test coincident_path_location(xy(0, 0), xy(1e-8, 1e-8))
    end
    @test coincidence_tolerance() == 1e-10  # restored
    @test !coincident_path_location(xy(0, 0), xy(1e-8, 1e-8))
  end

  @testset "collinearity_tolerance — Geometry.jl:214" begin
    @test collinearity_tolerance() == 1e-2
    # A mild bend (area ~2.5e-3) is collinear at the default 1e-2,
    # but not at a tightened 1e-4.
    p0, pm, p1 = xy(0, 0), xy(0.5, 0.005), xy(1, 0)
    @test KhepriBase.collinear_points(p0, pm, p1)
    with(collinearity_tolerance, 1e-4) do
      @test !KhepriBase.collinear_points(p0, pm, p1)
    end
    @test collinearity_tolerance() == 1e-2  # restored
  end

  @testset "planarity_tolerance — Paths.jl:810" begin
    @test planarity_tolerance() == 1e-6
    with(planarity_tolerance, 1e-3) do
      @test planarity_tolerance() == 1e-3
    end
    @test planarity_tolerance() == 1e-6
  end

  @testset "parallelism_tolerance — Geometry.jl:150" begin
    @test parallelism_tolerance() == 1e-8
    with(parallelism_tolerance, 1e-6) do
      @test parallelism_tolerance() == 1e-6
    end
    @test parallelism_tolerance() == 1e-8
  end

  @testset "zero_vector_tolerance — Coords.jl:555" begin
    @test zero_vector_tolerance() == 1e-20
    # A vector whose norm is far above the tolerance unitizes cleanly.
    @test norm(unitized(vxyz(3.0, 4.0, 0.0, world_cs))) ≈ 1.0
    # Below the tolerance we throw DomainError rather than producing
    # a meaningless unit vector.
    @test_throws DomainError unitized(vxyz(1e-25, 0.0, 0.0, world_cs))
  end

  @testset "truss_node_coincidence_tolerance — BIM.jl:1214" begin
    @test truss_node_coincidence_tolerance() == 1e-6
    with(truss_node_coincidence_tolerance, 1e-3) do
      @test truss_node_coincidence_tolerance() == 1e-3
    end
    @test truss_node_coincidence_tolerance() == 1e-6
  end

end
