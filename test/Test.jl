using Test
using KhepriBase

@testset "Coordinate conversions" begin
  @test pol(1, 0).x == 1
  @test pol(1, 0).y == 0
  @test pol(1, π/2).x ≈ 0 atol=1e-9
  @test pol(1, π/2).y == 1
  @test pol(1, π).x == -1
  @test pol(1, π).y == 0
  @test pol(1, 3π/2).x ≈ 0 atol=1e-9
  @test pol(1, 3π/2).y == -1
end

@testset "Paths" begin
  @test coincident_path_location(pol(1, π/2), xy(0,1))
  with(path_tolerance, 1e-15) do
    @test coincident_path_location(pol(1, π/2), xy(0,1))    
  end
  with(path_tolerance, 1e-20) do
    @test ! coincident_path_location(pol(1, π/2), xy(0,1))    
  end

  @test is_closed_path(circular_path())

  @test path_domain(circular_path()) == (0, 2π)
  @test path_domain(arc_path()) == (0, π*1) # The nasty π problem.
  @test path_domain(polygonal_path(x(0), x(2))) == (0, 2)
  @test path_domain(polygonal_path(x(0), x(2), xy(2, 3))) == (0, 5)
end



