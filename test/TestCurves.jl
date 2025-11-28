using Test
using KhepriBase

@testset "Geometry Utilities" begin
  @testset "Incircle" begin
    p1 = xy(0, 0)
    p2 = xy(3, 0)
    p3 = xy(0, 4)
    (c, r) = incircle(p1, p2, p3)
    @test r ≈ 1.0
    @test c.x ≈ 1.0 && c.y ≈ 1.0
  end

  @testset "Fit Line" begin
    pts = [xy(0, 0), xy(1, 1), xy(2, 2), xy(3, 3)]
    (m, c) = fit_line(pts)
    @test m ≈ 1.0
    @test c ≈ 0.0
  end

  @testset "Fit Circle" begin
    pts = [xy(1, 0), xy(0, 1), xy(-1, 0), xy(0, -1)]
    (c, r) = fit_circle(pts)
    @test r ≈ 1.0
    @test distance(c, xy(0, 0)) < 1e-9
  end

  @testset "Is Coplanar" begin
    pts = [xy(0, 0), xy(1, 0), xy(0, 1), xy(1, 1)]
    @test is_coplanar(pts)
    pts = [xyz(0, 0, 0), xyz(1, 0, 0), xyz(0, 1, 0), xyz(0, 0, 1)]
    @test !is_coplanar(pts)
    # Collinear start case
    pts = [xyz(0, 0, 0), xyz(1, 0, 0), xyz(2, 0, 0), xyz(0, 1, 0)]
    @test is_coplanar(pts)
    pts = [xyz(0, 0, 0), xyz(1, 0, 0), xyz(2, 0, 0), xyz(0, 0, 1)]
    @test is_coplanar(pts) # Technically 4 points with 3 collinear define a plane or are coplanar if the 4th is on the line.
    # Wait, if first 3 are collinear, they define a line. Any 4th point forms a plane with that line.
    # So 4 points where 3 are collinear are always coplanar.
    # Let's try 5 points where first 3 are collinear, 4th defines plane, 5th is outside.
    pts = [xyz(0, 0, 0), xyz(1, 0, 0), xyz(2, 0, 0), xyz(0, 1, 0), xyz(0, 0, 1)]
    @test !is_coplanar(pts)
  end

  @testset "Polygon Centroid" begin
    pts = [xy(0, 0), xy(4, 0), xy(4, 3), xy(0, 3)]
    c = polygon_centroid(pts)
    @test c.x ≈ 2.0
    @test c.y ≈ 1.5
  end
end

@testset "Path Analysis" begin
  @testset "Tangent" begin
    path = circular_path(xy(0, 0), 1)
    t = path_tangent(path, 0)
    @test t.x ≈ 0.0
    @test t.y ≈ 1.0 # Counter-clockwise tangent at (1,0) is (0,1)
  end

  @testset "Curvature" begin
    path = circular_path(xy(0, 0), 2)
    k = path_curvature(path, 0)
    @test k ≈ 0.5 # 1/r
  end

  @testset "Point In Path" begin
    path = rectangular_path(xy(0, 0), 2, 2)
    @test point_in_path(xy(1, 1), path)
    @test !point_in_path(xy(3, 3), path)
  end

  @testset "Closest Point" begin
    path = line(xy(0, 0), xy(10, 0))
    p = closest_point_at_path(xy(5, 5), path)
    @test p.x ≈ 5.0
    @test p.y ≈ 0.0
  end
end

@testset "Path Construction" begin
  @testset "Arc 3Pt" begin
    path = arc_3pt(xy(1, 0), xy(0, 1), xy(-1, 0))
    @test path_length(path) ≈ pi
    # CW Arc
    path_cw = arc_3pt(xy(1, 0), xy(0, -1), xy(-1, 0))
    @test path_length(path_cw) ≈ pi
  end

  @testset "Rectangle 3Pt" begin
    path = rectangle_3pt(xy(0, 0), xy(2, 0), xy(2, 1))
    @test path_length(path) ≈ 6
  end

  @testset "Tween Path" begin
    p1 = line(xy(0, 0), xy(10, 0))
    p2 = line(xy(0, 10), xy(10, 10))
    p_mid = tween_path(p1, p2, 0.5)
    center = path_centroid(p_mid)
    @test center.y ≈ 5.0
  end
end

@testset "Path Division" begin
  @testset "Shatter" begin
    path = line(xy(0, 0), xy(10, 0))
    parts = shatter(path, [2.0, 5.0])
    @test length(parts) == 3
    @test path_length(parts[1]) ≈ 2.0
    @test path_length(parts[2]) ≈ 3.0
    @test path_length(parts[3]) ≈ 5.0
  end

  @testset "Divide Length" begin
    path = line(xy(0, 0), xy(10, 0))
    pts = divide_path_by_length(path, 2.0)
    @test length(pts) == 5
    @test pts[1].x ≈ 2.0
  end
end
