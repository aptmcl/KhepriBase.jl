# test_geometry.jl - Tests for geometric utility functions

using Test
using KhepriBase

@testset "Geometry" begin

  @testset "Area calculations" begin
    @testset "triangle_area" begin
      # Equilateral triangle with side 2
      @test triangle_area(2, 2, 2) ≈ sqrt(3) atol=1e-10

      # 3-4-5 right triangle
      @test triangle_area(3, 4, 5) ≈ 6 atol=1e-10

      # Degenerate triangle (collinear points)
      @test triangle_area(1, 2, 3) ≈ 0 atol=1e-10

      # Isoceles triangle
      @test triangle_area(5, 5, 6) ≈ 12 atol=1e-10
    end

    @testset "circle_area" begin
      @test circle_area(1) ≈ π atol=1e-10
      @test circle_area(2) ≈ 4π atol=1e-10
      @test circle_area(0.5) ≈ π/4 atol=1e-10
    end

    @testset "annulus_area" begin
      # Annulus with outer radius 2 and inner radius 1
      @test annulus_area(2, 1) ≈ 3π atol=1e-10

      # Annulus where inner = 0 should equal circle area
      @test annulus_area(3, 0) ≈ circle_area(3) atol=1e-10
    end
  end

  @testset "Offset operations" begin
    @testset "offset_vertices open path" begin
      pts = [xy(0, 0), xy(10, 0), xy(10, 10)]
      offset_pts = KhepriBase.offset_vertices(pts, 1, false)
      @test length(offset_pts) == 3
      # Offset should push points outward
      @test offset_pts[1].y > 0  # First point moved up
      @test offset_pts[3].x < 10  # Last point moved left
    end

    @testset "offset_vertices closed path" begin
      pts = [xy(0, 0), xy(10, 0), xy(10, 10), xy(0, 10)]
      offset_pts = KhepriBase.offset_vertices(pts, 1, true)
      @test length(offset_pts) == 4
    end

    @testset "offset RectangularPath" begin
      p = rectangular_path(xy(0, 0), 10, 10)
      p_offset = offset(p, 1)
      @test p_offset.corner.x ≈ 1 atol=1e-10
      @test p_offset.corner.y ≈ 1 atol=1e-10
      @test p_offset.dx ≈ 8 atol=1e-10
      @test p_offset.dy ≈ 8 atol=1e-10
    end

    @testset "offset CircularPath" begin
      p = circular_path(u0(), 10)
      p_offset = offset(p, 2)
      @test p_offset.radius ≈ 8 atol=1e-10
    end

    @testset "offset ArcPath" begin
      p = arc_path(u0(), 10, 0, π)
      p_offset = offset(p, 2)
      @test p_offset.radius ≈ 8 atol=1e-10
      @test p_offset.amplitude ≈ π atol=1e-10
    end

    @testset "offset OpenPolygonalPath" begin
      p = open_polygonal_path([xy(0, 0), xy(10, 0)])
      p_offset = offset(p, 1)
      # For a horizontal line, offset should move it vertically
      @test p_offset.vertices[1].y ≈ 1 atol=1e-10
    end

    @testset "offset ClosedPolygonalPath" begin
      p = closed_polygonal_path([xy(0, 0), xy(10, 0), xy(10, 10), xy(0, 10)])
      p_offset = offset(p, 1)
      @test length(p_offset.vertices) == 4
    end

    @testset "offset zero returns same path" begin
      p = circular_path(u0(), 5)
      p_offset = offset(p, 0)
      @test p_offset === p
    end
  end

  @testset "Segment intersection" begin
    @testset "segments_intersection - crossing" begin
      # Two segments that cross
      p0, p1 = xy(0, 0), xy(10, 10)
      p2, p3 = xy(0, 10), xy(10, 0)
      result = KhepriBase.segments_intersection(p0, p1, p2, p3)
      @test result !== nothing
      @test result.x ≈ 5 atol=1e-10
      @test result.y ≈ 5 atol=1e-10
    end

    @testset "segments_intersection - parallel" begin
      # Two parallel segments (no intersection)
      p0, p1 = xy(0, 0), xy(10, 0)
      p2, p3 = xy(0, 5), xy(10, 5)
      result = KhepriBase.segments_intersection(p0, p1, p2, p3)
      @test result === nothing
    end

    @testset "segments_intersection - non-crossing" begin
      # Two segments that would cross if extended but don't actually cross
      p0, p1 = xy(0, 0), xy(1, 1)
      p2, p3 = xy(5, 0), xy(5, 5)
      result = KhepriBase.segments_intersection(p0, p1, p2, p3)
      @test result === nothing
    end

    @testset "lines_intersection - crossing" begin
      # Two lines (infinite) that cross
      p0, p1 = xy(0, 0), xy(10, 10)
      p2, p3 = xy(0, 10), xy(10, 0)
      result = KhepriBase.lines_intersection(p0, p1, p2, p3)
      @test result !== nothing
      @test result.x ≈ 5 atol=1e-10
      @test result.y ≈ 5 atol=1e-10
    end

    @testset "lines_intersection - parallel" begin
      # Two parallel lines (no intersection)
      p0, p1 = xy(0, 0), xy(10, 0)
      p2, p3 = xy(0, 5), xy(10, 5)
      result = KhepriBase.lines_intersection(p0, p1, p2, p3)
      @test result === nothing
    end
  end

  @testset "circle_from_three_points" begin
    @testset "unit circle" begin
      p0 = xy(1, 0)
      p1 = xy(0, 1)
      p2 = xy(-1, 0)
      (center, radius) = circle_from_three_points(p0, p1, p2)
      # circle_from_three_points returns center in a local CS
      # Use in_world to get world coordinates
      cw = in_world(center)
      @test cw.x ≈ 0 atol=1e-10
      @test cw.y ≈ 0 atol=1e-10
      @test radius ≈ 1 atol=1e-10
    end

    @testset "offset circle" begin
      # Circle centered at (5,5) with radius 2
      p0 = xy(7, 5)  # Right
      p1 = xy(5, 7)  # Top
      p2 = xy(3, 5)  # Left
      (center, radius) = circle_from_three_points(p0, p1, p2)
      cw = in_world(center)
      @test cw.x ≈ 5 atol=1e-10
      @test cw.y ≈ 5 atol=1e-10
      @test radius ≈ 2 atol=1e-10
    end
  end

  @testset "nearest_point_from_lines" begin
    # Two skew lines in 3D
    l0p0 = xyz(0, 0, 0)
    l0p1 = xyz(1, 0, 0)
    l1p0 = xyz(0, 1, 1)
    l1p1 = xyz(0, 1, 2)
    result = nearest_point_from_lines(l0p0, l0p1, l1p0, l1p1)
    @test result.y ≈ 0.5 atol=1e-10
  end

  @testset "Collinearity" begin
    @testset "collinear_points" begin
      # Three collinear points
      p0 = xy(0, 0)
      p1 = xy(5, 5)
      p2 = xy(10, 10)
      @test KhepriBase.collinear_points(p0, p1, p2)

      # Three non-collinear points
      p3 = xy(10, 0)
      @test !KhepriBase.collinear_points(p0, p1, p3)

      # Nearly collinear points (within tolerance)
      p4 = xy(5, 5.001)
      @test KhepriBase.collinear_points(p0, p4, p2, 0.01)
    end
  end

  @testset "Polygon operations" begin
    @testset "closest_vertices_indexes" begin
      pts1 = [xy(0, 0), xy(1, 0), xy(2, 0)]
      pts2 = [xy(10, 10), xy(10, 11), xy(1.1, 0.1)]  # Last point is closest
      (i, j) = closest_vertices_indexes(pts1, pts2)
      @test i == 2  # Second point in pts1
      @test j == 3  # Third point in pts2
    end

    @testset "subtract_polygon_vertices" begin
      # Outer square
      outer = [xy(0, 0), xy(10, 0), xy(10, 10), xy(0, 10)]
      # Inner square (hole)
      inner = [xy(3, 3), xy(7, 3), xy(7, 7), xy(3, 7)]
      result = subtract_polygon_vertices(outer, inner)
      # Result should have more vertices (to connect outer to inner)
      @test length(result) > length(outer) + length(inner)
    end
  end

  @testset "quad_grid operations" begin
    @testset "quad_grid" begin
      points = [xy(i, j) for i in 0:2, j in 0:2]
      count = Ref(0)
      quad_grid((p0, p1, p2, p3) -> count[] += 1, points, false, false)
      @test count[] == 4  # 2x2 grid of quads
    end

    @testset "quad_grid_indexes" begin
      idxs = quad_grid_indexes(3, 3, false, false)
      @test length(idxs) == 8  # 2 triangles per quad, 4 quads
      # Each index should be a 3-element array (triangle)
      @test all(length(idx) == 3 for idx in idxs)
    end

    @testset "quad_grid_indexes closed" begin
      idxs_closed_u = quad_grid_indexes(3, 3, true, false)
      @test length(idxs_closed_u) > length(quad_grid_indexes(3, 3, false, false))
    end
  end

  @testset "Vector helpers" begin
    @testset "v_in_v" begin
      v0 = vxy(1, 0)
      v1 = vxy(0, 1)
      result = KhepriBase.v_in_v(v0, v1)
      # Result should be a vector that bisects the angle
      @test result isa Vec
    end

    @testset "rotated_v" begin
      v = vxy(1, 0)
      rotated = KhepriBase.rotated_v(v, π/2)
      @test rotated.x ≈ 0 atol=1e-10
      @test rotated.y ≈ 1 atol=1e-10
    end
  end

  @testset "centered_rectangle" begin
    p0 = xy(0, 0)
    p1 = xy(10, 0)
    rect = KhepriBase.centered_rectangle(p0, 2, p1)
    @test rect isa Rectangle
    @test rect.dy ≈ 2 atol=1e-10
  end

  @testset "epsilon parameter" begin
    @test epsilon() == 1e-8
    with(epsilon, 1e-6) do
      @test epsilon() == 1e-6
    end
    @test epsilon() == 1e-8  # Restored
  end

  @testset "collinearity_tolerance parameter" begin
    @test KhepriBase.collinearity_tolerance() == 1e-2
    with(KhepriBase.collinearity_tolerance, 1e-4) do
      @test KhepriBase.collinearity_tolerance() == 1e-4
    end
  end

end
