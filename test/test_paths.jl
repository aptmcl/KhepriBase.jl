# test_paths.jl - Comprehensive tests for path operations

using Test
using KhepriBase

@testset "Paths" begin

  @testset "Path construction" begin
    @testset "EmptyPath" begin
      p = empty_path()
      @test KhepriBase.is_empty_path(p)
    end

    @testset "PointPath" begin
      p = point_path(xyz(1, 2, 3))
      @test p.location.x == 1
      @test p.location.y == 2
      @test p.location.z == 3
    end

    @testset "CircularPath" begin
      p = circular_path(u0(), 5)
      @test p.center.x == 0
      @test p.radius == 5
      @test is_closed_path(p)

      # Keyword construction
      p2 = circular_path(center=xy(1, 2), radius=3)
      @test p2.center.x == 1
      @test p2.radius == 3
    end

    @testset "ArcPath" begin
      p = arc_path(u0(), 5, 0, π/2)
      @test p.center.x == 0
      @test p.radius == 5
      @test p.start_angle == 0
      @test p.amplitude ≈ π/2 atol=1e-10
      @test !is_closed_path(p)
    end

    @testset "EllipticPath" begin
      p = elliptic_path(u0(), 10, 5)
      @test p.center.x == 0
      @test p.r1 == 10
      @test p.r2 == 5
      @test is_closed_path(p)
    end

    @testset "RectangularPath" begin
      p = rectangular_path(u0(), 10, 5)
      @test p.corner.x == 0
      @test p.dx == 10
      @test p.dy == 5
      @test is_closed_path(p)

      # Centered rectangular path
      p2 = centered_rectangular_path(xy(5, 5), 10, 10)
      @test p2.corner.x == 0
      @test p2.corner.y == 0
    end

    @testset "OpenPolygonalPath" begin
      verts = [xy(0, 0), xy(1, 0), xy(1, 1)]
      p = open_polygonal_path(verts)
      @test length(p.vertices) == 3
      @test !is_closed_path(p)
    end

    @testset "ClosedPolygonalPath" begin
      verts = [xy(0, 0), xy(1, 0), xy(1, 1), xy(0, 1)]
      p = closed_polygonal_path(verts)
      @test length(p.vertices) == 4
      @test is_closed_path(p)
    end

    @testset "PolygonalPath auto-detection" begin
      # Open path (different start/end)
      p1 = polygonal_path([xy(0, 0), xy(1, 0), xy(1, 1)])
      @test !is_closed_path(p1)

      # Closed path (same start/end)
      p2 = polygonal_path([xy(0, 0), xy(1, 0), xy(1, 1), xy(0, 0)])
      @test is_closed_path(p2)
      @test length(p2.vertices) == 3  # Last duplicate removed
    end

    @testset "OpenSplinePath" begin
      verts = [xy(0, 0), xy(1, 0.5), xy(2, 0), xy(3, 0.5)]
      p = open_spline_path(verts)
      @test length(p.vertices) == 4
      @test !is_closed_path(p)
      @test is_smooth_path(p)
    end

    @testset "ClosedSplinePath" begin
      verts = [xy(0, 0), xy(1, 0), xy(1, 1), xy(0, 1)]
      p = closed_spline_path(verts)
      @test length(p.vertices) == 4
      @test is_closed_path(p)
      @test is_smooth_path(p)
    end
  end

  @testset "path_length" begin
    @testset "CircularPath" begin
      p = circular_path(u0(), 1)
      @test path_length(p) ≈ 2π atol=1e-10

      p2 = circular_path(u0(), 5)
      @test path_length(p2) ≈ 10π atol=1e-10
    end

    @testset "ArcPath" begin
      p = arc_path(u0(), 1, 0, π)
      @test path_length(p) ≈ π atol=1e-10

      p2 = arc_path(u0(), 2, 0, π/2)
      @test path_length(p2) ≈ π atol=1e-10
    end

    @testset "RectangularPath" begin
      p = rectangular_path(u0(), 10, 5)
      @test path_length(p) ≈ 30 atol=1e-10  # 2*(10+5)
    end

    @testset "OpenPolygonalPath" begin
      p = open_polygonal_path([xy(0, 0), xy(3, 0), xy(3, 4)])
      @test path_length(p) ≈ 7 atol=1e-10  # 3 + 4
    end

    @testset "ClosedPolygonalPath" begin
      # Square
      p = closed_polygonal_path([xy(0, 0), xy(1, 0), xy(1, 1), xy(0, 1)])
      @test path_length(p) ≈ 4 atol=1e-10
    end
  end

  @testset "path_domain" begin
    @testset "CircularPath domain" begin
      p = circular_path()
      @test path_domain(p) == (0, 2π)
    end

    @testset "ArcPath domain" begin
      p = arc_path(u0(), 1, 0, π)
      @test path_domain(p) == (0, π)
    end

    @testset "PolygonalPath domain" begin
      p = polygonal_path([x(0), x(10)])
      @test path_domain(p) == (0, 10)
    end
  end

  @testset "location_at" begin
    # Note: location_at returns locations with local coordinate systems
    # Use in_world() to get world coordinates for comparison

    @testset "CircularPath" begin
      p = circular_path(u0(), 1)
      loc0 = in_world(location_at(p, 0))
      @test loc0.x ≈ 1 atol=1e-10
      @test loc0.y ≈ 0 atol=1e-10

      loc_half = in_world(location_at(p, π))
      @test loc_half.x ≈ -1 atol=1e-10
      @test loc_half.y ≈ 0 atol=1e-10

      loc_quarter = in_world(location_at(p, π/2))
      @test loc_quarter.x ≈ 0 atol=1e-10
      @test loc_quarter.y ≈ 1 atol=1e-10
    end

    @testset "ArcPath" begin
      p = arc_path(u0(), 1, 0, π)
      loc0 = in_world(location_at(p, 0))
      @test loc0.x ≈ 1 atol=1e-10
      @test loc0.y ≈ 0 atol=1e-10

      loc_end = in_world(location_at(p, π))
      @test loc_end.x ≈ -1 atol=1e-10
      @test loc_end.y ≈ 0 atol=1e-10
    end

    @testset "EllipticPath" begin
      p = elliptic_path(u0(), 2, 1)
      loc0 = in_world(location_at(p, 0))
      @test loc0.x ≈ 2 atol=1e-10
      @test loc0.y ≈ 0 atol=1e-10

      loc_quarter = in_world(location_at(p, π/2))
      @test loc_quarter.x ≈ 0 atol=1e-10
      @test loc_quarter.y ≈ 1 atol=1e-10
    end
  end

  @testset "location_at_length" begin
    @testset "CircularPath" begin
      p = circular_path(u0(), 1)
      loc0 = in_world(location_at_length(p, 0))
      @test loc0.x ≈ 1 atol=1e-10
      @test loc0.y ≈ 0 atol=1e-10

      # Half the circumference
      loc_half = in_world(location_at_length(p, π))
      @test loc_half.x ≈ -1 atol=1e-10
      @test loc_half.y ≈ 0 atol=1e-10
    end

    @testset "RectangularPath" begin
      p = rectangular_path(u0(), 4, 3)
      loc0 = in_world(location_at_length(p, 0))
      @test loc0.x ≈ 0 atol=1e-10
      @test loc0.y ≈ 0 atol=1e-10

      # After first side
      loc4 = in_world(location_at_length(p, 4))
      @test loc4.x ≈ 4 atol=1e-10
      @test loc4.y ≈ 0 atol=1e-10

      # Middle of second side
      loc5_5 = in_world(location_at_length(p, 5.5))
      @test loc5_5.x ≈ 4 atol=1e-10
      @test loc5_5.y ≈ 1.5 atol=1e-10
    end

    @testset "OpenPolygonalPath" begin
      p = open_polygonal_path([xy(0, 0), xy(3, 0), xy(3, 4)])
      loc0 = in_world(location_at_length(p, 0))
      @test loc0.x ≈ 0 atol=1e-10
      @test loc0.y ≈ 0 atol=1e-10

      # At first corner
      loc3 = in_world(location_at_length(p, 3))
      @test loc3.x ≈ 3 atol=1e-10
      @test loc3.y ≈ 0 atol=1e-10

      # At end
      loc7 = in_world(location_at_length(p, 7))
      @test loc7.x ≈ 3 atol=1e-10
      @test loc7.y ≈ 4 atol=1e-10
    end
  end

  @testset "path_start / path_end" begin
    @testset "PolygonalPath" begin
      verts = [xy(0, 0), xy(1, 0), xy(1, 1)]
      p = open_polygonal_path(verts)
      @test path_start(p).x ≈ 0 atol=1e-10
      @test path_end(p).x ≈ 1 atol=1e-10
      @test path_end(p).y ≈ 1 atol=1e-10
    end

    @testset "ClosedPath" begin
      p = closed_polygonal_path([xy(0, 0), xy(1, 0), xy(1, 1)])
      @test path_start(p).x ≈ 0 atol=1e-10
      @test path_end(p).x ≈ 0 atol=1e-10  # Same as start for closed
    end
  end

  @testset "subpath" begin
    @testset "CircularPath subpath" begin
      p = circular_path(u0(), 1)
      sub = subpath(p, 0, π)  # First half
      @test sub isa ArcPath
      @test sub.amplitude ≈ π atol=1e-10
    end

    @testset "ArcPath subpath" begin
      p = arc_path(u0(), 1, 0, π)
      sub = subpath(p, 0, π/2)  # First quarter
      @test sub.amplitude ≈ π/2 atol=1e-10
    end
  end

  @testset "path translation" begin
    @testset "translate CircularPath" begin
      p = circular_path(u0(), 5)
      p2 = translate(p, vxyz(10, 20, 30))
      @test p2.center.x ≈ 10 atol=1e-10
      @test p2.center.y ≈ 20 atol=1e-10
      @test p2.center.z ≈ 30 atol=1e-10
      @test p2.radius == 5
    end

    @testset "translate ArcPath" begin
      p = arc_path(u0(), 5, 0, π)
      p2 = translate(p, vxyz(10, 0, 0))
      @test p2.center.x ≈ 10 atol=1e-10
      @test p2.radius == 5
      @test p2.amplitude ≈ π atol=1e-10
    end

    @testset "translate RectangularPath" begin
      p = rectangular_path(u0(), 10, 5)
      p2 = translate(p, vxyz(5, 5, 0))
      @test p2.corner.x ≈ 5 atol=1e-10
      @test p2.corner.y ≈ 5 atol=1e-10
      @test p2.dx == 10
      @test p2.dy == 5
    end

    @testset "translate PolygonalPath" begin
      p = open_polygonal_path([xy(0, 0), xy(1, 0), xy(1, 1)])
      p2 = translate(p, vxyz(10, 10, 0))
      @test p2.vertices[1].x ≈ 10 atol=1e-10
      @test p2.vertices[1].y ≈ 10 atol=1e-10
    end
  end

  @testset "join_paths" begin
    p1 = open_polygonal_path([xy(0, 0), xy(1, 0)])
    p2 = open_polygonal_path([xy(1, 0), xy(1, 1)])
    joined = join_paths(p1, p2)
    @test length(joined.vertices) == 3
    @test joined.vertices[1].x ≈ 0 atol=1e-10
    @test joined.vertices[end].y ≈ 1 atol=1e-10
  end

  @testset "path_vertices" begin
    @testset "PolygonalPath vertices" begin
      verts = [xy(0, 0), xy(1, 0), xy(1, 1)]
      p = open_polygonal_path(verts)
      @test path_vertices(p) == verts
    end

    @testset "CircularPath vertices" begin
      p = circular_path(u0(), 1)
      verts = path_vertices(p)
      @test length(verts) > 3  # Should have multiple interpolated vertices
    end

    @testset "RectangularPath vertices" begin
      p = rectangular_path(u0(), 4, 3)
      verts = path_vertices(p)
      @test length(verts) == 4
      @test verts[1].x ≈ 0 atol=1e-10
      @test verts[2].x ≈ 4 atol=1e-10
    end
  end

  @testset "is_smooth_path" begin
    @test !is_smooth_path(open_polygonal_path([xy(0, 0), xy(1, 0)]))
    @test !is_smooth_path(closed_polygonal_path([xy(0, 0), xy(1, 0), xy(1, 1)]))
    @test is_smooth_path(circular_path())
    @test is_smooth_path(arc_path())
    @test is_smooth_path(open_spline_path([xy(0, 0), xy(1, 0.5), xy(2, 0)]))
    @test is_smooth_path(closed_spline_path([xy(0, 0), xy(1, 0), xy(1, 1), xy(0, 1)]))
  end

  @testset "coincident_path_location" begin
    @test coincident_path_location(xy(0, 0), xy(0, 0))
    @test coincident_path_location(pol(1, π/2), xy(0, 1))
    @test !coincident_path_location(xy(0, 0), xy(1, 1))

    # Test with different tolerances
    with(path_tolerance, 1e-5) do
      @test coincident_path_location(xy(0, 0), xy(1e-6, 1e-6))
    end
    with(path_tolerance, 1e-10) do
      @test !coincident_path_location(xy(0, 0), xy(1e-6, 1e-6))
    end
  end

  @testset "planar_path_normal" begin
    @testset "CircularPath normal" begin
      p = circular_path(u0(), 1)
      n = planar_path_normal(p)
      @test abs(n.z) ≈ 1 atol=1e-10
    end

    @testset "RectangularPath normal" begin
      p = rectangular_path(u0(), 1, 1)
      n = planar_path_normal(p)
      @test abs(n.z) ≈ 1 atol=1e-10
    end
  end

  @testset "PathOps" begin
    @testset "LineOp" begin
      p = open_path_ops(u0(), LineOp(vxy(5, 0)))
      @test path_length(p) ≈ 5 atol=1e-10

      loc = location_at_length(p, 2.5)
      @test loc.x ≈ 2.5 atol=1e-10
    end

    @testset "ArcOp" begin
      p = open_path_ops(u0(), ArcOp(1, 0, π/2))
      @test path_length(p) ≈ π/2 atol=1e-10
    end

    @testset "Combined path ops" begin
      p = open_path_ops(u0(), LineOp(vxy(4, 0)), ArcOp(1, 0, π/2))
      total = path_length(p)
      @test total ≈ 4 + π/2 atol=1e-10
    end
  end

  @testset "PathSequence" begin
    @testset "OpenPathSequence" begin
      p1 = open_polygonal_path([xy(0, 0), xy(1, 0)])
      p2 = open_polygonal_path([xy(1, 0), xy(1, 1)])
      seq = open_path_sequence(p1, p2)
      @test !is_closed_path(seq)
      @test path_length(seq) ≈ 2 atol=1e-10
    end

    @testset "ClosedPathSequence" begin
      p1 = open_polygonal_path([xy(0, 0), xy(1, 0)])
      p2 = open_polygonal_path([xy(1, 0), xy(1, 1)])
      p3 = open_polygonal_path([xy(1, 1), xy(0, 0)])
      seq = closed_path_sequence(p1, p2, p3)
      @test is_closed_path(seq)
    end
  end

  @testset "PathSet" begin
    p1 = circular_path(u0(), 1)
    p2 = circular_path(xy(10, 0), 1)
    pset = path_set(p1, p2)
    @test length(pset.paths) == 2
  end

  @testset "Region" begin
    @testset "Region construction" begin
      outer = circular_path(u0(), 10)
      inner = circular_path(u0(), 3)
      r = region(outer, inner)
      @test outer_path(r) === outer
      @test length(inner_paths(r)) == 1
    end

    @testset "Region from closed path" begin
      p = closed_polygonal_path([xy(0, 0), xy(10, 0), xy(10, 10), xy(0, 10)])
      r = region(p)
      @test outer_path(r).vertices[1].x ≈ 0 atol=1e-10
    end
  end

  @testset "Profiles" begin
    @testset "rectangular_profile" begin
      p = rectangular_profile(0.5, 0.3)
      @test p.dx ≈ 0.5 atol=1e-10
      @test p.dy ≈ 0.3 atol=1e-10
    end

    @testset "circular_profile" begin
      p = circular_profile(0.5)
      @test p.radius ≈ 0.5 atol=1e-10
    end

    @testset "top_aligned_rectangular_profile" begin
      p = top_aligned_rectangular_profile(0.4, 0.6)
      @test p.corner.y ≈ -0.6 atol=1e-10
    end

    @testset "bottom_aligned_rectangular_profile" begin
      p = bottom_aligned_rectangular_profile(0.4, 0.6)
      @test p.corner.y ≈ 0 atol=1e-10
    end
  end

  @testset "Path scaling" begin
    @testset "scale CircularPath" begin
      p = circular_path(xyz(5, 5, 0), 2)
      p2 = scale(p, 2, u0())
      @test p2.radius ≈ 4 atol=1e-10
      @test p2.center.x ≈ 10 atol=1e-10
      @test p2.center.y ≈ 10 atol=1e-10
    end

    @testset "scale RectangularPath" begin
      p = rectangular_path(xy(1, 1), 4, 2)
      p2 = scale(p, 2, u0())
      @test p2.dx ≈ 8 atol=1e-10
      @test p2.dy ≈ 4 atol=1e-10
    end
  end

  @testset "Path reversal" begin
    @testset "reverse OpenPolygonalPath" begin
      p = open_polygonal_path([xy(0, 0), xy(1, 0), xy(2, 0)])
      pr = reverse(p)
      @test pr.vertices[1].x ≈ 2 atol=1e-10
      @test pr.vertices[end].x ≈ 0 atol=1e-10
    end

    @testset "reverse ClosedPolygonalPath" begin
      p = closed_polygonal_path([xy(0, 0), xy(1, 0), xy(1, 1), xy(0, 1)])
      pr = reverse(p)
      @test pr.vertices[1].y ≈ 1 atol=1e-10
    end
  end

  @testset "mirrored_on_* operations" begin
    @testset "mirrored_on_x" begin
      p = open_polygonal_path([xy(0, 0), xy(1, 1)])
      pm = mirrored_on_x(p)
      # Should create a path from the original plus its x-mirrored version
      @test is_closed_path(pm) || length(pm.vertices) > 2
    end
  end

  @testset "Mesh" begin
    verts = [xyz(0, 0, 0), xyz(1, 0, 0), xyz(0, 1, 0)]
    faces = [[0, 1, 2]]
    m = mesh(verts, faces)
    @test length(m.vertices) == 3
    @test length(m.faces) == 1
  end

  @testset "length_at_location" begin
    p = open_polygonal_path([xy(0, 0), xy(10, 0)])
    loc = xy(5, 0)
    len = length_at_location(p, loc)
    @test len ≈ 5 atol=1e-1  # Approximate since it's iterative
  end

end
