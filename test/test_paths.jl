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
      @test p isa KhepriBase.ContourPath{false}
    end

    @testset "ClosedPolygonalPath" begin
      verts = [xy(0, 0), xy(1, 0), xy(1, 1), xy(0, 1)]
      p = closed_polygonal_path(verts)
      @test length(p.vertices) == 4
      @test is_closed_path(p)
      @test p isa KhepriBase.ContourPath{true}
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

    @testset "Spline hierarchy" begin
      p = open_spline_path([xy(0, 0), xy(1, 0.5), xy(2, 0)])
      @test p isa OpenInterpolatingSplinePath
      @test p isa SplinePath{false}
      @test p isa InterpolatingSplinePath{false}

      bez = bezier_path([xy(0, 0), xy(1, 1), xy(2, 0)])
      @test bez isa OpenBezierPath
      @test bez isa BezierPath{false}
      @test path_start(bez) == xy(0, 0)
      @test path_end(bez) == xy(2, 0)
      @test in_world(location_at(bez, 0.5)).y ≈ 0.5 atol=1e-10

      bs = bspline_path([xy(0, 0), xy(10, 0), xy(10, 10)], degree=1)
      @test bs isa OpenBSplinePath
      @test bs isa BSplinePath{false,false}
      @test bs isa SplinePath{false}
      @test in_world(location_at(bs, 0.25)).x ≈ 5 atol=1e-10

      p = nurbs_path([xy(0, 0), xy(10, 0)], degree=1)
      @test p isa OpenNurbsPath
      @test p isa BSplinePath{false,true}
      @test !is_closed_path(p)
      @test is_smooth_path(p)
      @test p.degree == 1

      cp = control_point_curve_path([xy(0, 0), xy(1, 0), xy(1, 1)], 2, true)
      @test cp isa ClosedBSplinePath
      @test cp isa BSplinePath{true,false}
      @test is_closed_path(cp)
    end
  end

  @testset "Path piece API" begin
    poly = open_polygonal_path([xy(0, 0), xy(2, 0), xy(2, 2)])
    pieces = path_pieces(poly)
    @test length(pieces) == 2
    @test all(s -> s isa LinePath, pieces)
    @test path_points(poly, mode=:control) == poly.vertices
    @test path_points(poly, mode=:breakpoints) == poly.vertices

    arc = arc_path(u0(), 2, 0, π/2)
    arc_pieces = path_pieces(arc)
    @test length(arc_pieces) == 1
    @test arc_pieces[1] isa ArcPath
    @test arc_pieces[1].radius == 2
    @test single_arc_path(arc) === arc
    @test single_arc_segment(arc) === arc

    seq = path_sequence(arc, open_polygonal_path([path_end(arc), xy(0, 3)]))
    seq_pieces = path_pieces(seq)
    @test length(seq_pieces) == 2
    @test seq_pieces[1] isa ArcPath
    @test seq_pieces[2] isa LinePath
    @test isnothing(single_arc_path(seq))
    unified = composite_path([line_path(xy(0, 0), xy(1, 0)),
                              line_path(xy(1, 0), xy(1, 1))])
    @test unified isa CompositePath{false}
    @test unified isa OpenPath
    @test path_vertices(unified) == [xy(0, 0), xy(1, 0), xy(1, 1)]
    tangent = in_world(vz(1, location_at_length(unified, 0.5).cs))
    @test tangent.x ≈ 1 atol=1e-10
    @test tangent.y ≈ 0 atol=1e-10
    @test path_start(reverse(unified)) == path_end(unified)

    closed_unified = composite_path([line_path(xy(0, 0), xy(1, 0)),
                                     line_path(xy(1, 0), xy(0, 0))];
                                    closed=true)
    @test closed_unified isa CompositePath{true}
    @test closed_unified isa ClosedPath
    @test is_closed_path(closed_unified)
    @test_throws ArgumentError path_sequence(open_polygonal_path([xy(0, 0), xy(1, 0)]),
                                             open_polygonal_path([xy(2, 0), xy(3, 0)]))
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

    @testset "NurbsPath domain" begin
      p = nurbs_path([xy(0, 0), xy(10, 0), xy(10, 10)], degree=1)
      @test path_domain(p) == (0, 1)
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

    @testset "NurbsPath degree 1" begin
      p = nurbs_path([xy(0, 0), xy(10, 0), xy(10, 10)], degree=1)
      loc_first_segment = in_world(location_at(p, 0.25))
      @test loc_first_segment.x ≈ 5 atol=1e-10
      @test loc_first_segment.y ≈ 0 atol=1e-10

      loc_second_segment = in_world(location_at(p, 0.75))
      @test loc_second_segment.x ≈ 10 atol=1e-10
      @test loc_second_segment.y ≈ 5 atol=1e-10
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

    @testset "NurbsPath arc-length lookup and division" begin
      p = nurbs_path([xy(0, 0), xy(10, 0), xy(10, 10)], degree=1)
      @test path_length(p) ≈ 20 atol=1e-6

      mid = in_world(location_at_length(p, 10))
      @test mid.x ≈ 10 atol=1e-6
      @test mid.y ≈ 0 atol=1e-6

      params = divide_path_parameters_by_count(circular_path(u0(), 2), 4)
      @test params ≈ [0, π/2, π, 3π/2, 2π] atol=1e-10

      pts = divide_path_by_count(p, 2)
      @test length(pts) == 3
      corner = in_world(pts[2])
      @test corner.x ≈ 10 atol=1e-6
      @test corner.y ≈ 0 atol=1e-6

      interior = divide_path_parameters_by_count(p, 2; include_ends=false)
      @test length(interior) == 1
      @test interior[1] ≈ 0.5 atol=1e-6

      q = nurbs_path([xy(0, 0), xy(10, 10), xy(20, 0)], degree=2)
      q_params = divide_path_parameters_by_count(q, 4)
      q_lengths = [length_at_parameter(q, t) for t in q_params]
      step = path_length(q) / 4
      @test all(isapprox(q_lengths[i + 1] - q_lengths[i], step; atol=1e-5)
                for i in 1:4)
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
    with(coincidence_tolerance, 1e-5) do
      @test coincident_path_location(xy(0, 0), xy(1e-6, 1e-6))
    end
    with(coincidence_tolerance, 1e-10) do
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
      @test p isa CompositePath{false}
      @test path_length(p) ≈ 5 atol=1e-10

      loc = in_world(location_at_length(p, 2.5))
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

  @testset "CompositePath" begin
    @testset "open composite" begin
      p1 = open_polygonal_path([xy(0, 0), xy(1, 0)])
      p2 = open_polygonal_path([xy(1, 0), xy(1, 1)])
      seq = open_path_sequence(p1, p2)
      @test !is_closed_path(seq)
      @test path_length(seq) ≈ 2 atol=1e-10
    end

    @testset "closed composite" begin
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
    @test pset isa KhepriBase.GeometryElement
    @test !(pset isa Path)
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

    # The ArcPath reverse invariant: the reversed arc must trace the same
    # world points as the original, in opposite order. A door/window frame
    # on a curved wall depends on this — the frame head is built as
    # `translate(reverse(subpath), vz(height))` and must meet both jambs.
    # See src/BIM.jl frame_path and src/Paths.jl Base.reverse(::ArcPath).
    @testset "reverse ArcPath" begin
      # Compares world-space positions only (CS of the reversed Loc differs).
      same_point(a, b; atol=1e-10) = begin
        aw = in_world(a); bw = in_world(b)
        isapprox(aw.x, bw.x; atol) && isapprox(aw.y, bw.y; atol) && isapprox(aw.z, bw.z; atol)
      end

      arcs = [
        arc_path(u0(), 5, 0, π),              # user's example
        arc_path(u0(), 5, 0.4, 0.2),          # sub-arc from the user's example
        arc_path(xyz(2, -3, 1), 2.5, 0.7, 1.3),  # off-origin, non-trivial start_angle
        arc_path(u0(), 1, -π/3, π/4),         # negative start_angle
      ]

      for p in arcs
        pr = reverse(p)
        L = path_length(p)
        @test path_length(pr) ≈ L atol=1e-10
        @test same_point(path_start(pr), path_end(p))
        @test same_point(path_end(pr),   path_start(p))
        # Endpoints alone could be satisfied by a different arc of equal
        # length — sample interior points too.
        for d in (L/4, L/2, 3L/4)
          @test same_point(location_at_length(pr, d),
                           location_at_length(p,  L - d))
        end
        # Reverse of reverse retraces the original.
        prr = reverse(pr)
        for d in (0.0, L/3, 2L/3, L)
          @test same_point(location_at_length(prr, d),
                           location_at_length(p,   d))
        end
      end
    end

    # CircularPath is closed, so start position is free; what matters
    # is that the traversal direction is reversed in world.
    @testset "reverse CircularPath" begin
      r = 5.0
      p = circular_path(u0(), r)
      pr = reverse(p)
      @test path_length(pr) ≈ path_length(p) atol=1e-10
      # Forward at (r,0,0) at t=0 steps toward +y (CCW).
      ε = 1e-4
      tangent_fwd = in_world(location_at_length(p, ε)) - in_world(location_at_length(p, 0))
      @test tangent_fwd.y > 0
      # Reversed circle starts at (0,r,0) (the +X' of the flipped CS)
      # and sweeps CW in world, so it reaches (r,0,0) after a quarter
      # turn — arc-length π·r/2. Tangent there points to −y (CW).
      quarter = π * r / 2
      at_r0 = location_at_length(pr, quarter)
      @test isapprox(in_world(at_r0).x, r; atol=1e-8)
      @test isapprox(in_world(at_r0).y, 0; atol=1e-8)
      tangent_rev = in_world(location_at_length(pr, quarter + ε)) -
                    in_world(location_at_length(pr, quarter))
      @test tangent_rev.y < 0
    end

    @testset "translate ∘ reverse ArcPath (frame_path invariant)" begin
      # This mirrors the construction in BIM.jl frame_path for a door on
      # a curved wall: translate(reverse(subpath), vz(h)) must meet the
      # translated subpath endpoints lifted by vz(h).
      same_point(a, b; atol=1e-10) = begin
        aw = in_world(a); bw = in_world(b)
        isapprox(aw.x, bw.x; atol) && isapprox(aw.y, bw.y; atol) && isapprox(aw.z, bw.z; atol)
      end
      wall = arc_path(u0(), 5, 0, π)
      sub = subpath(wall, 2, 3)   # door opening from 2m to 3m
      h = 2.0
      head = translate(reverse(sub), vz(h))
      # Right-jamb top = sub[end] + vz(h); head must start there.
      @test same_point(path_start(head), path_end(sub) + vz(h))
      # Left-jamb top = sub[begin] + vz(h); head must end there.
      @test same_point(path_end(head),   path_start(sub) + vz(h))
    end
  end

  @testset "join_paths arc combinations" begin
    # Regression: windows on arc walls foldr-join into this pattern.
    # Without a join_paths(ArcPath, CompositePath) method this would
    # raise MethodError.
    arc = arc_path(u0(), 5, 0, π/2)
    seq = path_sequence(
      open_polygonal_path([path_end(arc), path_end(arc) + vz(1)]),
      open_polygonal_path([path_end(arc) + vz(1), path_start(arc) + vz(1)]))
    joined = join_paths(arc, seq)
    @test joined isa CompositePath{false}
    @test length(joined.pieces) == 3
  end

  # The arc-wall door/window frame split in src/BIM.jl reads the wall
  # tangent from the Z axis of `path_end(subpath).cs` and
  # `path_start(subpath).cs`. If those CS axes don't point along the
  # arc tangent, the computed per-jamb profile rotation (`atan2(t.x, t.y)`)
  # will be wrong.
  @testset "ArcPath endpoint CS exposes tangent" begin
    # Unit-radius arc at origin spanning 0→π/2.
    p = arc_path(u0(), 1.0, 0.0, π/2)
    # At the start (θ=0), world tangent for CCW sweep is (0, 1, 0).
    t_start = in_world(vz(1, path_start(p).cs))
    @test isapprox(t_start.x, 0;  atol=1e-10)
    @test isapprox(t_start.y, 1;  atol=1e-10)
    @test isapprox(t_start.z, 0;  atol=1e-10)
    # At the end (θ=π/2), tangent is (-1, 0, 0).
    t_end = in_world(vz(1, path_end(p).cs))
    @test isapprox(t_end.x, -1; atol=1e-10)
    @test isapprox(t_end.y,  0; atol=1e-10)
    @test isapprox(t_end.z,  0; atol=1e-10)
    # Non-trivial start_angle: arc from θ=0.4 to 0.6 on radius 5.
    # These are the door's subpath bounds in the user's example.
    sub = arc_path(xyz(0, 0, 0.001), 5, 0.4, 0.2)
    # Wall tangent at θ = (-sin θ, cos θ, 0). The jamb rotation
    # `atan2(t.x, t.y)` should equal −θ, matching our derivation.
    t_begin = in_world(vz(1, path_start(sub).cs))
    @test isapprox(atan(t_begin.x, t_begin.y), -0.4; atol=1e-10)
    t_last = in_world(vz(1, path_end(sub).cs))
    @test isapprox(atan(t_last.x, t_last.y), -0.6; atol=1e-10)
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
    @test m isa KhepriBase.GeometryElement
    @test !(m isa Path)
  end

  @testset "length_at_location" begin
    p = open_polygonal_path([xy(0, 0), xy(10, 0)])
    loc = xy(5, 0)
    len = length_at_location(p, loc)
    @test len ≈ 5 atol=1e-1  # Approximate since it's iterative
  end

end
