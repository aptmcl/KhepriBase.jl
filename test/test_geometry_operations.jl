using Test
using KhepriBase

@testset "Geometry operations" begin
  @testset "line-line intersections" begin
    a = line_path(xy(0, 0), xy(2, 0))
    b = line_path(xy(1, -1), xy(1, 1))
    r = intersections(a, b)

    @test r isa IntersectionSet
    @test length(r) == 1
    @test r[1] isa PointIntersection
    @test r[1].kind == :transversal
    @test r[1].point.x ≈ 1 atol=1e-10
    @test r[1].point.y ≈ 0 atol=1e-10
    @test r[1].parameters.first ≈ 1 atol=1e-10
    @test r[1].parameters.second ≈ 1 atol=1e-10
  end

  @testset "line-line overlap" begin
    a = line_path(xy(0, 0), xy(4, 0))
    b = line_path(xy(1, 0), xy(3, 0))
    r = intersections(a, b)

    @test length(r) == 1
    @test r[1] isa CurveIntersection
    @test r[1].kind == :overlap
    @test path_start(r[1].curve).x ≈ 1 atol=1e-10
    @test path_end(r[1].curve).x ≈ 3 atol=1e-10
  end

  @testset "line-circle intersections" begin
    l = line_path(xy(-2, 0), xy(2, 0))
    c = circular_path(xy(0, 0), 1)
    pts = sort(intersection_points(intersections(l, c)); by=p -> p.x)

    @test length(pts) == 2
    @test pts[1].x ≈ -1 atol=1e-10
    @test pts[1].y ≈ 0 atol=1e-10
    @test pts[2].x ≈ 1 atol=1e-10
    @test pts[2].y ≈ 0 atol=1e-10
  end

  @testset "circle-circle intersections" begin
    a = circular_path(xy(0, 0), 1)
    b = circular_path(xy(1, 0), 1)
    pts = sort(intersection_points(intersections(a, b)); by=p -> p.y)

    @test length(pts) == 2
    @test pts[1].x ≈ 0.5 atol=1e-10
    @test pts[1].y ≈ -sqrt(3)/2 atol=1e-10
    @test pts[2].x ≈ 0.5 atol=1e-10
    @test pts[2].y ≈ sqrt(3)/2 atol=1e-10
  end

  @testset "composite path intersections" begin
    rect = rectangular_path(xy(0, 0), 2, 2)
    cut = line_path(xy(-1, 1), xy(3, 1))
    pts = sort(intersection_points(intersections(rect, cut)); by=p -> p.x)

    @test length(pts) == 2
    @test pts[1].x ≈ 0 atol=1e-10
    @test pts[1].y ≈ 1 atol=1e-10
    @test pts[2].x ≈ 2 atol=1e-10
    @test pts[2].y ≈ 1 atol=1e-10
  end

  @testset "curve-plane intersections" begin
    plane = plane_surface(u0())
    line = line_path(xyz(0, 0, -1), xyz(0, 0, 1))
    r = intersections(line, plane)

    @test length(r) == 1
    @test r[1] isa PointIntersection
    @test r[1].point.z ≈ 0 atol=1e-10
    @test r[1].parameters.second.u ≈ 0 atol=1e-10
    @test r[1].parameters.second.v ≈ 0 atol=1e-10

    circle = circular_path(xyz(0, 0, 0), 2)
    vertical = plane_surface(loc_from_o_rot_x(u0(), pi/2))
    pts = sort(intersection_points(intersections(circle, vertical)); by=p -> p.x)
    @test length(pts) == 2
    @test pts[1].x ≈ -2 atol=1e-10
    @test pts[2].x ≈ 2 atol=1e-10

    coplanar = intersections(circle, plane)
    @test length(coplanar) == 1
    @test coplanar[1] isa CurveIntersection
    @test coplanar[1].kind == :overlap
  end

  @testset "plane-plane section" begin
    xy_plane = plane_surface(u0())
    yz_plane = plane_surface(loc_from_o_rot_x(u0(), pi/2))
    r = section(xy_plane, yz_plane)

    @test length(r) == 1
    @test r[1] isa CurveIntersection
    @test r[1].curve isa InfiniteLine
    @test abs(r[1].curve.direction.y) < 1e-10
    @test abs(r[1].curve.direction.z) < 1e-10
  end

  @testset "projection and closest points" begin
    plane = plane_surface(u0())
    p = xyz(1, 2, 3)
    projected = project(p, plane)

    @test projected.geometry.x ≈ 1 atol=1e-10
    @test projected.geometry.y ≈ 2 atol=1e-10
    @test projected.geometry.z ≈ 0 atol=1e-10
    @test projected.distance ≈ 3 atol=1e-10

    line = line_path(xy(0, 0), xy(2, 0))
    closest = closest_points(xy(1, 1), line)
    @test closest.second.x ≈ 1 atol=1e-10
    @test closest.second.y ≈ 0 atol=1e-10
    @test closest.distance ≈ 1 atol=1e-10
  end

  @testset "split line by intersections" begin
    a = line_path(xy(0, 0), xy(2, 0))
    b = line_path(xy(1, -1), xy(1, 1))
    pieces = split(a, b)

    @test pieces isa PathSet
    @test length(pieces.paths) == 2
    @test in_world(path_end(pieces.paths[1])).x ≈ 1 atol=1e-10
    @test in_world(path_start(pieces.paths[2])).x ≈ 1 atol=1e-10
  end

  @testset "convex region intersection" begin
    a = region(rectangular_path(xy(0, 0), 4, 3))
    b = region(rectangular_path(xy(2, 1), 4, 3))
    r = boolean(:intersection, a, b)

    @test r isa Region
    pts = path_vertices(outer_path(r))
    @test length(pts) == 4
    @test minimum(p.x for p in pts) ≈ 2 atol=1e-10
    @test maximum(p.x for p in pts) ≈ 4 atol=1e-10
    @test minimum(p.y for p in pts) ≈ 1 atol=1e-10
    @test maximum(p.y for p in pts) ≈ 3 atol=1e-10

    c = region(rectangular_path(xy(10, 10), 1, 1))
    empty = boolean(:intersection, a, c)
    @test empty isa MultiRegion
    @test isempty(empty.regions)
  end

  @testset "result containers" begin
    r = region(rectangular_path(xy(0, 0), 1, 1))
    mr = multi_region(r)
    gc = geometry_collection(xy(0, 0), mr)

    @test mr isa MultiRegion
    @test length(mr.regions) == 1
    @test gc isa GeometryCollection
    @test length(gc.geometries) == 2
  end
end
