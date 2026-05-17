using Test
using KhepriBase

@testset "Surface geometry" begin
  @test region(rectangular_path()) isa SurfaceGeometry
  @test mesh() isa SurfaceGeometry

  @testset "PlaneSurface" begin
    s = plane_surface(u0())
    p = in_world(location_at(s, 0.25, 0.75))
    @test p.x ≈ 0.25 atol=1e-10
    @test p.y ≈ 0.75 atol=1e-10
    n = normal_at(s, 0.5, 0.5)
    @test n.z ≈ 1 atol=1e-10
  end

  @testset "BezierSurface" begin
    cps = [xy(0, 0) xy(1, 0);
           xy(0, 1) xy(1, 1)]
    s = bezier_surface(cps)
    @test s isa BezierSurface{false,false}
    @test surface_domain(s) == (0.0, 1.0, 0.0, 1.0)
    p = in_world(location_at(s, 0.5, 0.5))
    @test p.x ≈ 0.5 atol=1e-10
    @test p.y ≈ 0.5 atol=1e-10
    @test surface_points(s, mode=:control) == s.control_points
  end

  @testset "B-spline and NURBS surfaces" begin
    cps = [xy(0, 0) xy(0, 1);
           xy(1, 0) xy(1, 1)]
    bs = bspline_surface(cps, degree_u=1, degree_v=1)
    @test bs isa BSplineSurface{false,false,false}
    @test in_world(location_at(bs, 0.5, 0.5)).x ≈ 0.5 atol=1e-10
    @test in_world(location_at(bs, 0.5, 0.5)).y ≈ 0.5 atol=1e-10

    ns = nurbs_surface(cps, degree_u=1, degree_v=1)
    @test ns isa NurbsSurface{false,false}
    @test ns isa BSplineSurface{false,false,true}
    @test in_world(location_at(ns, 0.5, 0.5)).x ≈ 0.5 atol=1e-10
  end

  @testset "Tessellation and trims" begin
    s = bezier_surface([xy(0, 0) xy(1, 0);
                        xy(0, 1) xy(1, 1)])
    m = tessellate_surface(s, u_count=2, v_count=3)
    @test m isa KhepriBase.Mesh
    @test length(m.vertices) == 12
    @test length(m.faces) == 6

    r = region(rectangular_path(xy(0, 0), 4, 3),
               rectangular_path(xy(1, 1), 1, 1))
    ts = trimmed_surface(r)
    @test ts isa TrimmedSurface{PlaneSurface}
    @test surface_boundary(ts) == outer_path(r)
    @test surface_trims(ts) == inner_paths(r)
    @test trimmed_surface_contains(ts, 0.5, -0.5)
    @test !trimmed_surface_contains(ts, 1.5, -1.5)
    plane_domain = surface_domain(ts)
    @test plane_domain[1] ≈ 0 atol=1e-12
    @test plane_domain[2] ≈ 3 atol=1e-12
    @test plane_domain[3] ≈ -4 atol=1e-12
    @test plane_domain[4] ≈ 0 atol=1e-12
    plane_trimmed_mesh = tessellate_surface(ts, u_count=4, v_count=4)
    @test 0 < length(plane_trimmed_mesh.faces) < 16

    nonplanar = bezier_surface([xyz(0, 0, 0) xyz(0, 1, 0);
                                xyz(1, 0, 0.25) xyz(1, 1, 0)])
    trimmed = trimmed_surface(nonplanar,
                              rectangular_path(xy(0, 0), 0.5, 1.0),
                              rectangular_path(xy(0.1, 0.1), 0.2, 0.2))
    @test trimmed_surface_contains(trimmed, 0.25, 0.5)
    @test !trimmed_surface_contains(trimmed, 0.2, 0.2)
    @test !trimmed_surface_contains(trimmed, 0.75, 0.5)
    @test surface_trim_uv_points(trimmed) == [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]
    @test length(surface_trim_points(trimmed)) == 4

    trimmed_mesh = tessellate_surface(trimmed, u_count=4, v_count=4)
    @test trimmed_mesh isa KhepriBase.Mesh
    @test 0 < length(trimmed_mesh.faces) < length(tessellate_surface(nonplanar, u_count=4, v_count=4).faces)
    @test all(face -> all(i -> 0 <= i < length(trimmed_mesh.vertices), face), trimmed_mesh.faces)
  end
end
