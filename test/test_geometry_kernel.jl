# test_geometry_kernel.jl — rank-4 geometry-kernel correctness regression tests.
#
# These pin silent-corruption fixes (a wrong fix re-introduces corruption rather
# than crashing), so each asserts geometrically correct output, not just the
# absence of a NaN/MethodError.
using Test
using KhepriBase

@testset "Geometry kernel (rank 4)" begin
  @testset "_polygon_signed_area degenerate + shoelace sign (GEOMOPS-8)" begin
    psa = KhepriBase._polygon_signed_area
    # Degenerate polygons enclose zero area and must not crash (n=0 previously threw).
    @test psa(KhepriBase.Loc[]) == 0.0
    @test psa([xy(0, 0)]) == 0.0
    @test psa([xy(0, 0), xy(1, 0)]) == 0.0
    # Shoelace convention preserved for n >= 3: CCW positive, CW negative.
    @test psa([xy(0, 0), xy(1, 0), xy(0, 1)]) ≈ 0.5 atol=1e-12
    @test psa([xy(0, 0), xy(0, 1), xy(1, 0)]) ≈ -0.5 atol=1e-12
    @test psa([xy(0, 0), xy(1, 0), xy(1, 1), xy(0, 1)]) ≈ 1.0 atol=1e-12
    @test psa([xy(2, 3), xy(5, 3), xy(5, 7), xy(2, 7)]) ≈ 12.0 atol=1e-12
  end

  @testset "Cyl/Sph location minus matching curvilinear vector (COORDS-5)" begin
    @test in_world(cyl(2, 0, 0) - vcyl(1, 0, 0)) ≈ in_world(cyl(1, 0, 0))
    @test in_world(sph(2, 0, 0) - vsph(1, 0, 0)) ≈ in_world(sph(1, 0, 0))
    # Subtraction is the inverse of addition: p - v must equal p + (-v) in world.
    wc(loc) = (in_world(loc).x, in_world(loc).y, in_world(loc).z)
    let p = cyl(3, π/4, 2), v = vcyl(1.5, π/3, 0.7)
      @test all(isapprox.(wc(p - v), wc(p + (-v)); atol=1e-12))
    end
    let p = sph(3, π/5, π/6), v = vsph(1.2, π/7, π/8)
      @test all(isapprox.(wc(p - v), wc(p + (-v)); atol=1e-12))
    end
  end

  @testset "degenerate line-circle intersections (GEOMOPS-2)" begin
    c = circular_path(xy(0, 0), 1)
    # Zero-length line whose point lies ON the circle: yields that point, not a
    # NaN root from the singular quadratic (a = dx^2+dy^2 = 0).
    on = intersection_points(intersections(line_path(xyz(1, 0, 0), xyz(1, 0, 0)), c))
    @test length(on) == 1
    @test on[1].x ≈ 1 atol=1e-10
    @test on[1].y ≈ 0 atol=1e-10
    @test !any(p -> any(isnan, (p.x, p.y, p.z)), on)
    # Zero-length line off the circle: no intersection, no NaN.
    @test isempty(intersection_points(intersections(line_path(xyz(0.5, 0, 0), xyz(0.5, 0, 0)), c)))
    # Degenerate point still respects an arc's angular span.
    a = arc_path(u0(), 1, 0, pi)
    @test length(intersection_points(intersections(line_path(xyz(0, 1, 0), xyz(0, 1, 0)), a))) == 1
    @test isempty(intersection_points(intersections(line_path(xyz(0, -1, 0), xyz(0, -1, 0)), a)))
  end

  @testset "point_in_segment axis-aligned + bounds (COORDS-7)" begin
    pis = KhepriBase.point_in_segment
    # horizontal 2D (old per-axis-ratio code: NaN -> always false)
    @test pis(xy(0, 0), xy(0, 0), xy(10, 0))      # endpoint p
    @test pis(xy(10, 0), xy(0, 0), xy(10, 0))     # endpoint q
    @test pis(xy(5, 0), xy(0, 0), xy(10, 0))      # midpoint
    @test !pis(xy(5, 1), xy(0, 0), xy(10, 0))     # off the line
    @test !pis(xy(15, 0), xy(0, 0), xy(10, 0))    # collinear, beyond q
    # vertical 2D
    @test pis(xy(0, 5), xy(0, 0), xy(0, 10))
    @test !pis(xy(1, 5), xy(0, 0), xy(0, 10))
    # 3D diagonal: midpoint in, beyond-endpoint out
    @test pis(xyz(5, 5, 5), xyz(0, 0, 0), xyz(10, 10, 10))
    @test !pis(xyz(5, 5, 6), xyz(0, 0, 0), xyz(10, 10, 10))
    @test !pis(xyz(15, 15, 15), xyz(0, 0, 0), xyz(10, 10, 10))
    # zero-length segment
    @test pis(xy(3, 3), xy(3, 3), xy(3, 3))
    @test !pis(xy(4, 3), xy(3, 3), xy(3, 3))
  end

  @testset "Closed-surface tessellation (GEOMOPS-4)" begin
    # Open surface tessellation must be byte-identical to the pre-fix output.
    sopen = bezier_surface([xy(0, 0) xy(1, 0); xy(0, 1) xy(1, 1)])
    mo = tessellate_surface(sopen, u_count=2, v_count=3)
    @test length(mo.vertices) == 12
    @test length(mo.faces) == 6
    po = KhepriBase.sample_surface_points(sopen, 2, 3)
    nu, nv = size(po)
    oidx(i, j) = (j - 1) * nu + (i - 1)
    @test mo.faces == [[oidx(i, j), oidx(i + 1, j), oidx(i + 1, j + 1), oidx(i, j + 1)]
                       for j in 1:nv-1 for i in 1:nu-1]

    # Closed-in-u NURBS cylinder: no duplicate seam row, no degenerate quad.
    ring = [xyz(1, 0, 0), xyz(0, 1, 0), xyz(-1, 0, 0), xyz(0, -1, 0)]
    cps = Matrix{KhepriBase.Loc}(undef, 4, 2)
    for i in 1:4
      cps[i, 1] = ring[i]
      cps[i, 2] = ring[i] + vz(2)
    end
    cyl = nurbs_surface(cps, degree_u=2, degree_v=1, closed_u=true, closed_v=false)
    @test cyl isa KhepriBase.NurbsSurface{true,false}
    pc = KhepriBase.sample_surface_points(cyl, 6, 2)
    @test size(pc, 1) == 6            # seam endpoint dropped (was 7 = count+1)
    wp = [in_world(p) for p in pc]
    @test minimum(distance(wp[i, 1], wp[j, 1]) for i in 1:6 for j in 1:6 if i < j) > 1e-9
    mc = tessellate_surface(cyl, u_count=6, v_count=2)
    @test length(mc.faces) == 12     # ncu*ncv = 6*2, no extra duplicated band
    @test all(f -> all(i -> 0 <= i < length(mc.vertices), f), mc.faces)
  end
end
