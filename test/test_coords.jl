# test_coords.jl - Comprehensive tests for coordinate systems

using Test
using KhepriBase

@testset "Coords" begin

  @testset "Cartesian coordinate construction" begin
    @testset "X (1D)" begin
      p = x(5)
      @test p.x == 5
      @test p.y == 0
      @test p.z == 0
      @test p.cs === current_cs()
    end

    @testset "XY (2D)" begin
      p = xy(3, 4)
      @test p.x == 3
      @test p.y == 4
      @test p.z == 0

      p2 = y(7)
      @test p2.x == 0
      @test p2.y == 7
    end

    @testset "XYZ (3D)" begin
      p = xyz(1, 2, 3)
      @test p.x == 1
      @test p.y == 2
      @test p.z == 3

      p2 = xz(5, 10)
      @test p2.x == 5
      @test p2.y == 0
      @test p2.z == 10

      p3 = yz(7, 8)
      @test p3.x == 0
      @test p3.y == 7
      @test p3.z == 8

      p4 = z(15)
      @test p4.x == 0
      @test p4.y == 0
      @test p4.z == 15
    end
  end

  @testset "Polar coordinate construction" begin
    @testset "Pol (2D polar)" begin
      p = pol(1, 0)
      @test p.x ≈ 1 atol=1e-10
      @test p.y ≈ 0 atol=1e-10
      @test p.ρ ≈ 1 atol=1e-10
      @test p.ϕ ≈ 0 atol=1e-10

      p2 = pol(1, π/2)
      @test p2.x ≈ 0 atol=1e-10
      @test p2.y ≈ 1 atol=1e-10
      @test p2.ρ ≈ 1 atol=1e-10
      @test p2.ϕ ≈ π/2 atol=1e-10

      p3 = pol(1, π)
      @test p3.x ≈ -1 atol=1e-10
      @test p3.y ≈ 0 atol=1e-10
      @test p3.ρ ≈ 1 atol=1e-10
      @test p3.ϕ ≈ π atol=1e-10

      p4 = pol(2, π/4)
      @test p4.x ≈ √2 atol=1e-10
      @test p4.y ≈ √2 atol=1e-10
      @test p4.ρ ≈ 2 atol=1e-10
    end

    @testset "Pold (polar with degrees)" begin
      p = KhepriBase.pold(1, 90)
      # Pold stores polar coordinates with degrees
      @test p.ρ ≈ 1 atol=1e-10
      @test p.ϕ ≈ 90 atol=1e-10
      # Cartesian coords are in the raw field
      @test p.raw[1] ≈ 0 atol=1e-10  # x = cos(90°) = 0
      @test p.raw[2] ≈ 1 atol=1e-10  # y = sin(90°) = 1
    end
  end

  @testset "Cylindrical coordinate construction" begin
    p = cyl(2, π/4, 5)
    @test p.x ≈ √2 atol=1e-10
    @test p.y ≈ √2 atol=1e-10
    @test p.z ≈ 5 atol=1e-10
    @test p.ρ ≈ 2 atol=1e-10
    @test p.ϕ ≈ π/4 atol=1e-10
  end

  @testset "Spherical coordinate construction" begin
    # At the north pole
    p = sph(1, 0, 0)
    @test p.x ≈ 0 atol=1e-10
    @test p.y ≈ 0 atol=1e-10
    @test p.z ≈ 1 atol=1e-10

    # On the equator
    p2 = sph(1, 0, π/2)
    @test p2.x ≈ 1 atol=1e-10
    @test p2.y ≈ 0 atol=1e-10
    @test p2.z ≈ 0 atol=1e-10

    # On the equator at 90 degrees
    p3 = sph(1, π/2, π/2)
    @test p3.x ≈ 0 atol=1e-10
    @test p3.y ≈ 1 atol=1e-10
    @test p3.z ≈ 0 atol=1e-10
  end

  @testset "Vector construction" begin
    @testset "VX, VXY, VXYZ" begin
      v = vx(3)
      @test v.x == 3
      @test v.y == 0
      @test v.z == 0

      v2 = vxy(4, 5)
      @test v2.x == 4
      @test v2.y == 5
      @test v2.z == 0

      v3 = vxyz(1, 2, 3)
      @test v3.x == 1
      @test v3.y == 2
      @test v3.z == 3

      v4 = vy(7)
      @test v4.x == 0
      @test v4.y == 7

      v5 = vz(9)
      @test v5.x == 0
      @test v5.y == 0
      @test v5.z == 9
    end

    @testset "VPol (polar vectors)" begin
      v = vpol(1, 0)
      @test v.x ≈ 1 atol=1e-10
      @test v.y ≈ 0 atol=1e-10

      v2 = vpol(1, π/2)
      @test v2.x ≈ 0 atol=1e-10
      @test v2.y ≈ 1 atol=1e-10
    end

    @testset "VCyl (cylindrical vectors)" begin
      v = vcyl(2, π/4, 5)
      @test v.x ≈ √2 atol=1e-10
      @test v.y ≈ √2 atol=1e-10
      @test v.z ≈ 5 atol=1e-10
    end

    @testset "VSph (spherical vectors)" begin
      v = vsph(1, 0, π/2)
      @test v.x ≈ 1 atol=1e-10
      @test v.y ≈ 0 atol=1e-10
      @test v.z ≈ 0 atol=1e-10
    end
  end

  @testset "Coordinate selectors" begin
    p = xyz(3, 4, 5)
    @test cx(p) == 3
    @test cy(p) == 4
    @test cz(p) == 5

    @test pol_rho(p) ≈ 5 atol=1e-10  # sqrt(3^2 + 4^2)
    @test cyl_rho(p) ≈ 5 atol=1e-10
    @test cyl_z(p) == 5

    @test sph_rho(p) ≈ sqrt(50) atol=1e-10  # sqrt(3^2 + 4^2 + 5^2)
  end

  @testset "Arithmetic operations" begin
    @testset "Location + Vector" begin
      p = xyz(1, 2, 3)
      v = vxyz(4, 5, 6)
      result = p + v
      @test result.x ≈ 5 atol=1e-10
      @test result.y ≈ 7 atol=1e-10
      @test result.z ≈ 9 atol=1e-10
    end

    @testset "Location - Vector" begin
      p = xyz(10, 10, 10)
      v = vxyz(1, 2, 3)
      result = p - v
      @test result.x ≈ 9 atol=1e-10
      @test result.y ≈ 8 atol=1e-10
      @test result.z ≈ 7 atol=1e-10
    end

    @testset "Location - Location = Vector" begin
      p1 = xyz(10, 20, 30)
      p2 = xyz(1, 2, 3)
      v = p1 - p2
      @test v.x ≈ 9 atol=1e-10
      @test v.y ≈ 18 atol=1e-10
      @test v.z ≈ 27 atol=1e-10
    end

    @testset "Vector + Vector" begin
      v1 = vxyz(1, 2, 3)
      v2 = vxyz(4, 5, 6)
      result = v1 + v2
      @test result.x ≈ 5 atol=1e-10
      @test result.y ≈ 7 atol=1e-10
      @test result.z ≈ 9 atol=1e-10
    end

    @testset "Vector - Vector" begin
      v1 = vxyz(10, 10, 10)
      v2 = vxyz(1, 2, 3)
      result = v1 - v2
      @test result.x ≈ 9 atol=1e-10
      @test result.y ≈ 8 atol=1e-10
      @test result.z ≈ 7 atol=1e-10
    end

    @testset "Vector negation" begin
      v = vxyz(1, 2, 3)
      neg_v = -v
      @test neg_v.x ≈ -1 atol=1e-10
      @test neg_v.y ≈ -2 atol=1e-10
      @test neg_v.z ≈ -3 atol=1e-10
    end

    @testset "Vector * Scalar" begin
      v = vxyz(1, 2, 3)
      result = v * 2
      @test result.x ≈ 2 atol=1e-10
      @test result.y ≈ 4 atol=1e-10
      @test result.z ≈ 6 atol=1e-10

      result2 = 3 * v
      @test result2.x ≈ 3 atol=1e-10
      @test result2.y ≈ 6 atol=1e-10
      @test result2.z ≈ 9 atol=1e-10
    end

    @testset "Vector / Scalar" begin
      v = vxyz(4, 6, 8)
      result = v / 2
      @test result.x ≈ 2 atol=1e-10
      @test result.y ≈ 3 atol=1e-10
      @test result.z ≈ 4 atol=1e-10
    end
  end

  @testset "add_* operations" begin
    p = xyz(0, 0, 0)
    @test add_x(p, 5).x == 5
    @test add_y(p, 5).y == 5
    @test add_z(p, 5).z == 5

    p2 = add_xy(p, 3, 4)
    @test p2.x == 3
    @test p2.y == 4
    @test p2.z == 0

    p3 = add_xz(p, 3, 5)
    @test p3.x == 3
    @test p3.z == 5

    p4 = add_yz(p, 4, 6)
    @test p4.y == 4
    @test p4.z == 6

    p5 = add_xyz(p, 1, 2, 3)
    @test p5.x == 1
    @test p5.y == 2
    @test p5.z == 3

    p6 = add_pol(p, 1, 0)
    @test p6.x ≈ 1 atol=1e-10
    @test p6.y ≈ 0 atol=1e-10

    p7 = add_cyl(p, 1, π/2, 5)
    @test p7.x ≈ 0 atol=1e-10
    @test p7.y ≈ 1 atol=1e-10
    @test p7.z ≈ 5 atol=1e-10
  end

  @testset "Vector operations" begin
    @testset "norm/length" begin
      v = vxyz(3, 4, 0)
      @test norm(v) ≈ 5 atol=1e-10
      @test length(v) ≈ 5 atol=1e-10

      v2 = vxyz(1, 2, 2)
      @test norm(v2) ≈ 3 atol=1e-10
    end

    @testset "unitized" begin
      v = vxyz(3, 4, 0)
      u = unitized(v)
      @test norm(u) ≈ 1 atol=1e-10
      @test u.x ≈ 0.6 atol=1e-10
      @test u.y ≈ 0.8 atol=1e-10
    end

    @testset "dot product" begin
      v1 = vxyz(1, 0, 0)
      v2 = vxyz(0, 1, 0)
      @test dot(v1, v2) ≈ 0 atol=1e-10

      v3 = vxyz(1, 2, 3)
      v4 = vxyz(4, 5, 6)
      @test dot(v3, v4) ≈ 32 atol=1e-10  # 1*4 + 2*5 + 3*6
    end

    @testset "cross product" begin
      v1 = vxyz(1, 0, 0)
      v2 = vxyz(0, 1, 0)
      c = cross(v1, v2)
      @test c.x ≈ 0 atol=1e-10
      @test c.y ≈ 0 atol=1e-10
      @test c.z ≈ 1 atol=1e-10

      c2 = cross(v2, v1)
      @test c2.z ≈ -1 atol=1e-10
    end

    @testset "angle_between" begin
      v1 = vxyz(1, 0, 0)
      v2 = vxyz(0, 1, 0)
      @test angle_between(v1, v2) ≈ π/2 atol=1e-10

      v3 = vxyz(1, 1, 0)
      @test angle_between(v1, v3) ≈ π/4 atol=1e-10
    end
  end

  @testset "distance" begin
    p1 = xyz(0, 0, 0)
    p2 = xyz(3, 4, 0)
    @test distance(p1, p2) ≈ 5 atol=1e-10

    p3 = xyz(1, 1, 1)
    p4 = xyz(2, 2, 2)
    @test distance(p3, p4) ≈ sqrt(3) atol=1e-10
  end

  @testset "min_loc / max_loc" begin
    p1 = xyz(1, 5, 3)
    p2 = xyz(4, 2, 6)

    minp = min_loc(p1, p2)
    @test minp.x ≈ 1 atol=1e-10
    @test minp.y ≈ 2 atol=1e-10
    @test minp.z ≈ 3 atol=1e-10

    maxp = max_loc(p1, p2)
    @test maxp.x ≈ 4 atol=1e-10
    @test maxp.y ≈ 5 atol=1e-10
    @test maxp.z ≈ 6 atol=1e-10
  end

  @testset "Unit locations and vectors" begin
    @test u0().x == 0
    @test u0().y == 0
    @test u0().z == 0

    @test ux().x == 1
    @test uy().y == 1
    @test uz().z == 1

    @test uvx().x == 1
    @test uvy().y == 1
    @test uvz().z == 1
  end

  @testset "Coordinate system transformations" begin
    @testset "translated_cs" begin
      cs1 = translated_cs(world_cs, 1, 2, 3)
      p = xyz(0, 0, 0, cs1)
      pw = in_world(p)
      @test pw.x ≈ 1 atol=1e-10
      @test pw.y ≈ 2 atol=1e-10
      @test pw.z ≈ 3 atol=1e-10
    end

    @testset "scaled_cs" begin
      cs1 = scaled_cs(world_cs, 2, 2, 2)
      p = xyz(1, 1, 1, cs1)
      pw = in_world(p)
      @test pw.x ≈ 2 atol=1e-10
      @test pw.y ≈ 2 atol=1e-10
      @test pw.z ≈ 2 atol=1e-10
    end

    @testset "rotated_z_cs" begin
      cs1 = rotated_z_cs(world_cs, π/2)
      p = xyz(1, 0, 0, cs1)
      pw = in_world(p)
      @test pw.x ≈ 0 atol=1e-10
      @test pw.y ≈ 1 atol=1e-10
      @test pw.z ≈ 0 atol=1e-10
    end
  end

  @testset "in_cs / in_world / on_cs" begin
    @testset "in_world" begin
      p = xyz(1, 2, 3)
      pw = in_world(p)
      @test pw.x ≈ 1 atol=1e-10
      @test pw.y ≈ 2 atol=1e-10
      @test pw.z ≈ 3 atol=1e-10
      @test pw.cs === world_cs
    end

    @testset "in_cs with translated CS" begin
      cs1 = translated_cs(world_cs, 10, 20, 30)
      p = xyz(1, 2, 3)
      p_in_cs1 = in_cs(p, cs1)
      @test p_in_cs1.x ≈ -9 atol=1e-10
      @test p_in_cs1.y ≈ -18 atol=1e-10
      @test p_in_cs1.z ≈ -27 atol=1e-10
    end

    @testset "on_cs" begin
      cs1 = translated_cs(world_cs, 10, 0, 0)
      p = xyz(1, 2, 3)
      p_on_cs1 = on_cs(p, cs1)
      pw = in_world(p_on_cs1)
      @test pw.x ≈ 11 atol=1e-10
      @test pw.y ≈ 2 atol=1e-10
      @test pw.z ≈ 3 atol=1e-10
    end
  end

  @testset "cs_from_* constructors" begin
    @testset "cs_from_o_vx_vy_vz" begin
      cs = cs_from_o_vx_vy_vz(xyz(0, 0, 0), vxyz(1, 0, 0), vxyz(0, 1, 0), vxyz(0, 0, 1))
      @test is_world_cs(cs)
    end

    @testset "cs_from_o_phi" begin
      cs = cs_from_o_phi(xyz(5, 5, 5), π/4)
      p = xyz(1, 0, 0, cs)
      pw = in_world(p)
      @test pw.x ≈ 5 + √2/2 atol=1e-10
      @test pw.y ≈ 5 + √2/2 atol=1e-10
    end
  end

  @testset "loc_from_* constructors" begin
    @testset "loc_from_o_vz" begin
      loc = loc_from_o_vz(xyz(0, 0, 0), vxyz(0, 0, 1))
      @test loc.x ≈ 0 atol=1e-10
      @test loc.y ≈ 0 atol=1e-10
      @test loc.z ≈ 0 atol=1e-10
    end

    @testset "loc_from_o_phi" begin
      loc = loc_from_o_phi(xyz(1, 2, 3), 0)
      @test loc.x ≈ 0 atol=1e-10
      @test loc.y ≈ 0 atol=1e-10
      @test loc.z ≈ 0 atol=1e-10

      pw = in_world(loc)
      @test pw.x ≈ 1 atol=1e-10
      @test pw.y ≈ 2 atol=1e-10
      @test pw.z ≈ 3 atol=1e-10
    end
  end

  @testset "intermediate_loc" begin
    p1 = xyz(0, 0, 0)
    p2 = xyz(10, 10, 10)

    mid = intermediate_loc(p1, p2)
    @test mid.x ≈ 5 atol=1e-10
    @test mid.y ≈ 5 atol=1e-10
    @test mid.z ≈ 5 atol=1e-10

    quarter = intermediate_loc(p1, p2, 0.25)
    @test quarter.x ≈ 2.5 atol=1e-10
    @test quarter.y ≈ 2.5 atol=1e-10
    @test quarter.z ≈ 2.5 atol=1e-10
  end

  @testset "regular_polygon_vertices" begin
    # Triangle
    verts = regular_polygon_vertices(3, u0(), 1, 0, true)
    @test length(verts) == 3

    # Square
    verts = regular_polygon_vertices(4, u0(), 1, 0, true)
    @test length(verts) == 4

    # Hexagon
    verts = regular_polygon_vertices(6, u0(), 1, 0, true)
    @test length(verts) == 6
    # First vertex at angle 0
    @test verts[1].x ≈ 1 atol=1e-10
    @test verts[1].y ≈ 0 atol=1e-10
  end

  @testset "Coordinate equality" begin
    p1 = xyz(1, 2, 3)
    p2 = xyz(1, 2, 3)
    @test isequal(p1, p2)
    @test isapprox(p1, p2)

    p3 = xyz(1.0000000001, 2, 3)
    @test isapprox(p1, p3, atol=1e-9)
  end

  @testset "trig/quad utilities" begin
    p0 = xyz(0, 0, 0)
    p1 = xyz(1, 0, 0)
    p2 = xyz(0, 1, 0)

    center = trig_center(p0, p1, p2)
    @test center.x ≈ 1/3 atol=1e-10
    @test center.y ≈ 1/3 atol=1e-10
    @test center.z ≈ 0 atol=1e-10

    normal = trig_normal(p0, p1, p2)
    @test abs(normal.z) ≈ 1 atol=1e-10
  end

  @testset "rotate_vector" begin
    v = vxyz(1, 0, 0)
    axis = vxyz(0, 0, 1)
    rotated = rotate_vector(v, axis, π/2)
    @test rotated.x ≈ 0 atol=1e-10
    @test rotated.y ≈ 1 atol=1e-10
    @test rotated.z ≈ 0 atol=1e-10
  end

  @testset "raw_point / raw_plane" begin
    p = xyz(1, 2, 3)
    raw = raw_point(p)
    @test raw == (1.0, 2.0, 3.0)
  end

  @testset "Conversions between coordinate types" begin
    # XY to Pol
    p = xy(1, 0)
    pp = pol(p)
    @test pp.ρ ≈ 1 atol=1e-10
    @test pp.ϕ ≈ 0 atol=1e-10

    # Pol to XY
    p2 = pol(1, π/4)
    pxy = xy(p2)
    @test pxy.x ≈ √2/2 atol=1e-10
    @test pxy.y ≈ √2/2 atol=1e-10

    # XYZ to Cyl - cyl_rho, cyl_phi extract cylindrical coords
    p3 = xyz(1, 1, 5)
    @test cyl_rho(p3) ≈ √2 atol=1e-10
    @test cyl_phi(p3) ≈ π/4 atol=1e-10
    @test cyl_z(p3) ≈ 5 atol=1e-10

    # XYZ to Sph
    p4 = xyz(0, 0, 1)
    ps = sph(p4)
    @test ps.ρ ≈ 1 atol=1e-10
    @test ps.ψ ≈ 0 atol=1e-10
  end

end
