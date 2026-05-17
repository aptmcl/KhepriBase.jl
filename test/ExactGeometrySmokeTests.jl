module ExactGeometrySmokeTests

using Test
using KhepriBase

export run_exact_geometry_smoke_tests

_smoke_mat(b) = KhepriBase.backend_name(b) == "Rhino" ? -1 : KhepriBase.void_ref(b)
_raw_coordinates(p::KhepriBase.Loc) = KhepriBase.raw_point(p)
_raw_coordinates(p) = p

function _raw_distance(p, q)
  a = _raw_coordinates(p)
  b = _raw_coordinates(q)
  sqrt(sum(abs2(a[i] - b[i]) for i in eachindex(a)))
end

function _same_location(actual, expected; atol)
  @test _raw_distance(actual, expected) <= atol
end

function _curve_samples_match(b, ref, path; atol)
  domain = Tuple(KhepriBase.@remote(b, CurveDomain(ref)))
  path_dom = KhepriBase.path_domain(path)
  backend_ts = collect(range(domain[1], domain[2]; length=5))
  path_ts = collect(range(path_dom[1], path_dom[2]; length=5))
  pts = KhepriBase.@remote(b, CurvePointsAt(ref, backend_ts))
  for (actual, t) in zip(pts, path_ts)
    _same_location(actual, KhepriBase.location_at(path, t); atol=atol)
  end
end

function _surface_samples_match(b, ref, surface; atol)
  domain = Tuple(KhepriBase.@remote(b, SurfaceDomain(ref)))
  surf_dom = KhepriBase.surface_domain(surface)
  backend_us = collect(range(domain[1], domain[2]; length=3))
  backend_vs = collect(range(domain[3], domain[4]; length=3))
  surface_us = collect(range(surf_dom[1], surf_dom[2]; length=3))
  surface_vs = collect(range(surf_dom[3], surf_dom[4]; length=3))
  for (bu, su) in zip(backend_us, surface_us), (bv, sv) in zip(backend_vs, surface_vs)
    frame = KhepriBase.@remote(b, SurfaceFrameAt(ref, bu, bv))
    _same_location(frame, KhepriBase.location_at(surface, su, sv); atol=atol)
  end
end

function _curve_cases()
  cps = [xyz(0, 0, 0), xyz(1, 0, 0.2), xyz(1, 1, 0.1), xyz(2, 1, 0)]
  bs_cps = [xyz(0, 0, 0), xyz(0.5, 1, 0.2), xyz(1.5, -0.5, 0.3), xyz(2, 0.5, 0)]
  nurbs_cps = [xyz(0, 0, 0), xyz(0.5, 1, 0), xyz(1.5, -0.5, 0.4), xyz(2, 0.5, 0)]
  [
    (:bezier, bezier_path(cps)),
    (:bspline, bspline_path(bs_cps; degree=3)),
    (:nurbs, nurbs_path(nurbs_cps; degree=3, weights=[1.0, 0.6, 1.4, 1.0])),
  ]
end

function _multi_segment_bezier()
  p0 = xyz(0, 2, 0)
  p1 = xyz(1, 2.5, 0.1)
  p2 = xyz(2, 2, 0)
  p3 = xyz(3, 1.5, 0.1)
  bezier_path(BezierSegment[BezierSegment([p0, p0 + vx(0.4), p1 - vx(0.4), p1]),
                            BezierSegment([p1, p1 + vx(0.4), p2 - vx(0.4), p2]),
                            BezierSegment([p2, p2 + vx(0.4), p3 - vx(0.4), p3])])
end

function _surface_cases()
  pts = [xyz(i, j, 0.12 * sin(i + j)) for i in 0:3, j in 0:3]
  nurbs_weights = [1.0 + 0.15 * ((i + j) % 2) for i in axes(pts, 1), j in axes(pts, 2)]
  [
    (:bezier_surface, bezier_surface(pts)),
    (:bspline_surface, bspline_surface(pts; degree_u=3, degree_v=3)),
    (:nurbs_surface, nurbs_surface(pts; degree_u=3, degree_v=3, weights=nurbs_weights)),
  ]
end

function run_exact_geometry_smoke_tests(b; verify_samples::Bool=true,
                                        verify_surface_samples::Bool=true,
                                        atol::Real=1e-5)
  mat = _smoke_mat(b)

  @testset "Exact curve realization" begin
    for (name, path) in _curve_cases()
      @testset "$(name)" begin
        ref = KhepriBase.b_stroke(b, path, mat)
        @test ref != KhepriBase.void_ref(b)
        verify_samples && _curve_samples_match(b, ref, path; atol=atol)
      end
    end

    @testset "multi-segment Bezier" begin
      path = _multi_segment_bezier()
      ref = KhepriBase.b_stroke(b, path, mat)
      @test ref != KhepriBase.void_ref(b)
    end
  end

  @testset "Exact surface realization" begin
    for (name, surface) in _surface_cases()
      @testset "$(name)" begin
        ref = KhepriBase.b_surface(b, surface, mat)
        @test ref != KhepriBase.void_ref(b)
        verify_surface_samples && _surface_samples_match(b, ref, surface; atol=atol)
      end
    end
  end
end

end
