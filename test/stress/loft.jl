# stress/loft.jl — combinatorial coverage for loft operations.
#
# Axes:
#   - profile count: 2, 3, 5, 8
#   - profile type: all curves vs all surfaces vs (curve → point) vs (surface → point)
#   - ruled vs smooth × open vs closed
#   - mismatched vertex counts (forces map_division)
#   - rails: 0, 1, 2

stress_loft(b, reset!, verify) =
  @testset "Loft" begin
    reset!()
    slot = Slot(:loft, 0, 0)

    # AutoCAD's CreateLoftedSurface requires ≥ 3 cross-sections when
    # closed=true (the loft must form a closed loop in profile-space). With
    # 2 cross-sections, only ruled/closed=false combinations are valid.
    for ruled in (true, false)
      run_one_test(b, slot,
        "loft_2_curves_ruled=$(ruled)_closed=false",
        () -> loft([circle(xyz(0, 0, 0), 3.0), circle(xyz(0, 0, 6), 3.0)],
                   Shape[], ruled, false),
        nothing,
        verify)
    end
    # Closed loft requires ≥ 3 cross-sections.
    for ruled in (true, false)
      run_one_test(b, slot,
        "loft_3_curves_ruled=$(ruled)_closed=true",
        () -> loft([circle(xyz(0, 0, 0), 3.0),
                    circle(xyz(0, 0, 4), 2.0),
                    circle(xyz(0, 0, 8), 3.0)],
                   Shape[], ruled, true),
        nothing,
        verify)
    end

    # ── 3-curve loft (varying radius) ─────────────────────────────────
    run_one_test(b, slot, "loft_3_circles",
      () -> loft([circle(xyz(0, 0, 0), 3.0),
                  circle(xyz(0, 0, 4), 1.5),
                  circle(xyz(0, 0, 8), 2.5)]),
      nothing,
      verify)

    # ── 5-curve loft ───────────────────────────────────────────────────
    run_one_test(b, slot, "loft_5_circles",
      () -> loft([circle(xyz(0, 0, 2*i), 1.0 + 0.3*i) for i in 0:4]),
      nothing,
      verify)

    # ── Mismatched vertex counts (forces map_division) ─────────────────
    run_one_test(b, slot, "loft_mismatched_polys",
      () -> loft([regular_polygon(3, xyz(0,0,0), 3.0, 0.0, true),
                  regular_polygon(7, xyz(0,0,5), 3.0, 0.0, true)]),
      nothing,
      verify)

    # ── Loft to a point (curve → apex) ─────────────────────────────────
    run_one_test(b, slot, "loft_curve_to_point",
      () -> loft([circle(xyz(0, 0, 0), 3.0), point(xyz(0, 0, 8))]),
      nothing,
      verify)

    # ── Loft surfaces (closed profiles → closed solid loft) ───────────
    run_one_test(b, slot, "loft_surfaces_3",
      () -> loft([surface_circle(xyz(0, 0, 0), 3.0),
                  surface_circle(xyz(0, 0, 4), 1.5),
                  surface_circle(xyz(0, 0, 8), 2.5)]),
      nothing,
      verify)

    # ── Loft surface to point ──────────────────────────────────────────
    run_one_test(b, slot, "loft_surface_to_point",
      () -> loft([surface_circle(xyz(0, 0, 0), 3.0), point(xyz(0, 0, 8))]),
      nothing,
      verify)

    # Loft rails are Shapes (Shape1D curves), not Paths. The cross-sections
    # must touch the rails at corresponding parameter positions.
    run_one_test(b, slot, "loft_with_1_rail",
      () -> loft([circle(xyz(0, 0, 0), 3.0), circle(xyz(0, 0, 8), 3.0)],
                 [spline([xyz(3,0,0), xyz(5,0,4), xyz(3,0,8)])]),
      nothing,
      verify)
    run_one_test(b, slot, "loft_with_2_rails",
      () -> loft([circle(xyz(0, 0, 0), 3.0), circle(xyz(0, 0, 8), 3.0)],
                 [spline([xyz(3,0,0), xyz(5,0,4), xyz(3,0,8)]),
                  spline([xyz(-3,0,0), xyz(-5,0,4), xyz(-3,0,8)])]),
      nothing,
      verify)

    # ── loft_ruled sugar ───────────────────────────────────────────────
    run_one_test(b, slot, "loft_ruled_sugar",
      () -> loft_ruled([circle(xyz(0, 0, 0), 3.0), circle(xyz(0, 0, 6), 2.0)]),
      nothing,
      verify)
  end
