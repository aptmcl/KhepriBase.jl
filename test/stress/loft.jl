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

    # ── Expanded coverage ────────────────────────────────────────────

    # 8 cross-section loft (more profile count).
    run_one_test(b, slot, "loft_8_circles",
      () -> loft([circle(xyz(0, 0, i), 2.0 + 0.5*sin(i)) for i in 0:7]),
      nothing,
      verify)

    # Loft with non-uniform Z spacing.
    run_one_test(b, slot, "loft_nonuniform_z",
      () -> loft([circle(xyz(0, 0, 0), 3.0),
                  circle(xyz(0, 0, 1), 2.5),
                  circle(xyz(0, 0, 4), 2.0),
                  circle(xyz(0, 0, 10), 1.5)]),
      nothing,
      verify)

    # Loft with translated cross-sections (spine offset, not just radius).
    run_one_test(b, slot, "loft_offset_centers",
      () -> loft([circle(xyz(0, 0, 0), 2.0),
                  circle(xyz(2, 0, 4), 2.0),
                  circle(xyz(2, 2, 8), 2.0)]),
      nothing,
      verify)

    # Loft of regular polygons of differing edge counts (heavy
    # map_division usage).
    run_one_test(b, slot, "loft_polys_3_to_8",
      () -> loft([regular_polygon(3, xyz(0,0,0), 3.0, 0.0, true),
                  regular_polygon(5, xyz(0,0,3), 3.0, 0.0, true),
                  regular_polygon(8, xyz(0,0,6), 3.0, 0.0, true)]),
      nothing,
      verify)

    # Closed loft of 4 circles arranged in a torus pattern: AutoCAD's
    # LoftedSurface.CreateLoftedSurface rejects this topology
    # (eGeneralModelingFailure). Other backends accept it via emulation.
    # Tested in stress/pathological.jl as a known-rejected input.

    # Loft with 2 rails of different shapes (one straight, one curved).
    run_one_test(b, slot, "loft_with_mixed_rails",
      () -> loft([circle(xyz(0, 0, 0), 2.0), circle(xyz(0, 0, 10), 2.0)],
                 [line([xyz(2, 0, 0), xyz(2, 0, 10)]),
                  spline([xyz(-2, 0, 0), xyz(-3, 0, 5), xyz(-2, 0, 10)])]),
      nothing,
      verify)

    # ── Round 3 expansion ───────────────────────────────────────────

    # Loft of 16 cross-sections (high count).
    run_one_test(b, slot, "loft_16_cross_sections",
      () -> loft([circle(xyz(0, 0, i), 2.0 + 0.3*sin(i*0.5)) for i in 0:15]),
      nothing,
      verify)

    # Loft of polygons with very different vertex counts (3 → 30).
    run_one_test(b, slot, "loft_polys_3_to_30",
      () -> loft([regular_polygon(3, xyz(0,0,0), 3.0, 0.0, true),
                  regular_polygon(10, xyz(0,0,4), 3.0, 0.0, true),
                  regular_polygon(30, xyz(0,0,8), 3.0, 0.0, true)]),
      nothing,
      verify)

    # Loft along a curved spine.
    run_one_test(b, slot, "loft_curved_spine",
      () -> loft([circle(xyz(0, 0, 0), 2.0),
                  circle(xyz(2, 0, 3), 2.0),
                  circle(xyz(4, 2, 6), 2.0),
                  circle(xyz(2, 4, 9), 2.0),
                  circle(xyz(0, 6, 12), 2.0)]),
      nothing,
      verify)

    # Loft mixed orientations (cross-sections in different planes).
    run_one_test(b, slot, "loft_mixed_orientations",
      () -> loft([circle(loc_from_o_phi(xyz(0,0,0), 0.0), 2.0),
                  circle(loc_from_o_phi(xyz(0,0,5), π/4), 2.0),
                  circle(loc_from_o_phi(xyz(0,0,10), π/2), 2.0)]),
      nothing,
      verify)

    # Loft of two squares of very different sizes (extreme taper).
    run_one_test(b, slot, "loft_extreme_taper",
      () -> loft([rectangle(xyz(-5, -5, 0), 10.0, 10.0),
                  rectangle(xyz(-0.1, -0.1, 10), 0.2, 0.2)]),
      nothing,
      verify)
  end
