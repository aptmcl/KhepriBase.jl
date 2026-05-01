# stress/revolve.jl — combinatorial coverage for revolve operations.
#
# Axes:
#   - profile: point (degenerate), line, arc, spline, region
#   - axis: 4 variants (vertical at origin, vertical offset, horizontal,
#     oblique). Backends route 0D/1D/2D revolves to different b_revolved_*.
#   - amplitude: {π/2, π, 3π/2, 2π}
#   - non-zero start_angle (catches the AutoCAD sign-flip in AutoCAD.jl:861)

stress_revolve(b, reset!, verify) =
  @testset "Revolve" begin
    reset!()
    slot = Slot(:revolve, 0, 0)

    # Profiles for revolution around the z-axis must lie in a *meridional*
    # plane (one containing the z-axis) and have x ≥ 0, otherwise the
    # generated surface self-intersects and AutoCAD rejects with eInvalidInput.
    # The arc here is constructed in the xz-plane (CS normal = +Y), not the
    # default horizontal-plane CS that arc()'s default would yield.
    profiles = [
      ("point",  () -> point(xyz(5, 0, 0))),
      ("line",   () -> line([xyz(5, 0, 0), xyz(5, 0, 10)])),
      ("arc",    () -> arc(loc_from_o_vz(xyz(5, 0, 5), vy(1)), 2.0, 0.0, π)),
      ("spline", () -> spline([xyz(5,0,0), xyz(7,0,3), xyz(6,0,7), xyz(8,0,10)])),
      ("region", () -> region(closed_polygonal_path(
                                [xyz(5,0,0), xyz(8,0,0), xyz(8,0,3), xyz(5,0,3)]))),
    ]

    for (pname, pfn) in profiles
      run_one_test(b, slot, "revolve_$(pname)_full",
        () -> revolve(pfn(), u0(), vz(1), 0.0, 2π),
        nothing,
        verify)
    end

    # ── 4 axes on one representative profile ───────────────────────────
    axes = [
      ("z_origin",  u0(),         vz(1)),
      ("z_offset",  xyz(2, 0, 0), vz(1)),
      ("x_axis",    u0(),         vx(1)),
      ("oblique",   u0(),         vxyz(1, 1, 1)),
    ]
    for (aname, p, n) in axes
      run_one_test(b, slot, "revolve_arc_axis=$(aname)",
        # arc in xz-plane to give the revolve a valid meridional profile
        () -> revolve(arc(loc_from_o_vz(xyz(5, 0, 5), vy(1)), 2.0, 0.0, π),
                      p, n, 0.0, 2π),
        nothing,
        verify)
    end

    # ── Amplitude variants on one representative profile ───────────────
    for (alabel, amp) in (("quarter", π/2), ("half", π),
                          ("three_quarter", 3π/2), ("full", 2π))
      run_one_test(b, slot, "revolve_line_amp=$(alabel)",
        () -> revolve(line([xyz(5, 0, 0), xyz(5, 0, 10)]), u0(), vz(1), 0.0, amp),
        nothing,
        verify)
    end

    # ── Non-zero start angle (exercises AutoCAD sign-flip path) ────────
    run_one_test(b, slot, "revolve_line_start_pi4",
      () -> revolve(line([xyz(5, 0, 0), xyz(5, 0, 10)]), u0(), vz(1), π/4, π),
      nothing,
      verify)
    run_one_test(b, slot, "revolve_arc_start_pi3",
      () -> revolve(arc(loc_from_o_vz(xyz(5, 0, 5), vy(1)), 2.0, 0.0, π),
                    u0(), vz(1), π/3, 3π/2),
      nothing,
      verify)

    # ── Expanded coverage ────────────────────────────────────────────

    # Negative amplitude (CW revolution).
    run_one_test(b, slot, "revolve_line_neg_amplitude",
      () -> revolve(line([xyz(5, 0, 0), xyz(5, 0, 10)]),
                    u0(), vz(1), 0.0, -π),
      nothing,
      verify)

    # Very small amplitude (thin slice).
    run_one_test(b, slot, "revolve_line_tiny_amplitude",
      () -> revolve(line([xyz(5, 0, 0), xyz(5, 0, 10)]),
                    u0(), vz(1), 0.0, 0.05),
      nothing,
      verify)

    # Profile far from the axis (large radius revolution).
    run_one_test(b, slot, "revolve_line_far_from_axis",
      () -> revolve(line([xyz(20, 0, 0), xyz(20, 0, 5)]),
                    u0(), vz(1), 0.0, 2π),
      nothing,
      verify)

    # Spline profile at varied positions.
    run_one_test(b, slot, "revolve_spline_offset",
      () -> revolve(spline([xyz(8,0,0), xyz(10,0,3), xyz(9,0,7), xyz(11,0,10)]),
                    u0(), vz(1), 0.0, 2π),
      nothing,
      verify)

    # Region with a hole, revolved around z-axis (annular cross-section).
    run_one_test(b, slot, "revolve_region_with_hole",
      () -> revolve(region(closed_polygonal_path(
                              [xyz(5,0,0), xyz(7,0,0), xyz(7,0,3), xyz(5,0,3)]),
                           closed_polygonal_path(
                              [xyz(5.5,0,0.5), xyz(6.5,0,0.5),
                               xyz(6.5,0,2.5), xyz(5.5,0,2.5)])),
                    u0(), vz(1), 0.0, 2π),
      nothing,
      verify)

    # Combined start_angle + amplitude variations.
    for (label, sa, amp) in (("sa_pi6_amp_pi", π/6, π),
                             ("sa_negpi4_amp_2pi", -π/4, 2π),
                             ("sa_pi_amp_pi2", π, π/2))
      run_one_test(b, slot, "revolve_line_$label",
        () -> revolve(line([xyz(5, 0, 0), xyz(5, 0, 8)]),
                      u0(), vz(1), sa, amp),
        nothing,
        verify)
    end
  end
