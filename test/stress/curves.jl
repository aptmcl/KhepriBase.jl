# stress/curves.jl — combinatorial coverage for 1D primitives.
#
# Axes (chosen to exercise dispatch paths in the b_* layer):
#   - circle/arc/ellipse: radius span (0.01 to 100) × center variation × angles
#   - regular_polygon: edge counts {3,4,5,6,8,30} × inscribed/circumscribed
#   - polygon/closed_line/line: vertex counts, varargs vs vector form
#   - spline/closed_spline: control-point counts, optional end tangents
#   - rectangle: corner+dxdy form vs corner+corner form, with negative deltas

#=
Choice of radii: 0.01 (sub-unit, exercises numerical edge of plugin), 1.0
(typical small geometry), 5.0 (mid-range), 100.0 (large). We deliberately
avoid 1e6+ because some plugins clamp coordinates to a finite envelope and
that's a separate concern from "does the call route correctly".
=#
const CURVE_RADII = (0.01, 1.0, 5.0, 100.0)

#=
Arc angle samples cover four regimes: (0, π/4) very short arc, (0, π) half
circle (can dispatch differently in some plugins because endpoints are
diametrical), (π/4, 3π/2) start-angle non-zero plus large amplitude (some
plugins flip sign or wrap), (0, 2π - 1e-3) near-full but not closed.
=#
const ARC_ANGLES = (
  ("quarter",   0.0,    π/2),
  ("half",      0.0,    π),
  ("offset_3p4",  π/4,  3π/2),
  ("near_full", 0.0,    2π - 1e-3),
)

stress_curves(b, reset!, verify) =
  @testset "Curves" begin
    reset!()
    slot = Slot(:curves, 0, 0)

    # ── Circle: radius span ────────────────────────────────────────────
    for r in CURVE_RADII
      run_one_test(b, slot, "circle_r=$r",
        () -> circle(u0(), r),
        (-r, r, -r, r, 0.0, 0.0),
        verify)
    end

    # ── Circle: center variation (offset within slot) ──────────────────
    for (label, dx, dy, dz) in (("xy", 5.0, 3.0, 0.0),
                                ("xyz", 4.0, 4.0, 2.0),
                                ("z_only", 0.0, 0.0, 7.0))
      r = 1.0
      run_one_test(b, slot, "circle_center_$label",
        () -> circle(xyz(dx, dy, dz), r),
        (dx-r, dx+r, dy-r, dy+r, dz, dz),
        verify)
    end

    # ── Arc: angle variations ──────────────────────────────────────────
    for (label, sa, amp) in ARC_ANGLES
      r = 5.0
      run_one_test(b, slot, "arc_$label",
        () -> arc(u0(), r, sa, amp),
        (-r, r, -r, r, 0.0, 0.0),  # loose envelope; arcs are subset of circle bbox
        verify)
    end

    # ── Ellipse: aspect ratios ─────────────────────────────────────────
    for (rx, ry) in ((1.0, 1.0), (5.0, 1.0), (1.0, 5.0), (10.0, 0.1))
      run_one_test(b, slot, "ellipse_rx=$(rx)_ry=$(ry)",
        () -> ellipse(u0(), rx, ry),
        (-rx, rx, -ry, ry, 0.0, 0.0),
        verify)
    end

    # ── Elliptic arc: angles × aspect ──────────────────────────────────
    for (rx, ry) in ((5.0, 2.0), (2.0, 5.0))
      for (label, sa, amp) in ARC_ANGLES[1:2]
        run_one_test(b, slot, "elliptic_arc_$(rx)_$(ry)_$label",
          () -> elliptic_arc(u0(), rx, ry, sa, amp),
          (-rx, rx, -ry, ry, 0.0, 0.0),
          verify)
      end
    end

    # ── Regular polygon: edges × inscribed/circumscribed × angle ──────
    for n in (3, 4, 5, 6, 8, 30)
      r = 5.0
      for inscribed in (true, false)
        run_one_test(b, slot, "regular_polygon_n=$(n)_$(inscribed ? "inscr" : "circ")",
          () -> regular_polygon(n, u0(), r, 0.0, inscribed),
          (-r, r, -r, r, 0.0, 0.0),
          verify)
      end
    end
    # Non-zero start angle on a hexagon
    run_one_test(b, slot, "regular_polygon_n=6_rot=π/4",
      () -> regular_polygon(6, u0(), 5.0, π/4, true),
      (-5.0, 5.0, -5.0, 5.0, 0.0, 0.0),
      verify)

    # ── Line: 2-point, multi-point, varargs ────────────────────────────
    run_one_test(b, slot, "line_2pt",
      () -> line([u0(), xyz(10, 0, 0)]),
      (0.0, 10.0, 0.0, 0.0, 0.0, 0.0),
      verify)
    run_one_test(b, slot, "line_multi_pt",
      () -> line([u0(), xyz(5, 0, 0), xyz(5, 5, 0), xyz(0, 5, 0)]),
      (0.0, 5.0, 0.0, 5.0, 0.0, 0.0),
      verify)
    run_one_test(b, slot, "line_varargs",
      () -> line(u0(), xyz(5, 0, 0), xyz(5, 5, 0)),
      (0.0, 5.0, 0.0, 5.0, 0.0, 0.0),
      verify)
    run_one_test(b, slot, "line_3d",
      () -> line([u0(), xyz(5, 5, 5), xyz(10, 0, 10)]),
      (0.0, 10.0, 0.0, 5.0, 0.0, 10.0),
      verify)

    # ── Closed line / polygon ─────────────────────────────────────────
    run_one_test(b, slot, "closed_line_quad",
      () -> closed_line([u0(), xyz(5, 0, 0), xyz(5, 5, 0), xyz(0, 5, 0)]),
      (0.0, 5.0, 0.0, 5.0, 0.0, 0.0),
      verify)
    run_one_test(b, slot, "polygon_3",
      () -> polygon([u0(), xyz(5, 0, 0), xyz(2.5, 5, 0)]),
      (0.0, 5.0, 0.0, 5.0, 0.0, 0.0),
      verify)
    run_one_test(b, slot, "polygon_concave_6",
      () -> polygon([xyz(0,0,0), xyz(4,0,0), xyz(4,2,0),
                     xyz(2,2,0), xyz(2,4,0), xyz(0,4,0)]),
      (0.0, 4.0, 0.0, 4.0, 0.0, 0.0),
      verify)

    # ── Spline: ctrl-point counts × end tangents ──────────────────────
    run_one_test(b, slot, "spline_3",
      () -> spline([u0(), xyz(5, 5, 0), xyz(10, 0, 0)]),
      (0.0, 10.0, 0.0, 5.0, 0.0, 0.0),
      verify)
    run_one_test(b, slot, "spline_5",
      () -> spline([u0(), xyz(2,2,0), xyz(5,5,0), xyz(8,2,0), xyz(10,0,0)]),
      (0.0, 10.0, 0.0, 5.0, 0.0, 0.0),
      verify)
    run_one_test(b, slot, "spline_with_tangents",
      () -> spline([u0(), xyz(5, 5, 0), xyz(10, 0, 0)], vx(1), vx(1)),
      (0.0, 10.0, 0.0, 5.0, 0.0, 0.0),
      verify)
    run_one_test(b, slot, "spline_3d_helix_like",
      () -> spline([xyz(0,0,0), xyz(2,0,2), xyz(0,2,4), xyz(-2,0,6), xyz(0,-2,8)]),
      (-2.0, 2.0, -2.0, 2.0, 0.0, 8.0),
      verify)

    # ── Closed spline: ctrl-point counts ──────────────────────────────
    run_one_test(b, slot, "closed_spline_4",
      () -> closed_spline([xyz(5,0,0), xyz(0,5,0), xyz(-5,0,0), xyz(0,-5,0)]),
      (-5.0, 5.0, -5.0, 5.0, 0.0, 0.0),
      verify)
    run_one_test(b, slot, "closed_spline_8",
      () -> closed_spline([xyz(5cos(2π*i/8), 5sin(2π*i/8), 0) for i in 0:7]),
      (-5.0, 5.0, -5.0, 5.0, 0.0, 0.0),
      verify)

    # ── Rectangle: both constructor forms × signed deltas ─────────────
    run_one_test(b, slot, "rectangle_dxdy",
      () -> rectangle(u0(), 10.0, 5.0),
      (0.0, 10.0, 0.0, 5.0, 0.0, 0.0),
      verify)
    run_one_test(b, slot, "rectangle_corners",
      () -> rectangle(u0(), xyz(10, 5, 0)),
      (0.0, 10.0, 0.0, 5.0, 0.0, 0.0),
      verify)
    run_one_test(b, slot, "rectangle_neg_dx",
      () -> rectangle(xyz(5, 0, 0), -5.0, 5.0),
      (0.0, 5.0, 0.0, 5.0, 0.0, 0.0),
      verify)
    run_one_test(b, slot, "rectangle_neg_dy",
      () -> rectangle(u0(), 5.0, -5.0),
      (0.0, 5.0, -5.0, 0.0, 0.0, 0.0),
      verify)
  end
