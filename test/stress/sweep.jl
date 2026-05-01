# stress/sweep.jl — combinatorial coverage for sweep operations.
#
# Axes:
#   - path (open curve to follow): line, arc, spline, polygonal, sequence
#   - profile (closed curve to extrude along path): circle, rectangle,
#     regular_polygon, closed_spline, region-with-hole
#   - rotation (twist along path): {0, π, 4π}
#   - scale (taper along path): {1.0, 0.3, 2.0}

stress_sweep(b, reset!, verify) =
  @testset "Sweep" begin
    reset!()
    slot = Slot(:sweep, 0, 0)

    # Helper: sweep paths (open curves)
    sweep_paths = [
      ("line",       () -> open_polygonal_path([u0(), xyz(0,0,15)])),
      ("arc",        () -> arc_path(u0(), 5.0, 0.0, π/2)),
      ("spline",     () -> open_spline_path(
                            [u0(), xyz(2,2,5), xyz(0,4,10), xyz(-2,2,15)])),
      ("polygonal",  () -> open_polygonal_path(
                            [u0(), xyz(3,0,0), xyz(3,0,8), xyz(0,0,15)])),
      ("sequence",   () -> open_path_sequence(
                            arc_path(u0(), 3.0, -π/2, π/2),
                            open_polygonal_path([xyz(3,0,0), xyz(3,0,10)]))),
    ]

    # Helper: sweep profiles (closed curves)
    sweep_profiles = [
      ("circle",       () -> circular_path(u0(), 0.8)),
      ("rectangle",    () -> rectangular_path(xyz(-0.6, -0.4, 0), 1.2, 0.8)),
      ("hexagon",      () -> closed_polygonal_path(
                              [xyz(0.8cos(2π*i/6), 0.8sin(2π*i/6), 0) for i in 0:5])),
      ("closed_spline",() -> closed_spline_path(
                              [xyz(cos(2π*i/8) + 0.2sin(4π*i/8),
                                   sin(2π*i/8), 0) for i in 0:7])),
    ]

    # ── Baseline: 5 paths × 4 profiles, rotation=0, scale=1 ────────────
    for (pname, pfn) in sweep_paths
      for (profname, proffn) in sweep_profiles
        run_one_test(b, slot, "sweep_$(pname)_$(profname)",
          () -> sweep(pfn(), proffn()),
          nothing,  # sweep envelope depends on path geometry; skip strict check
          verify)
      end
    end

    # ── Rotation/scale variants on one representative pair ────────────
    for (rot_label, rot, scale_label, scl) in
        (("rot=0_scl=1", 0.0, "rot=0_scl=1", 1.0),
         ("rot=pi", π, "scl=1", 1.0),
         ("rot=4pi", 4π, "scl=1", 1.0),
         ("rot=0", 0.0, "scl=0.3", 0.3),
         ("rot=0", 0.0, "scl=2.0", 2.0),
         ("rot=pi_scl=0.5", π, "scl=0.5", 0.5))
      run_one_test(b, slot, "sweep_helix_$(rot_label)_$(scale_label)",
        () -> sweep(open_spline_path([u0(), xyz(2,2,5), xyz(0,4,10), xyz(-2,2,15)]),
                    circular_path(u0(), 0.8),
                    rot, scl),
        nothing,
        verify)
    end

    # ── Region profile (sweep through a hole) ──────────────────────────
    run_one_test(b, slot, "sweep_region_with_hole",
      () -> sweep(open_polygonal_path([u0(), xyz(0,0,15)]),
                  region(circular_path(u0(), 1.5),
                         circular_path(u0(), 0.5))),
      nothing,
      verify)

    # ── Expanded coverage ────────────────────────────────────────────

    # Sweep with negative rotation (CW twist).
    run_one_test(b, slot, "sweep_helix_neg_rot",
      () -> sweep(open_spline_path([u0(), xyz(2,2,5), xyz(0,4,10), xyz(-2,2,15)]),
                  rectangular_path(xyz(-0.6, -0.4, 0), 1.2, 0.8),
                  -π, 1.0),
      nothing,
      verify)

    # Sweep with very large twist (4π = 2 full rotations).
    run_one_test(b, slot, "sweep_4pi_rot",
      () -> sweep(open_polygonal_path([u0(), xyz(0,0,30)]),
                  rectangular_path(xyz(-0.5, -0.3, 0), 1.0, 0.6),
                  4π, 1.0),
      nothing,
      verify)

    # Sweep on a large arc-shaped 2D path.
    run_one_test(b, slot, "sweep_arc_path_circle_profile",
      () -> sweep(arc_path(u0(), 10.0, 0.0, π), circular_path(u0(), 0.5)),
      nothing,
      verify)

    # Sweep with shrinking scale (thin tip).
    run_one_test(b, slot, "sweep_helix_shrink",
      () -> sweep(open_spline_path([u0(), xyz(0,0,5), xyz(0,0,10), xyz(0,0,15)]),
                  circular_path(u0(), 1.0),
                  0.0, 0.1),
      nothing,
      verify)

    # Sweep with growing scale (cone-like).
    run_one_test(b, slot, "sweep_growing",
      () -> sweep(open_polygonal_path([u0(), xyz(0,0,12)]),
                  circular_path(u0(), 0.3),
                  0.0, 6.0),
      nothing,
      verify)

    # ── Round 3 expansion ───────────────────────────────────────────

    # Sweep along a horizontal arc path.
    run_one_test(b, slot, "sweep_horizontal_arc_path",
      () -> sweep(arc_path(u0(), 8.0, 0.0, π),
                  circular_path(u0(), 0.4)),
      nothing,
      verify)

    # Sweep along an S-curve spline path.
    run_one_test(b, slot, "sweep_s_curve",
      () -> sweep(open_spline_path([u0(), xyz(2,3,4), xyz(4,-3,8),
                                    xyz(2,3,12), xyz(0,0,16)]),
                  rectangular_path(xyz(-0.4,-0.2,0), 0.8, 0.4)),
      nothing,
      verify)

    # Sweep with combined large twist and shrink.
    run_one_test(b, slot, "sweep_4pi_shrink",
      () -> sweep(open_polygonal_path([u0(), xyz(0,0,20)]),
                  rectangular_path(xyz(-1.0,-0.5,0), 2.0, 1.0),
                  4π, 0.2),
      nothing,
      verify)

    # Sweep with closed sequence path (loop).
    run_one_test(b, slot, "sweep_closed_loop_path",
      () -> sweep(closed_path_sequence(
                    arc_path(u0(), 5.0, 0.0, π),
                    open_polygonal_path([xyz(-5,0,0), xyz(-5,-3,0),
                                         xyz(5,-3,0), xyz(5,0,0)])),
                  circular_path(u0(), 0.3)),
      nothing,
      verify)
  end
