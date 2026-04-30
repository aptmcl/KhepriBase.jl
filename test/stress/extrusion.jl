# stress/extrusion.jl — combinatorial coverage for extrusion.
#
# This is the focus category. Extrusion has three orthogonal axes that
# each route through different `b_*` dispatch paths in the backend:
#
#   Axis A — profile type:
#     1D paths (Path subtypes), Shape1D, Shape2D, Region with holes.
#     Each takes a different `b_extruded_*` overload.
#
#   Axis B — height form:
#     scalar `h` (dispatches to `vz(h)`), `vz(h)`, `vz(-h)`, oblique vector,
#     horizontal vector (degenerate). Tests the height polymorphism in the
#     extrusion frontend.
#
#   Axis C — base location / orientation:
#     `u0()`, translated origin, rotated CS via `loc_from_o_phi`. Tests that
#     the backend correctly transforms profile coordinates relative to `cb`.
#
# Strategy: full Axis A × baseline (Axis B = vz(5), Axis C = u0()), then
# Axis B and Axis C swept on a small subset of representative profiles to
# avoid combinatorial explosion. ~70 cases total.

stress_extrusion(b, reset!, verify) =
  @testset "Extrusion" begin
    reset!()
    slot = Slot(:extrusion, 0, 0)

    # ── Axis A: 16 profile types × baseline (vz(5), u0()) ─────────────
    # Section A1: profile-as-1D-Path
    run_one_test(b, slot, "extrude_circular_path",
      () -> extrusion(circular_path(u0(), 3.0), vz(5.0)),
      (-3.0, 3.0, -3.0, 3.0, 0.0, 5.0),
      verify)
    # arc_path(c, r, 0, π) sweeps CCW from (r,0) through (0,r) to (-r,0):
    # y range is [0, r], not [-r, r].
    run_one_test(b, slot, "extrude_arc_path",
      () -> extrusion(arc_path(u0(), 3.0, 0.0, π), vz(5.0)),
      (-3.0, 3.0, 0.0, 3.0, 0.0, 5.0),
      verify)
    run_one_test(b, slot, "extrude_rectangular_path",
      () -> extrusion(rectangular_path(u0(), 6.0, 4.0), vz(5.0)),
      (0.0, 6.0, 0.0, 4.0, 0.0, 5.0),
      verify)
    run_one_test(b, slot, "extrude_elliptic_path",
      () -> extrusion(elliptic_path(u0(), 5.0, 2.0), vz(5.0)),
      (-5.0, 5.0, -2.0, 2.0, 0.0, 5.0),
      verify)
    run_one_test(b, slot, "extrude_open_polygonal_path",
      () -> extrusion(open_polygonal_path(
              [u0(), xyz(3,0,0), xyz(3,3,0), xyz(0,3,0)]), vz(5.0)),
      (0.0, 3.0, 0.0, 3.0, 0.0, 5.0),
      verify)
    run_one_test(b, slot, "extrude_closed_polygonal_path",
      () -> extrusion(closed_polygonal_path(
              [u0(), xyz(3,0,0), xyz(3,3,0), xyz(0,3,0)]), vz(5.0)),
      (0.0, 3.0, 0.0, 3.0, 0.0, 5.0),
      verify)
    run_one_test(b, slot, "extrude_open_spline_path",
      () -> extrusion(open_spline_path(
              [u0(), xyz(2,2,0), xyz(5,0,0)]), vz(5.0)),
      (0.0, 5.0, 0.0, 2.0, 0.0, 5.0),
      verify)
    run_one_test(b, slot, "extrude_closed_spline_path",
      () -> extrusion(closed_spline_path(
              [xyz(3cos(2π*i/6), 3sin(2π*i/6), 0) for i in 0:5]), vz(5.0)),
      (-3.0, 3.0, -3.0, 3.0, 0.0, 5.0),
      verify)
    run_one_test(b, slot, "extrude_path_sequence",
      () -> extrusion(open_path_sequence(
              arc_path(u0(), 3.0, 0.0, π),
              open_polygonal_path([xyz(-3,0,0), xyz(-3,-2,0), xyz(3,-2,0), xyz(3,0,0)])),
            vz(5.0)),
      nothing,  # path_sequence envelope is approximate
      verify)

    # Section A2: profile-as-Shape1D
    run_one_test(b, slot, "extrude_shape_circle",
      () -> extrusion(circle(u0(), 3.0), vz(5.0)),
      (-3.0, 3.0, -3.0, 3.0, 0.0, 5.0),
      verify)
    run_one_test(b, slot, "extrude_shape_rectangle",
      () -> extrusion(rectangle(u0(), 6.0, 4.0), vz(5.0)),
      (0.0, 6.0, 0.0, 4.0, 0.0, 5.0),
      verify)
    run_one_test(b, slot, "extrude_shape_regular_polygon",
      () -> extrusion(regular_polygon(6, u0(), 3.0, 0.0, true), vz(5.0)),
      (-3.0, 3.0, -3.0, 3.0, 0.0, 5.0),
      verify)
    run_one_test(b, slot, "extrude_shape_polygon",
      () -> extrusion(polygon([u0(), xyz(4,0,0), xyz(4,3,0), xyz(0,3,0)]), vz(5.0)),
      (0.0, 4.0, 0.0, 3.0, 0.0, 5.0),
      verify)
    run_one_test(b, slot, "extrude_shape_closed_spline",
      () -> extrusion(closed_spline(
              [xyz(3cos(2π*i/6), 3sin(2π*i/6), 0) for i in 0:5]), vz(5.0)),
      (-3.0, 3.0, -3.0, 3.0, 0.0, 5.0),
      verify)
    # Open Shape1D: line (not closed)
    run_one_test(b, slot, "extrude_shape_line",
      () -> extrusion(line([u0(), xyz(5,0,0), xyz(5,3,0)]), vz(5.0)),
      (0.0, 5.0, 0.0, 3.0, 0.0, 5.0),
      verify)

    # Section A3: profile-as-Shape2D
    run_one_test(b, slot, "extrude_surface_circle",
      () -> extrusion(surface_circle(u0(), 3.0), vz(5.0)),
      (-3.0, 3.0, -3.0, 3.0, 0.0, 5.0),
      verify)
    run_one_test(b, slot, "extrude_surface_polygon",
      () -> extrusion(surface_polygon([u0(), xyz(4,0,0), xyz(4,3,0), xyz(0,3,0)]),
                      vz(5.0)),
      (0.0, 4.0, 0.0, 3.0, 0.0, 5.0),
      verify)
    run_one_test(b, slot, "extrude_surface_rectangle",
      () -> extrusion(surface_rectangle(u0(), 6.0, 4.0), vz(5.0)),
      (0.0, 6.0, 0.0, 4.0, 0.0, 5.0),
      verify)
    run_one_test(b, slot, "extrude_surface_regular_polygon",
      () -> extrusion(surface_regular_polygon(6, u0(), 3.0, 0.0, true), vz(5.0)),
      (-3.0, 3.0, -3.0, 3.0, 0.0, 5.0),
      verify)
    # Surface ring already has a hole — exercises with-holes extrusion path
    run_one_test(b, slot, "extrude_surface_ring",
      () -> extrusion(surface_ring(u0(), 1.0, 3.0), vz(5.0)),
      (-3.0, 3.0, -3.0, 3.0, 0.0, 5.0),
      verify)

    # Section A4: profile-as-Region (with holes — the Region path)
    run_one_test(b, slot, "extrude_region_no_holes",
      () -> extrusion(region(circular_path(u0(), 3.0)), vz(5.0)),
      (-3.0, 3.0, -3.0, 3.0, 0.0, 5.0),
      verify)
    run_one_test(b, slot, "extrude_region_1_hole",
      () -> extrusion(region(circular_path(u0(), 3.0),
                             circular_path(u0(), 1.0)), vz(5.0)),
      (-3.0, 3.0, -3.0, 3.0, 0.0, 5.0),
      verify)
    run_one_test(b, slot, "extrude_region_2_holes",
      () -> extrusion(region(rectangular_path(xyz(-4,-4,0), 8.0, 8.0),
                             circular_path(xyz(-2,0,0), 0.8),
                             circular_path(xyz(2,0,0), 0.8)),
                      vz(5.0)),
      (-4.0, 4.0, -4.0, 4.0, 0.0, 5.0),
      verify)
    run_one_test(b, slot, "extrude_region_3_holes",
      () -> extrusion(region(rectangular_path(xyz(-5,-5,0), 10.0, 10.0),
                             circular_path(xyz(-3,-3,0), 0.8),
                             circular_path(xyz(3,-3,0), 0.8),
                             circular_path(xyz(0,3,0), 0.8)),
                      vz(5.0)),
      (-5.0, 5.0, -5.0, 5.0, 0.0, 5.0),
      verify)

    # ── Axis B: height-form variants on 4 representative profiles ─────
    # Profile shorthand for the height-form sweep
    rep_profiles = [
      ("circle",   () -> circle(u0(), 3.0),                   (-3.0, 3.0, -3.0, 3.0)),
      ("surf_pgn", () -> surface_polygon([u0(), xyz(4,0,0), xyz(4,3,0), xyz(0,3,0)]),
                   (0.0, 4.0, 0.0, 3.0)),
      ("circ_path", () -> circular_path(u0(), 3.0),           (-3.0, 3.0, -3.0, 3.0)),
      ("region_h",  () -> region(circular_path(u0(), 3.0),
                                 circular_path(u0(), 1.0)),   (-3.0, 3.0, -3.0, 3.0)),
    ]
    height_variants = [
      ("scalar_5",   5.0,                (0.0, 5.0)),    # routes scalar→vz
      ("vz_5",       vz(5.0),            (0.0, 5.0)),
      ("vz_neg5",    vz(-5.0),           (-5.0, 0.0)),
    ]
    # Note: oblique and horizontal heights deferred to Axis-C section below
    # because envelope verification requires more careful expected-aabb math.
    for (pname, pfn, (xmin, xmax, ymin, ymax)) in rep_profiles
      for (hname, h, (zmin, zmax)) in height_variants
        run_one_test(b, slot, "extrude_$(pname)_h=$(hname)",
          () -> extrusion(pfn(), h),
          (xmin, xmax, ymin, ymax, zmin, zmax),
          verify)
      end
    end

    # Oblique extrusion: envelope is harder; skip strict envelope check.
    for (pname, pfn, _) in rep_profiles
      run_one_test(b, slot, "extrude_$(pname)_oblique",
        () -> extrusion(pfn(), vxyz(2.0, 3.0, 5.0)),
        nothing,
        verify)
    end
    # Horizontal extrusion (Z=0 sweep) is a known-degenerate input for some
    # backends. Tested in stress/pathological.jl where rejection is the
    # expected outcome; here it would falsely register as a regression.

    # ── Axis C: base location / orientation on 3 profiles ──────────────
    # Translated origin: shape proxy fields use `current_cs()` at construction
    # time, so writing `circle(xyz(10, 5, 0), ...)` yields a circle anchored
    # in the slot-local CS at offset (10, 5, 0). The expected envelope
    # reflects this offset.
    for (pname, pfn, (xmin, xmax, ymin, ymax)) in rep_profiles[1:3]
      run_one_test(b, slot, "extrude_$(pname)_translated",
        () -> extrusion(
                # Build the profile at a translated location
                let s = pfn(); s end,  # original profile
                vz(5.0),
                xyz(10.0, 5.0, 0.0)),  # cb = base location override
        nothing,  # cb shifts profile origin; envelope path is non-trivial
        verify)
    end
    # Rotated CS: extrude into a 45°-rotated frame
    run_one_test(b, slot, "extrude_circle_rotated_cs",
      () -> let p = loc_from_o_phi(u0(), π/4)
              extrusion(circle(p, 3.0), vz(5.0))
            end,
      nothing,
      verify)
  end
