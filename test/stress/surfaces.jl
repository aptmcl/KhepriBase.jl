# stress/surfaces.jl — combinatorial coverage for surface (2D) primitives.
#
# Axes:
#   - surface_circle/_arc/_rectangle/_polygon/_regular_polygon/_ellipse/_ring:
#     same radius/center/angle variations as curves
#   - surface_polygon: vertex counts (3,5,8), convex/concave, planar-tilted
#   - surface(c1, c2, ...) frontier from N curves
#   - surface_path(p) for each closed-path constructor
#   - region(outer, holes...) with 0, 1, 2, 3 holes
#   - surface_grid: small/medium/large × closed_u/v × smooth_u/v
#   - surface_mesh: quad-only and mixed quad/triangle faces

stress_surfaces(b, reset!, verify) =
  @testset "Surfaces" begin
    reset!()
    slot = Slot(:surfaces, 0, 0)

    # ── surface_circle: radius span ────────────────────────────────────
    for r in (0.1, 1.0, 5.0, 50.0)
      run_one_test(b, slot, "surface_circle_r=$r",
        () -> surface_circle(u0(), r),
        (-r, r, -r, r, 0.0, 0.0),
        verify)
    end

    # ── surface_arc: closed pie sector at varied angles ────────────────
    # Envelope omitted: backends with native b_surface_arc keep the SurfaceArc
    # proxy whose shape_locs returns the full-circle bbox; backends without it
    # fall back through chains that realize as SurfacePolygon with vertices
    # on the actual arc, giving a tighter bbox. Both are valid realizations.
    for (label, sa, amp) in (("quarter", 0.0, π/2), ("half", 0.0, π),
                             ("offset", π/4, 3π/2))
      r = 5.0
      run_one_test(b, slot, "surface_arc_$label",
        () -> surface_arc(u0(), r, sa, amp),
        nothing,
        verify)
    end

    # ── surface_rectangle: both forms × signed deltas ──────────────────
    run_one_test(b, slot, "surface_rect_dxdy",
      () -> surface_rectangle(u0(), 10.0, 5.0),
      (0.0, 10.0, 0.0, 5.0, 0.0, 0.0),
      verify)
    run_one_test(b, slot, "surface_rect_corners",
      () -> surface_rectangle(u0(), xyz(8, 4, 0)),
      (0.0, 8.0, 0.0, 4.0, 0.0, 0.0),
      verify)
    run_one_test(b, slot, "surface_rect_neg_dx",
      () -> surface_rectangle(xyz(5, 0, 0), -5.0, 5.0),
      (0.0, 5.0, 0.0, 5.0, 0.0, 0.0),
      verify)

    # ── surface_polygon: 3, 5, 8 verts × convex/concave ────────────────
    run_one_test(b, slot, "surface_polygon_3",
      () -> surface_polygon([u0(), xyz(5, 0, 0), xyz(2.5, 5, 0)]),
      (0.0, 5.0, 0.0, 5.0, 0.0, 0.0),
      verify)
    # Pentagon vertices don't reach ±5 on both axes; observed bbox computed
    # from the actual vertices placed at 2π·i/5 angles.
    run_one_test(b, slot, "surface_polygon_5",
      () -> surface_polygon([xyz(5cos(2π*i/5), 5sin(2π*i/5), 0) for i in 0:4]),
      let pts = [(5cos(2π*i/5), 5sin(2π*i/5)) for i in 0:4]
        (minimum(p[1] for p in pts), maximum(p[1] for p in pts),
         minimum(p[2] for p in pts), maximum(p[2] for p in pts), 0.0, 0.0)
      end,
      verify)
    run_one_test(b, slot, "surface_polygon_8",
      () -> surface_polygon([xyz(5cos(2π*i/8), 5sin(2π*i/8), 0) for i in 0:7]),
      (-5.0, 5.0, -5.0, 5.0, 0.0, 0.0),
      verify)
    run_one_test(b, slot, "surface_polygon_concave_L",
      () -> surface_polygon([xyz(0,0,0), xyz(4,0,0), xyz(4,2,0),
                             xyz(2,2,0), xyz(2,4,0), xyz(0,4,0)]),
      (0.0, 4.0, 0.0, 4.0, 0.0, 0.0),
      verify)
    run_one_test(b, slot, "surface_polygon_planar_tilted",
      () -> surface_polygon([xyz(0,0,0), xyz(5,0,0), xyz(5,3,3), xyz(0,3,3)]),
      (0.0, 5.0, 0.0, 3.0, 0.0, 3.0),
      verify)

    # ── surface_regular_polygon: edges × inscribed ─────────────────────
    for n in (3, 4, 5, 6, 8)
      r = 5.0
      for inscribed in (true, false)
        run_one_test(b, slot, "surface_reg_polygon_n=$(n)_$(inscribed ? "inscr" : "circ")",
          () -> surface_regular_polygon(n, u0(), r, 0.0, inscribed),
          (-r, r, -r, r, 0.0, 0.0),
          verify)
      end
    end

    # ── surface_ellipse: aspect ratios ─────────────────────────────────
    for (rx, ry) in ((1.0, 1.0), (5.0, 2.0), (2.0, 5.0))
      run_one_test(b, slot, "surface_ellipse_rx=$(rx)_ry=$(ry)",
        () -> surface_ellipse(u0(), rx, ry),
        (-rx, rx, -ry, ry, 0.0, 0.0),
        verify)
    end

    # ── surface_ring: hole radii ───────────────────────────────────────
    for (ri, ro) in ((1.0, 5.0), (3.0, 5.0), (4.5, 5.0))
      run_one_test(b, slot, "surface_ring_$(ri)_$(ro)",
        () -> surface_ring(u0(), ri, ro),
        (-ro, ro, -ro, ro, 0.0, 0.0),
        verify)
    end

    # ── surface(c1, c2, ...): close N curves into a frontier ───────────
    # Two arcs (top/bottom semicircle) → a circle
    run_one_test(b, slot, "surface_2_arcs",
      () -> surface(arc(u0(), 5.0, 0.0, π), arc(u0(), 5.0, π, π)),
      (-5.0, 5.0, -5.0, 5.0, 0.0, 0.0),
      verify)
    # Four lines forming a rectangle
    run_one_test(b, slot, "surface_4_lines",
      () -> surface(line(u0(), xyz(5,0,0)), line(xyz(5,0,0), xyz(5,3,0)),
                    line(xyz(5,3,0), xyz(0,3,0)), line(xyz(0,3,0), u0())),
      (0.0, 5.0, 0.0, 3.0, 0.0, 0.0),
      verify)

    # ── surface_path(p) over each closed-path constructor ──────────────
    run_one_test(b, slot, "surface_path_circular",
      () -> surface_path(circular_path(u0(), 5.0)),
      (-5.0, 5.0, -5.0, 5.0, 0.0, 0.0),
      verify)
    run_one_test(b, slot, "surface_path_rectangular",
      () -> surface_path(rectangular_path(u0(), 10.0, 5.0)),
      (0.0, 10.0, 0.0, 5.0, 0.0, 0.0),
      verify)
    run_one_test(b, slot, "surface_path_closed_polygonal",
      () -> surface_path(closed_polygonal_path(
              [u0(), xyz(5,0,0), xyz(5,5,0), xyz(0,5,0)])),
      (0.0, 5.0, 0.0, 5.0, 0.0, 0.0),
      verify)
    run_one_test(b, slot, "surface_path_closed_spline",
      () -> surface_path(closed_spline_path(
              [xyz(5cos(2π*i/6), 5sin(2π*i/6), 0) for i in 0:5])),
      (-5.0, 5.0, -5.0, 5.0, 0.0, 0.0),
      verify)
    run_one_test(b, slot, "surface_path_elliptic",
      () -> surface_path(elliptic_path(u0(), 5.0, 2.0)),
      (-5.0, 5.0, -2.0, 2.0, 0.0, 0.0),
      verify)

    # NOTE: region(outer, holes...) does not feed `surface(...)` directly —
    # the high-level `surface(...)` only accepts `Shapes1D`. Region-with-holes
    # is consumed by `extrusion(::Region, v)`; tested in stress/extrusion.jl.

    # ── surface_grid: sizes × closed_u/v × smooth_u/v ─────────────────
    # Helper: build a small 3x3 wavy grid
    grid_3x3() = [xyz(i, j, 0.5*sin(i)*cos(j)) for i in 0:2, j in 0:2]
    grid_5x5() = [xyz(i, j, 0.3*sin(i)*cos(j)) for i in 0:4, j in 0:4]
    for (size_label, gen) in (("3x3", grid_3x3), ("5x5", grid_5x5))
      for (cu, cv) in ((false, false), (true, false), (false, true))
        for (su, sv) in ((false, false), (true, true))
          name = "surface_grid_$(size_label)_cu=$(cu)_cv=$(cv)_su=$(su)_sv=$(sv)"
          run_one_test(b, slot, name,
            () -> surface_grid(gen(), cu, cv, su, sv),
            nothing,  # surface_grid envelope is mesh-dependent; skip strict check
            verify)
        end
      end
    end

    # ── surface_mesh: quad faces ───────────────────────────────────────
    run_one_test(b, slot, "surface_mesh_quad",
      () -> surface_mesh(
              [xyz(0,0,0), xyz(5,0,0), xyz(5,5,0), xyz(0,5,0), xyz(2.5,2.5,2)],
              [[1,2,3,4], [1,2,5], [2,3,5], [3,4,5], [4,1,5]]),
      nothing,
      verify)

    # ── Expanded coverage ──────────────────────────────────────────────

    # Surface circle on a rotated CS — exercises CS-aware surface plane
    # derivation. Backends that hard-code Vector3d.ZAxis would fail here.
    for (label, ϕ) in (("rot_pi4", π/4), ("rot_pi3", π/3))
      run_one_test(b, slot, "surface_circle_oblique_$label",
        () -> surface_circle(loc_from_o_phi(u0(), ϕ), 3.0),
        nothing,
        verify)
    end

    # Surface arc covering the full circle (Δα = 2π) — should give a disc.
    run_one_test(b, slot, "surface_arc_full_circle",
      () -> surface_arc(u0(), 5.0, 0.0, 2π),
      nothing,
      verify)
    # Negative-amplitude arc (CW direction).
    run_one_test(b, slot, "surface_arc_neg_amplitude",
      () -> surface_arc(u0(), 5.0, 0.0, -π/2),
      nothing,
      verify)

    # Surface frontiers built from 4 connected curves (line+arc+line+arc).
    run_one_test(b, slot, "surface_4_curves_mixed",
      () -> surface(line(u0(), xyz(5,0,0)),
                    arc(xyz(5,0.5,0), 0.5, -π/2, π),
                    line(xyz(5,1,0), xyz(0,1,0)),
                    arc(xyz(0,0.5,0), 0.5, π/2, π)),
      nothing,
      verify)

    # Surface from a path_sequence — exercises the closed-sequence stroke
    # path that revealed the b_stroke_unite issue earlier.
    run_one_test(b, slot, "surface_path_closed_sequence",
      () -> surface_path(closed_path_sequence(
              arc_path(u0(), 3.0, 0.0, π),
              open_polygonal_path([xyz(-3,0,0), xyz(-3,-1,0), xyz(3,-1,0), xyz(3,0,0)]))),
      nothing,
      verify)

    # Larger / irregular surface_grid configurations.
    grid_8x8_wave() = [xyz(i, j, 0.4*sin(i*0.7)*cos(j*0.7)) for i in 0:7, j in 0:7]
    grid_3x10()   = [xyz(i, j, 0.0) for i in 0:2, j in 0:9]
    for (label, gen, cu, cv, su, sv) in (("8x8_smooth", grid_8x8_wave, false, false, true, true),
                                          ("8x8_closed_u_smooth", grid_8x8_wave, true, false, true, true),
                                          ("3x10_flat", grid_3x10, false, false, false, false))
      run_one_test(b, slot, "surface_grid_extra_$label",
        () -> surface_grid(gen(), cu, cv, su, sv),
        nothing,
        verify)
    end

    # Surface_polygon with reverse winding (CW) — some backends invert
    # normals here.
    run_one_test(b, slot, "surface_polygon_reverse_winding",
      () -> surface_polygon([u0(), xyz(0,5,0), xyz(5,5,0), xyz(5,0,0)]),
      (0.0, 5.0, 0.0, 5.0, 0.0, 0.0),
      verify)
    # 20-vertex regular polygon as a surface.
    run_one_test(b, slot, "surface_polygon_20",
      () -> surface_polygon([xyz(5cos(2π*i/20), 5sin(2π*i/20), 0) for i in 0:19]),
      (-5.0, 5.0, -5.0, 5.0, 0.0, 0.0),
      verify)
  end
