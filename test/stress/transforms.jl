# stress/transforms.jl — combinatorial coverage for transformations.
#
# Axes:
#   - operation: move, rotate, scale, mirror, transform
#   - target: representative curve, surface, solid
#   - nested: move(rotate(scale(...)))

stress_transforms(b, reset!, verify) =
  @testset "Transforms" begin
    reset!()
    slot = Slot(:transforms, 0, 0)

    # Representative shapes (one per dimension class)
    targets = [
      ("circle",     () -> circle(u0(), 3.0)),
      ("rect",       () -> surface_rectangle(u0(), 6.0, 4.0)),
      ("sphere",     () -> sphere(u0(), 3.0)),
      ("box",        () -> box(xyz(-2,-2,-2), 4.0, 4.0, 4.0)),
      ("cylinder",   () -> cylinder(u0(), 2.0, 5.0)),
    ]

    # ── move ───────────────────────────────────────────────────────────
    for (tname, tfn) in targets
      run_one_test(b, slot, "move_$(tname)",
        () -> move(tfn(), vxyz(8.0, 4.0, 2.0)),
        nothing,
        verify)
    end

    # ── rotate ─────────────────────────────────────────────────────────
    for (tname, tfn) in targets
      run_one_test(b, slot, "rotate_$(tname)",
        () -> rotate(tfn(), π/4, u0(), vz(1)),
        nothing,
        verify)
    end

    # ── scale ──────────────────────────────────────────────────────────
    for (tname, tfn) in targets
      run_one_test(b, slot, "scale_$(tname)",
        () -> scale(tfn(), 1.5, u0()),
        nothing,
        verify)
    end

    # ── mirror ─────────────────────────────────────────────────────────
    for (tname, tfn) in targets[1:3]
      run_one_test(b, slot, "mirror_$(tname)",
        () -> mirror(tfn(), u0(), vx(1)),
        nothing,
        verify)
    end

    # ── transform via Loc ──────────────────────────────────────────────
    run_one_test(b, slot, "transform_box_via_loc",
      () -> transform(box(xyz(-1,-1,0), 2.0, 2.0, 4.0),
                      loc_from_o_phi(xyz(5,5,2), π/3)),
      nothing,
      verify)

    # ── Nested transformations ─────────────────────────────────────────
    run_one_test(b, slot, "nested_move_rotate_scale",
      () -> move(rotate(scale(box(xyz(-1,-1,0), 2.0, 2.0, 2.0), 1.5, u0()),
                        π/6, u0(), vz(1)),
                 vxyz(5.0, 0.0, 0.0)),
      nothing,
      verify)

    # ── Transform-then-CSG ─────────────────────────────────────────────
    run_one_test(b, slot, "subtract_after_rotate",
      () -> subtraction(box(xyz(-3,-3,0), 6.0, 6.0, 4.0),
                        rotate(cylinder(xyz(0,0,-1), 1.5, 6.0),
                               π/4, u0(), vx(1))),
      nothing,
      verify)

    # ── Expanded coverage ────────────────────────────────────────────

    # Move with various vectors.
    for (label, v) in (("vxyz", vxyz(3.0, 5.0, 2.0)),
                       ("vneg", vxyz(-2.0, -3.0, -1.0)),
                       ("vlong", vxyz(50.0, 0.0, 0.0)))
      run_one_test(b, slot, "move_sphere_$label",
        () -> move(sphere(u0(), 2.0), v),
        nothing,
        verify)
    end

    # Rotate with multiple angle values.
    for (label, θ) in (("pi6", π/6), ("pi3", π/3),
                       ("2pi3", 2π/3), ("neg_pi4", -π/4))
      run_one_test(b, slot, "rotate_box_$label",
        () -> rotate(box(xyz(-2,-2,0), 4.0, 4.0, 4.0), θ, u0(), vz(1)),
        nothing,
        verify)
    end

    # Rotate around non-z axes.
    run_one_test(b, slot, "rotate_box_x_axis",
      () -> rotate(box(xyz(-2,-2,0), 4.0, 4.0, 4.0), π/3, u0(), vx(1)),
      nothing,
      verify)
    run_one_test(b, slot, "rotate_box_y_axis",
      () -> rotate(box(xyz(-2,-2,0), 4.0, 4.0, 4.0), π/3, u0(), vy(1)),
      nothing,
      verify)
    run_one_test(b, slot, "rotate_box_oblique_axis",
      () -> rotate(box(xyz(-2,-2,0), 4.0, 4.0, 4.0), π/3, u0(), vxyz(1,1,1)),
      nothing,
      verify)

    # Scale with various factors and pivots.
    for (label, s) in (("0.25", 0.25), ("3.0", 3.0), ("10.0", 10.0))
      run_one_test(b, slot, "scale_sphere_$label",
        () -> scale(sphere(u0(), 2.0), s, u0()),
        nothing,
        verify)
    end
    # Scale with off-origin pivot — exercises pivot translation in matrix.
    run_one_test(b, slot, "scale_box_off_pivot",
      () -> scale(box(xyz(-1,-1,0), 2.0, 2.0, 2.0), 2.0, xyz(5, 0, 0)),
      nothing,
      verify)

    # Mirror across non-axis-aligned planes.
    run_one_test(b, slot, "mirror_box_oblique_plane",
      () -> mirror(box(xyz(2,2,0), 4.0, 4.0, 4.0), u0(), vxyz(1,1,0)),
      nothing,
      verify)

    # Two-level nested transforms (move(rotate(...))).
    run_one_test(b, slot, "move_then_rotate",
      () -> move(rotate(box(xyz(-1,-1,0), 2.0, 2.0, 2.0), π/4, u0(), vz(1)),
                 vxyz(5, 0, 0)),
      nothing,
      verify)

    # Three-level nested transforms (different order from existing test).
    run_one_test(b, slot, "rotate_then_move_then_scale",
      () -> scale(move(rotate(box(xyz(-1,-1,0), 2.0, 2.0, 2.0), π/4, u0(), vz(1)),
                       vxyz(5, 0, 0)),
                  1.5, u0()),
      nothing,
      verify)

    # Transform a Shape2D (surface).
    run_one_test(b, slot, "rotate_surface_polygon",
      () -> rotate(surface_polygon([u0(), xyz(5,0,0), xyz(5,5,0), xyz(0,5,0)]),
                   π/4, u0(), vy(1)),
      nothing,
      verify)

    # ── Round 3 expansion ───────────────────────────────────────────

    # Mirror across z-axis plane.
    run_one_test(b, slot, "mirror_box_xy_plane",
      () -> mirror(box(xyz(2,2,2), 4.0, 4.0, 4.0), u0(), vz(1)),
      nothing,
      verify)
    # Mirror across plane through non-origin point.
    run_one_test(b, slot, "mirror_box_offset_plane",
      () -> mirror(box(xyz(0,0,0), 4.0, 4.0, 4.0), xyz(5, 0, 0), vx(1)),
      nothing,
      verify)

    # 4-deep nested transforms (move(rotate(scale(mirror(...))))).
    run_one_test(b, slot, "deep_nested_4",
      () -> move(rotate(scale(mirror(box(u0(), 2.0, 2.0, 2.0), u0(), vx(1)),
                              1.5, u0()),
                        π/3, u0(), vz(1)),
                 vxyz(5, 5, 0)),
      nothing,
      verify)

    # Transform applied to compound CSG result.
    run_one_test(b, slot, "rotate_after_subtract",
      () -> rotate(subtraction(box(xyz(-2,-2,0), 4.0, 4.0, 4.0),
                               sphere(xyz(0,0,2), 1.5)),
                   π/4, u0(), vz(1)),
      nothing,
      verify)

    # Move + scale + rotate compound on a torus.
    run_one_test(b, slot, "move_scale_rotate_torus",
      () -> move(scale(rotate(torus(u0(), 5.0, 1.5), π/4, u0(), vx(1)),
                       0.5, u0()),
                 vxyz(10, 0, 0)),
      nothing,
      verify)

    # Multiple successive moves (translation chain).
    run_one_test(b, slot, "triple_move",
      () -> move(move(move(sphere(u0(), 2.0), vxyz(3,0,0)),
                      vxyz(0,3,0)),
                 vxyz(0,0,3)),
      nothing,
      verify)

    # Rotate around an oblique axis through an off-origin point.
    run_one_test(b, slot, "rotate_around_oblique_offset_axis",
      () -> rotate(box(xyz(-1,-1,0), 2.0, 2.0, 2.0),
                   π/3, xyz(5, 5, 0), vxyz(1, 0, 1)),
      nothing,
      verify)

    # Scale with explicit non-origin pivot on a sweep result.
    run_one_test(b, slot, "scale_after_sweep",
      () -> scale(sweep(open_polygonal_path([u0(), xyz(0,0,10)]),
                        circular_path(u0(), 0.5)),
                  0.5, u0()),
      nothing,
      verify)
  end
