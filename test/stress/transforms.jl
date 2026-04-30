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
  end
