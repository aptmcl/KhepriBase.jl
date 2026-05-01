# stress/solids.jl — combinatorial coverage for 3D primitives.
#
# Axes:
#   - box: signed deltas + corner-form overload
#   - sphere/torus/cylinder/cone/cone_frustum: position × size span
#   - cylinder/cone: both (cb, r, h) and (cb, r, ct::Loc) overloads
#   - cuboid: 8 corners, including skewed
#   - regular_pyramid/_prism/_pyramid_frustum: edges × inscribed × h-form
#   - pyramid/pyramid_frustum: arbitrary base polygons, mismatched top vertex count
#   - prism: (bs, v::Vec) and (bs, h::Real) overloads
#   - right_cuboid with rotation angle

stress_solids(b, reset!, verify) =
  @testset "Solids" begin
    reset!()
    slot = Slot(:solids, 0, 0)

    # ── Box ────────────────────────────────────────────────────────────
    run_one_test(b, slot, "box_unit",
      () -> box(u0(), 5.0, 5.0, 5.0),
      (0.0, 5.0, 0.0, 5.0, 0.0, 5.0),
      verify)
    run_one_test(b, slot, "box_skewed_dims",
      () -> box(u0(), 10.0, 2.0, 4.0),
      (0.0, 10.0, 0.0, 2.0, 0.0, 4.0),
      verify)
    run_one_test(b, slot, "box_neg_dx",
      () -> box(xyz(5, 0, 0), -5.0, 3.0, 3.0),
      (0.0, 5.0, 0.0, 3.0, 0.0, 3.0),
      verify)
    run_one_test(b, slot, "box_corners",
      () -> box(u0(), xyz(5, 4, 3)),
      (0.0, 5.0, 0.0, 4.0, 0.0, 3.0),
      verify)
    run_one_test(b, slot, "box_corners_neg",
      () -> box(xyz(5, 4, 3), u0()),
      (0.0, 5.0, 0.0, 4.0, 0.0, 3.0),
      verify)

    # ── Sphere ─────────────────────────────────────────────────────────
    for r in (0.5, 5.0, 50.0)
      run_one_test(b, slot, "sphere_r=$r",
        () -> sphere(u0(), r),
        (-r, r, -r, r, -r, r),
        verify)
    end

    # ── Torus ──────────────────────────────────────────────────────────
    for (re, ri) in ((10.0, 3.0), (5.0, 2.0), (5.0, 4.5))
      run_one_test(b, slot, "torus_re=$(re)_ri=$(ri)",
        () -> torus(u0(), re, ri),
        (-(re+ri), re+ri, -(re+ri), re+ri, -ri, ri),
        verify)
    end

    # ── Cylinder ───────────────────────────────────────────────────────
    run_one_test(b, slot, "cylinder_h_scalar",
      () -> cylinder(u0(), 3.0, 10.0),
      (-3.0, 3.0, -3.0, 3.0, 0.0, 10.0),
      verify)
    run_one_test(b, slot, "cylinder_ct_loc_axial",
      () -> cylinder(u0(), 3.0, xyz(0, 0, 10)),
      (-3.0, 3.0, -3.0, 3.0, 0.0, 10.0),
      verify)
    run_one_test(b, slot, "cylinder_ct_loc_oblique",
      () -> cylinder(u0(), 2.0, xyz(5, 0, 8)),
      nothing,  # oblique cylinder envelope is approximate
      verify)
    run_one_test(b, slot, "cylinder_thin_tall",
      () -> cylinder(u0(), 0.2, 20.0),
      (-0.2, 0.2, -0.2, 0.2, 0.0, 20.0),
      verify)

    # ── Cone ───────────────────────────────────────────────────────────
    run_one_test(b, slot, "cone_h_scalar",
      () -> cone(u0(), 5.0, 10.0),
      (-5.0, 5.0, -5.0, 5.0, 0.0, 10.0),
      verify)
    run_one_test(b, slot, "cone_ct_loc",
      () -> cone(u0(), 5.0, xyz(0, 0, 10)),
      (-5.0, 5.0, -5.0, 5.0, 0.0, 10.0),
      verify)
    run_one_test(b, slot, "cone_ct_loc_oblique",
      () -> cone(u0(), 4.0, xyz(3, 0, 8)),
      nothing,
      verify)

    # ── Cone frustum ───────────────────────────────────────────────────
    run_one_test(b, slot, "cone_frustum_h_scalar",
      () -> cone_frustum(u0(), 5.0, 10.0, 2.0),
      (-5.0, 5.0, -5.0, 5.0, 0.0, 10.0),
      verify)
    run_one_test(b, slot, "cone_frustum_inverted",
      () -> cone_frustum(u0(), 2.0, 10.0, 5.0),
      (-5.0, 5.0, -5.0, 5.0, 0.0, 10.0),
      verify)
    run_one_test(b, slot, "cone_frustum_ct_loc",
      () -> cone_frustum(u0(), 5.0, xyz(0, 0, 10), 2.0),
      nothing,
      verify)

    # ── Cuboid (8 corners, including skewed) ──────────────────────────
    run_one_test(b, slot, "cuboid_axis_aligned",
      () -> cuboid(u0(), xyz(5,0,0), xyz(5,5,0), xyz(0,5,0),
                   xyz(0,0,3), xyz(5,0,3), xyz(5,5,3), xyz(0,5,3)),
      (0.0, 5.0, 0.0, 5.0, 0.0, 3.0),
      verify)
    run_one_test(b, slot, "cuboid_skewed_top",
      () -> cuboid(u0(), xyz(5,0,0), xyz(5,5,0), xyz(0,5,0),
                   xyz(1,1,3), xyz(4,1,3), xyz(4,4,3), xyz(1,4,3)),
      (0.0, 5.0, 0.0, 5.0, 0.0, 3.0),
      verify)

    # ── Regular pyramid: edges × inscribed × h-form ────────────────────
    for n in (3, 4, 5, 6, 8)
      r = 5.0
      run_one_test(b, slot, "regular_pyramid_n=$(n)_h_scalar",
        () -> regular_pyramid(n, u0(), r, 0.0, 8.0, true),
        (-r, r, -r, r, 0.0, 8.0),
        verify)
    end
    run_one_test(b, slot, "regular_pyramid_circ",
      () -> regular_pyramid(4, u0(), 5.0, 0.0, 8.0, false),
      nothing,  # circumscribed: actual envelope larger than r
      verify)
    run_one_test(b, slot, "regular_pyramid_ct_loc",
      () -> regular_pyramid(6, u0(), 5.0, 0.0, xyz(0,0,8), true),
      nothing,
      verify)

    # ── Regular pyramid frustum ────────────────────────────────────────
    for n in (3, 4, 6)
      run_one_test(b, slot, "regular_pyramid_frustum_n=$n",
        () -> regular_pyramid_frustum(n, u0(), 5.0, 0.0, 8.0, 2.0, true),
        (-5.0, 5.0, -5.0, 5.0, 0.0, 8.0),
        verify)
    end
    run_one_test(b, slot, "regular_pyramid_frustum_inverted",
      () -> regular_pyramid_frustum(4, u0(), 2.0, 0.0, 8.0, 5.0, true),
      (-5.0, 5.0, -5.0, 5.0, 0.0, 8.0),
      verify)

    # ── Pyramid (arbitrary base) ───────────────────────────────────────
    run_one_test(b, slot, "pyramid_quad_base",
      () -> pyramid([u0(), xyz(5,0,0), xyz(5,5,0), xyz(0,5,0)], xyz(2.5, 2.5, 8)),
      (0.0, 5.0, 0.0, 5.0, 0.0, 8.0),
      verify)
    run_one_test(b, slot, "pyramid_hex_base",
      () -> pyramid([xyz(5cos(2π*i/6), 5sin(2π*i/6), 0) for i in 0:5],
                    xyz(0, 0, 10)),
      (-5.0, 5.0, -5.0, 5.0, 0.0, 10.0),
      verify)

    # ── Pyramid frustum (top polygon may have different vertex count) ─
    run_one_test(b, slot, "pyramid_frustum_quad_quad",
      () -> pyramid_frustum(
              [u0(), xyz(5,0,0), xyz(5,5,0), xyz(0,5,0)],
              [xyz(1,1,5), xyz(4,1,5), xyz(4,4,5), xyz(1,4,5)]),
      (0.0, 5.0, 0.0, 5.0, 0.0, 5.0),
      verify)

    # ── Prism: vec form vs scalar form ─────────────────────────────────
    run_one_test(b, slot, "prism_vec_axial",
      () -> prism([u0(), xyz(5,0,0), xyz(5,5,0), xyz(0,5,0)], vz(8.0)),
      (0.0, 5.0, 0.0, 5.0, 0.0, 8.0),
      verify)
    run_one_test(b, slot, "prism_scalar",
      () -> prism([u0(), xyz(5,0,0), xyz(5,5,0), xyz(0,5,0)], 8.0),
      (0.0, 5.0, 0.0, 5.0, 0.0, 8.0),
      verify)
    run_one_test(b, slot, "prism_vec_oblique",
      () -> prism([u0(), xyz(5,0,0), xyz(2.5,5,0)], vxyz(2.0, 3.0, 8.0)),
      nothing,
      verify)

    # ── Regular prism ──────────────────────────────────────────────────
    for n in (3, 4, 5, 6, 8)
      run_one_test(b, slot, "regular_prism_n=$n",
        () -> regular_prism(n, u0(), 4.0, 0.0, 8.0, true),
        (-4.0, 4.0, -4.0, 4.0, 0.0, 8.0),
        verify)
    end

    # ── right_cuboid with angle ────────────────────────────────────────
    for angle in (0.0, π/4, π/2)
      run_one_test(b, slot, "right_cuboid_angle=$(round(angle, digits=3))",
        () -> right_cuboid(u0(), 5.0, 3.0, xyz(0, 0, 8), angle),
        nothing,  # rotated cuboid envelope depends on angle
        verify)
    end

    # ── Expanded coverage: oblique CSes, mismatched bases, edge cases ─

    # Box at oblique CS.
    run_one_test(b, slot, "box_oblique_cs",
      () -> box(loc_from_o_phi(u0(), π/4), 5.0, 3.0, 4.0),
      nothing,
      verify)

    # Sphere off-axis to verify position propagation.
    run_one_test(b, slot, "sphere_off_axis",
      () -> sphere(xyz(7, 4, 2), 2.5),
      (4.5, 9.5, 1.5, 6.5, -0.5, 4.5),
      verify)

    # Cylinder with horizontal axis (oblique ct::Loc form).
    run_one_test(b, slot, "cylinder_horizontal",
      () -> cylinder(u0(), 1.5, xyz(8, 0, 0)),
      nothing,
      verify)

    # Cone with extreme aspect (very thin and tall).
    run_one_test(b, slot, "cone_thin_tall",
      () -> cone(u0(), 0.5, 30.0),
      (-0.5, 0.5, -0.5, 0.5, 0.0, 30.0),
      verify)
    # Cone with extreme aspect (very wide and short).
    run_one_test(b, slot, "cone_wide_short",
      () -> cone(u0(), 10.0, 0.5),
      (-10.0, 10.0, -10.0, 10.0, 0.0, 0.5),
      verify)

    # Torus with extreme ratios.
    run_one_test(b, slot, "torus_thin",
      () -> torus(u0(), 8.0, 0.3),
      (-8.3, 8.3, -8.3, 8.3, -0.3, 0.3),
      verify)

    # Pyramid with non-convex (L-shaped) base.
    run_one_test(b, slot, "pyramid_concave_L_base",
      () -> pyramid([xyz(0,0,0), xyz(4,0,0), xyz(4,2,0),
                     xyz(2,2,0), xyz(2,4,0), xyz(0,4,0)],
                    xyz(2, 2, 6)),
      (0.0, 4.0, 0.0, 4.0, 0.0, 6.0),
      verify)

    # Pyramid frustum with mismatched-vertex top (unusual but legal — the
    # default emulation should bridge via ngon-on-quads).
    run_one_test(b, slot, "pyramid_frustum_4_to_4_offset",
      () -> pyramid_frustum(
              [u0(), xyz(5,0,0), xyz(5,5,0), xyz(0,5,0)],
              [xyz(0.5,0.5,5), xyz(4.5,0.5,5),
               xyz(4.5,4.5,5), xyz(0.5,4.5,5)]),
      (0.0, 5.0, 0.0, 5.0, 0.0, 5.0),
      verify)

    # Larger regular_prism vertex counts.
    for n in (10, 16, 24)
      run_one_test(b, slot, "regular_prism_n=$n",
        () -> regular_prism(n, u0(), 4.0, 0.0, 8.0, true),
        (-4.0, 4.0, -4.0, 4.0, 0.0, 8.0),
        verify)
    end

    # Regular pyramid with non-zero base rotation.
    run_one_test(b, slot, "regular_pyramid_rotated_base",
      () -> regular_pyramid(6, u0(), 4.0, π/6, 8.0, true),
      (-4.0, 4.0, -4.0, 4.0, 0.0, 8.0),
      verify)

    # Box with zero negative dimensions (corner-form auto-corrects).
    run_one_test(b, slot, "box_corners_negative_dx",
      () -> box(xyz(5, 0, 0), xyz(0, 4, 3)),
      (0.0, 5.0, 0.0, 4.0, 0.0, 3.0),
      verify)
  end
