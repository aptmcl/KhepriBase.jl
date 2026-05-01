# stress/csg.jl — combinatorial coverage for CSG (boolean) operations.
#
# Axes:
#   - operation: subtraction, intersection, union
#   - operand pairs: 5 representative 3D pairs + 3 representative 2D pairs
#   - chained varargs: subtraction(a, b, c, d)
#   - slice: each solid × {axis-aligned plane, oblique plane}
#
# Note: AutoCAD implements b_subtract_ref/_intersect_ref/_unite_ref natively
# (AutoCAD.jl:892–901). The stress test exercises representative shapes for
# each pair to surface plugin-side limitations (e.g., CSG between curved
# and faceted solids).

stress_csg(b, reset!, verify) =
  @testset "CSG" begin
    reset!()
    slot = Slot(:csg, 0, 0)

    # ── 3D pairs across {subtract, intersect, union} ───────────────────
    pairs_3d = [
      ("sphere_box",    () -> sphere(xyz(0,0,0), 4.0),
                        () -> box(xyz(-2,-2,-2), 5.0, 5.0, 5.0)),
      ("sphere_cyl",    () -> sphere(xyz(0,0,0), 4.0),
                        () -> cylinder(xyz(0,0,-5), 2.0, 10.0)),
      ("box_cyl",       () -> box(xyz(-3,-3,0), 6.0, 6.0, 6.0),
                        () -> cylinder(xyz(0,0,-1), 2.5, 8.0)),
      ("torus_box",     () -> torus(xyz(0,0,0), 5.0, 1.5),
                        () -> box(xyz(-7,-2,-2), 14.0, 4.0, 4.0)),
      # CSG between two extrusions of Shape2D profiles (produces solids).
      # Using Shape1D profiles (circle, rectangle) here would yield surfaces
      # — AutoCAD's boolean engine cannot CSG two surfaces.
      ("ext_ext",       () -> extrusion(surface_circle(xyz(0,0,0), 3.0), vz(5.0)),
                        () -> extrusion(surface_rectangle(xyz(-1,-1,1), 2.0, 2.0), vz(8.0))),
    ]

    for (op_name, op_fn) in (("subtract", subtraction),
                             ("intersect", intersection),
                             ("union", union))
      for (pname, afn, bfn) in pairs_3d
        run_one_test(b, slot, "$(op_name)_$(pname)",
          () -> op_fn(afn(), bfn()),
          nothing,
          verify)
      end
    end

    # ── 2D booleans on surface pairs ───────────────────────────────────
    pairs_2d = [
      ("circle_rect",   () -> surface_circle(xyz(0,0,0), 3.0),
                        () -> surface_rectangle(xyz(-1.5,-1.5,0), 3.0, 3.0)),
      ("two_circles",   () -> surface_circle(xyz(-1,0,0), 3.0),
                        () -> surface_circle(xyz(1,0,0), 3.0)),
      ("rect_polygon",  () -> surface_rectangle(xyz(-3,-3,0), 6.0, 6.0),
                        () -> surface_regular_polygon(5, u0(), 3.0, 0.0, true)),
    ]

    for (op_name, op_fn) in (("subtract2d", subtraction),
                             ("intersect2d", intersection),
                             ("union2d", union))
      for (pname, afn, bfn) in pairs_2d
        run_one_test(b, slot, "$(op_name)_$(pname)",
          () -> op_fn(afn(), bfn()),
          nothing,
          verify)
      end
    end

    # ── Chained varargs ────────────────────────────────────────────────
    run_one_test(b, slot, "subtract_chain_4",
      () -> subtraction(box(xyz(-5,-5,-5), 10.0, 10.0, 10.0),
                       sphere(xyz(-3,-3,-3), 2.0),
                       sphere(xyz(3,3,-3), 2.0),
                       sphere(xyz(0,0,3), 2.5)),
      nothing,
      verify)
    run_one_test(b, slot, "union_chain_3",
      () -> union(sphere(xyz(0,0,0), 3.0),
                  sphere(xyz(4,0,0), 3.0),
                  sphere(xyz(2,3,0), 3.0)),
      nothing,
      verify)

    # ── Slice ──────────────────────────────────────────────────────────
    for (sname, sfn) in (("sphere", () -> sphere(u0(), 4.0)),
                         ("box",    () -> box(xyz(-3,-3,-3), 6.0, 6.0, 6.0)),
                         ("cylinder", () -> cylinder(xyz(0,0,-3), 3.0, 6.0)),
                         ("torus",  () -> torus(u0(), 5.0, 1.5)))
      # Axis-aligned slice
      run_one_test(b, slot, "slice_$(sname)_horizontal",
        () -> slice(sfn(), u0(), vz(1)),
        nothing,
        verify)
      # Oblique slice
      run_one_test(b, slot, "slice_$(sname)_oblique",
        () -> slice(sfn(), u0(), vxyz(1, 1, 1)),
        nothing,
        verify)
    end

    # ── Expanded coverage ────────────────────────────────────────────

    # CSG with off-center / overlapping pairs of similar primitives.
    run_one_test(b, slot, "subtract_two_spheres",
      () -> subtraction(sphere(u0(), 4.0), sphere(xyz(2,2,0), 3.0)),
      nothing,
      verify)
    run_one_test(b, slot, "intersect_two_cylinders_perpendicular",
      () -> intersection(cylinder(xyz(0,0,-5), 2.0, 10.0),
                         rotate(cylinder(xyz(0,0,-5), 2.0, 10.0),
                                π/2, u0(), vy(1))),
      nothing,
      verify)

    # 3D pair with one operand fully inside the other.
    run_one_test(b, slot, "subtract_inner_inside_outer",
      () -> subtraction(sphere(u0(), 5.0), sphere(u0(), 2.5)),
      nothing,
      verify)

    # CSG with translated/rotated operands (exercises proxy realization).
    run_one_test(b, slot, "subtract_after_move",
      () -> subtraction(box(xyz(-3,-3,0), 6.0, 6.0, 4.0),
                        move(sphere(u0(), 2.0), vxyz(0, 0, 4))),
      nothing,
      verify)

    # Chain of intersections (3 spheres).
    run_one_test(b, slot, "intersect_chain_3",
      () -> intersection(sphere(xyz(-1,0,0), 3.0),
                         sphere(xyz(1,0,0), 3.0),
                         sphere(xyz(0,1.5,0), 3.0)),
      nothing,
      verify)

    # Larger varargs chain (5 sphere subtractions from a box).
    run_one_test(b, slot, "subtract_chain_5",
      () -> subtraction(box(xyz(-6,-6,-3), 12.0, 12.0, 6.0),
                        sphere(xyz(-4,-4,0), 1.5),
                        sphere(xyz(4,-4,0), 1.5),
                        sphere(xyz(0,0,0), 1.5),
                        sphere(xyz(-4,4,0), 1.5),
                        sphere(xyz(4,4,0), 1.5)),
      nothing,
      verify)

    # 2D booleans: intersection of three surfaces (chain).
    run_one_test(b, slot, "intersect2d_chain_3",
      () -> intersection(surface_circle(xyz(-1,0,0), 3.0),
                         surface_circle(xyz(1,0,0), 3.0),
                         surface_circle(xyz(0,1.5,0), 3.0)),
      nothing,
      verify)

    # Slice with axis-perpendicular plane (different axis from xy/xyz).
    for (sname, sfn) in (("sphere", () -> sphere(u0(), 4.0)),
                         ("box",    () -> box(xyz(-3,-3,-3), 6.0, 6.0, 6.0)))
      run_one_test(b, slot, "slice_$(sname)_x_axis",
        () -> slice(sfn(), u0(), vx(1)),
        nothing,
        verify)
      run_one_test(b, slot, "slice_$(sname)_y_axis",
        () -> slice(sfn(), u0(), vy(1)),
        nothing,
        verify)
    end

    # Slice with a non-origin point.
    run_one_test(b, slot, "slice_sphere_offset_plane",
      () -> slice(sphere(u0(), 4.0), xyz(0, 0, 2), vz(1)),
      nothing,
      verify)
  end
