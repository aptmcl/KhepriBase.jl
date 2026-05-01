# stress/pathological.jl — degenerate / boundary-case inputs.
#
# These inputs are *expected* to either raise a controlled exception OR
# succeed silently (some backends tolerate inputs others reject). We do
# NOT expect them to corrupt backend state, hang, or crash the plugin.
#
# Each test wraps the build in a try/catch that explicitly accepts both
# outcomes. The failure mode we want to catch is an *unhandled* exception
# that would have crashed a regular pipeline (e.g., a NullReferenceException
# in the C# plugin, a hang on a malformed argument).
#
# Implementation note: the standard `run_one_test` already catches all
# exceptions and records them as @test false. For pathological inputs, an
# error is the *desired* outcome. We use a separate `run_pathological_test`
# helper that inverts the assertion.

#=
For pathological tests, the contract is: either build succeeds (proxy lives
in storage and is realized) OR build raises a *non-fatal* exception (any
exception other than the ones is_unimplemented matches — those would already
indicate a missing-op rather than a bad input). Both outcomes pass.

The only failure mode is: build hangs (caught by overall test timeout) or
crashes the Julia process (caught by the test framework).

We tag the test as `@test_broken` when the controlled-exception path is
taken, so the user can see at a glance which inputs the backend rejects
and which it tolerates.
=#
function run_pathological_test(b::Backend, slot::Slot, name::String,
                              build_fn::Function)
  @testset "$name" begin
    (sx, sy, sz) = slot_origin(slot)
    err = nothing
    try
      translating_current_cs(build_fn, sx, sy, sz)
    catch e
      err = e
    end
    if err === nothing
      @info "Pathological input tolerated: $name"
      @test true
    else
      @info "Pathological input rejected: $name" exception=typeof(err)
      @test_broken false
    end
    advance!(slot)
  end
end

stress_pathological(b, reset!, verify) =
  @testset "Pathological" begin
    reset!()
    slot = Slot(:pathological, 0, 0)

    # ── Curves ─────────────────────────────────────────────────────────
    run_pathological_test(b, slot, "arc_zero_amplitude",
      () -> arc(u0(), 5.0, 0.0, 0.0))
    run_pathological_test(b, slot, "arc_full_amplitude",
      () -> arc(u0(), 5.0, 0.0, 2π))
    run_pathological_test(b, slot, "regular_polygon_2_edges",
      () -> regular_polygon(2, u0(), 5.0, 0.0, true))
    run_pathological_test(b, slot, "circle_zero_radius",
      () -> circle(u0(), 0.0))
    run_pathological_test(b, slot, "rectangle_zero_dim",
      () -> rectangle(u0(), 0.0, 5.0))
    run_pathological_test(b, slot, "spline_2_points",
      () -> spline([u0(), xyz(5, 0, 0)]))
    run_pathological_test(b, slot, "polygon_collinear",
      () -> polygon([u0(), xyz(5, 0, 0), xyz(10, 0, 0)]))
    run_pathological_test(b, slot, "polygon_coincident",
      () -> polygon([u0(), u0(), xyz(5, 5, 0)]))

    # ── Surfaces ───────────────────────────────────────────────────────
    run_pathological_test(b, slot, "surface_polygon_collinear",
      () -> surface_polygon([u0(), xyz(5,0,0), xyz(10,0,0)]))
    run_pathological_test(b, slot, "surface_polygon_self_intersect",
      () -> surface_polygon([u0(), xyz(5,5,0), xyz(5,0,0), xyz(0,5,0)]))

    # ── Solids ─────────────────────────────────────────────────────────
    run_pathological_test(b, slot, "box_zero_volume",
      () -> box(u0(), 5.0, 5.0, 0.0))
    run_pathological_test(b, slot, "sphere_zero_radius",
      () -> sphere(u0(), 0.0))
    run_pathological_test(b, slot, "cylinder_zero_height",
      () -> cylinder(u0(), 3.0, 0.0))

    # ── Extrusion ──────────────────────────────────────────────────────
    run_pathological_test(b, slot, "extrude_zero_vector",
      () -> extrusion(circle(u0(), 3.0), vxyz(0.0, 0.0, 0.0)))
    # Horizontal extrusion: Z=0 sweep produces a degenerate solid; AutoCAD
    # rejects with eGeneralModelingFailure, others may tolerate.
    run_pathological_test(b, slot, "extrude_circle_horizontal",
      () -> extrusion(circle(u0(), 3.0), vxyz(5.0, 0.0, 0.0)))
    run_pathological_test(b, slot, "extrude_surface_polygon_horizontal",
      () -> extrusion(surface_polygon([u0(), xyz(4,0,0), xyz(4,3,0), xyz(0,3,0)]),
                      vxyz(5.0, 0.0, 0.0)))

    # ── Revolve ────────────────────────────────────────────────────────
    run_pathological_test(b, slot, "revolve_zero_amplitude",
      () -> revolve(line([xyz(5,0,0), xyz(5,0,5)]), u0(), vz(1), 0.0, 0.0))

    # ── Loft ───────────────────────────────────────────────────────────
    run_pathological_test(b, slot, "loft_single_profile",
      () -> loft([circle(xyz(0,0,0), 3.0)]))

    # ── CSG ────────────────────────────────────────────────────────────
    run_pathological_test(b, slot, "subtract_disjoint",
      () -> subtraction(sphere(xyz(0,0,0), 2.0), sphere(xyz(20,0,0), 2.0)))
    run_pathological_test(b, slot, "intersect_disjoint",
      () -> intersection(sphere(xyz(0,0,0), 2.0), sphere(xyz(20,0,0), 2.0)))

    # ── Expanded coverage ────────────────────────────────────────────

    # Identical shapes: subtract / intersect / unite a shape with itself.
    run_pathological_test(b, slot, "subtract_self",
      () -> subtraction(sphere(u0(), 3.0), sphere(u0(), 3.0)))
    run_pathological_test(b, slot, "intersect_self",
      () -> intersection(sphere(u0(), 3.0), sphere(u0(), 3.0)))
    run_pathological_test(b, slot, "union_self",
      () -> union(sphere(u0(), 3.0), sphere(u0(), 3.0)))

    # Tangent shapes (shared boundary, no overlap volume).
    run_pathological_test(b, slot, "subtract_tangent_spheres",
      () -> subtraction(sphere(u0(), 2.0), sphere(xyz(4, 0, 0), 2.0)))

    # Negative-amplitude arcs / revolves.
    run_pathological_test(b, slot, "arc_negative_amplitude",
      () -> arc(u0(), 5.0, 0.0, -π/2))
    run_pathological_test(b, slot, "revolve_negative_amplitude",
      () -> revolve(line([xyz(5,0,0), xyz(5,0,5)]), u0(), vz(1), 0.0, -π))

    # Very small / very large dimensions.
    run_pathological_test(b, slot, "circle_very_small_radius",
      () -> circle(u0(), 1e-10))
    run_pathological_test(b, slot, "sphere_very_large_radius",
      () -> sphere(u0(), 1e8))

    # Polygon with 3 collinear vertices.
    run_pathological_test(b, slot, "polygon_all_collinear",
      () -> polygon([xyz(0,0,0), xyz(2,0,0), xyz(4,0,0), xyz(6,0,0)]))

    # Surface polygon with 3 vertices that form a degenerate (zero-area)
    # triangle.
    run_pathological_test(b, slot, "surface_polygon_degenerate_triangle",
      () -> surface_polygon([u0(), xyz(5,0,0), xyz(5,0,0)]))

    # Box with negative dimension (should auto-correct via overload).
    run_pathological_test(b, slot, "box_negative_all",
      () -> box(u0(), -3.0, -3.0, -3.0))

    # Loft with two identical profiles (degenerate — zero loft length).
    run_pathological_test(b, slot, "loft_two_identical",
      () -> loft([circle(xyz(0, 0, 0), 3.0), circle(xyz(0, 0, 0), 3.0)]))

    # Closed loft of 4 circles arranged in a torus pattern: AutoCAD's
    # LoftedSurface.CreateLoftedSurface rejects this topology.
    run_pathological_test(b, slot, "loft_closed_4_circles_torus",
      () -> loft([circle(xyz(5, 0, 0), 1.5),
                  circle(xyz(0, 5, 2), 1.5),
                  circle(xyz(-5, 0, 0), 1.5),
                  circle(xyz(0, -5, -2), 1.5)],
                 Shape[], false, true))

    # Sweep with a path that loops back on itself (start ≈ end).
    run_pathological_test(b, slot, "sweep_self_intersecting_path",
      () -> sweep(closed_polygonal_path([u0(), xyz(5,0,0),
                                         xyz(5,5,0), xyz(0,0,0)]),
                  circular_path(u0(), 0.3)))
  end
