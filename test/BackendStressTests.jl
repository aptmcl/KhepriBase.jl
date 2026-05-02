# BackendStressTests.jl — combinatorial stress tests for Khepri backends
#
# Verifies that a backend tolerates a wide variety of argument combinations
# across the modeling API without crashing. Goes beyond BackendConformanceTests
# (one example per primitive) and VisualTests (one composition per scene).
#
# Each test:
#   1. Builds shapes inside a translated coordinate system so successive tests
#      occupy disjoint slots in the X-Y plane (visible side-by-side after the run).
#   2. Asserts the produced proxy is `realized` on the backend.
#   3. Optionally verifies the shape's frontend-side AABB (via `shape_locs`)
#      matches an expected envelope. Disabled per-test by passing `nothing`
#      for the expected envelope.
#
# Usage:
#   include("BackendStressTests.jl")
#   using .BackendStressTests
#   run_stress_tests(my_backend;
#     reset! = () -> begin delete_all_shapes(); backend(my_backend) end,
#     verify = :envelope,
#     skip = Symbol[])
#
# `verify` accepts `:none` (just check no error + `realized`) or `:envelope`
# (additionally compare proxy AABB against per-test expected bounds).
#
# Categories: :curves :surfaces :solids :extrusion :sweep :revolve :loft :csg
#             :transforms :pathological. Each can be skipped via `skip=[:name]`.
#
# Layout: the suite places shapes in a Y-banded grid. Each category occupies
# a fixed Y band; tests within a category fill rows of `SLOTS_PER_ROW` slots.
# Slot size is generous (30 units) so most operations fit without bleeding
# into neighbours. After a run, the AutoCAD model preserves all shapes for
# manual inspection.

module BackendStressTests

using Test
using KhepriBase
using KhepriBase: realized, backend_name, shape_refs_storage
KhepriBase.@import_backend_api

export run_stress_tests

#=
Tolerance for envelope verification, expressed as a fraction of the AABB
diagonal. The AABB we check is the *frontend* AABB computed from `shape_locs`
— for parametric primitives (circle, sphere, ...) the loc set returned is a
sparse sample (extrema along world axes) rather than a tight bound, so we
need slack. 5% catches "approximately right size" while still flagging an
order-of-magnitude or sign error. Tighten only if a backend's storage starts
mutating proxy fields between construction and storage (which would indicate
a frontend bug, not a backend one).
=#
"Relative tolerance for AABB envelope checks (fraction of bbox diagonal)."
const ENVELOPE_TOL = 0.05

#=
Slot size and row width were chosen so a single category fits horizontally
in one or two rows for inspection in AutoCAD. `SLOT_SIZE=30` accommodates
typical test shapes (radii up to ~10, extrusion heights ~5) with margin.
`SLOTS_PER_ROW=12` keeps the grid roughly square for the largest category
(extrusion, ~70 cases → 6 rows).
=#
"Side length of one test slot in the layout grid (model units)."
const SLOT_SIZE = 30.0

"Number of slots per row before wrapping to the next row within a category."
const SLOTS_PER_ROW = 12

#=
Y origin of each category's band. Bands are spaced generously (200+ units of
Y headroom per category) so that even if a category overflows its expected
row count it does not collide with the next category. Order matches the
documented test execution order.
=#
const CATEGORY_Y = Dict{Symbol,Float64}(
  :curves       => 0.0,
  :surfaces     => 250.0,
  :solids       => 550.0,
  :extrusion    => 900.0,
  :sweep        => 1400.0,
  :revolve      => 1750.0,
  :loft         => 2050.0,
  :csg          => 2300.0,
  :transforms   => 2550.0,
  :pathological => 2800.0,
  :exercises    => 3100.0,
)

# ── Unimplemented detection (mirrors VisualTests.jl) ───────────────────

is_unimplemented(e::ErrorException) =
  occursin("UnimplementedBackendOperationException", e.msg)
is_unimplemented(e::UndefVarError) =
  startswith(string(e.var), "b_")
is_unimplemented(::Any) = false

# ── Slot layout ─────────────────────────────────────────────────────────

mutable struct Slot
  category::Symbol
  col::Int
  row::Int
end

slot_origin(s::Slot) =
  (s.col * SLOT_SIZE, CATEGORY_Y[s.category] + s.row * SLOT_SIZE, 0.0)

advance!(s::Slot) = begin
  s.col += 1
  if s.col >= SLOTS_PER_ROW
    s.col = 0
    s.row += 1
  end
end

# ── Envelope verification ───────────────────────────────────────────────

#=
Compute the world-space AABB of a list of new shape proxies using
`shape_locs`, then translate the bounds back into the slot-local frame
(so expected bounds can be specified relative to a u0()-anchored test).

Returns `nothing` when no proxy in the list reports any locs (e.g., shapes
of types lacking a `shape_locs` method); callers should treat this as
"verification not applicable" rather than failure.
=#
function compute_local_aabb(new_shapes, slot_x, slot_y, slot_z)
  bmin = [Inf, Inf, Inf]
  bmax = [-Inf, -Inf, -Inf]
  for s in new_shapes
    for loc in shape_locs(s)
      let wp = in_world(loc)
        bmin[1] = min(bmin[1], wp.x); bmax[1] = max(bmax[1], wp.x)
        bmin[2] = min(bmin[2], wp.y); bmax[2] = max(bmax[2], wp.y)
        bmin[3] = min(bmin[3], wp.z); bmax[3] = max(bmax[3], wp.z)
      end
    end
  end
  isfinite(bmin[1]) || return nothing
  ((bmin[1] - slot_x, bmax[1] - slot_x),
   (bmin[2] - slot_y, bmax[2] - slot_y),
   (bmin[3] - slot_z, bmax[3] - slot_z))
end

#=
Compare an observed local AABB to an expected one. Expected is given as
`(xmin, xmax, ymin, ymax, zmin, zmax)` — six numbers in the slot-local
frame (test code uses u0()-anchored coords). Tolerance scales with the
AABB diagonal and bottoms out at 1e-6 for degenerate (zero-extent) cases.
=#
function envelope_within(observed, expected)
  ((omx, oMx), (omy, oMy), (omz, oMz)) = observed
  (emx, eMx, emy, eMy, emz, eMz) = expected
  diag = sqrt((eMx-emx)^2 + (eMy-emy)^2 + (eMz-emz)^2)
  tol = max(diag * ENVELOPE_TOL, 1e-6)
  isapprox(omx, emx; atol=tol) && isapprox(oMx, eMx; atol=tol) &&
  isapprox(omy, emy; atol=tol) && isapprox(oMy, eMy; atol=tol) &&
  isapprox(omz, emz; atol=tol) && isapprox(oMz, eMz; atol=tol)
end

# ── Single test runner ──────────────────────────────────────────────────

#=
Run one stress test:

  - `name`     test identifier (for @testset and diagnostics)
  - `build_fn` zero-argument closure that constructs the shapes; runs inside
               a translated `current_cs` so coordinates written as `u0()` /
               `xyz(...)` inside the closure resolve to the slot's origin.
  - `expected` either `nothing` (skip envelope check) or
               `(xmin, xmax, ymin, ymax, zmin, zmax)` in slot-local coords.
  - `slot`     the slot the test lives in; advanced after the test runs.
  - `verify`   `:none` or `:envelope`.

Failure modes:
  - Build raises `UnimplementedBackendOperationException` ⇒ @test_broken.
  - Build raises any other exception ⇒ @test false (suite continues).
  - Build succeeds but no new shapes appear ⇒ @test false.
  - Any new shape is not `realized` ⇒ @test false.
  - Envelope mismatch (when verify=:envelope) ⇒ @test false.
=#
function run_one_test(b::Backend, slot::Slot, name::String,
                     build_fn::Function, expected, verify::Symbol)
  @testset "$name" begin
    (sx, sy, sz) = slot_origin(slot)
    before = Set(keys(shape_refs_storage(b)))
    err = nothing
    try
      translating_current_cs(build_fn, sx, sy, sz)
    catch e
      err = e
    end

    if err !== nothing
      if is_unimplemented(err)
        @warn "Skipping $name — unimplemented backend operation"
        @test_broken false
      else
        @error "Stress test $name raised during build" exception=(err, catch_backtrace())
        @test false
      end
      advance!(slot)
      return
    end

    new_shapes = collect(setdiff(keys(shape_refs_storage(b)), before))
    @test !isempty(new_shapes)
    if !isempty(new_shapes)
      @test all(realized(b, s) for s in new_shapes)
    end

    if verify == :envelope && expected !== nothing && !isempty(new_shapes)
      local_aabb = compute_local_aabb(new_shapes, sx, sy, sz)
      if local_aabb !== nothing
        if !envelope_within(local_aabb, expected)
          @warn "Envelope mismatch for $name" observed=local_aabb expected=expected
          @test false
        else
          @test true
        end
      end
      # If local_aabb is nothing, no shape has shape_locs entries — skip silently.
    end

    advance!(slot)
  end
end

# ── Entry point ─────────────────────────────────────────────────────────

#=
Available categories. Skipping a category bypasses both its tests and its
slot allocation, so the layout stays compact when running a subset.
=#
const CATEGORIES = Symbol[
  :curves, :surfaces, :solids, :extrusion, :sweep, :revolve,
  :loft, :csg, :transforms, :pathological, :exercises,
]

should_run(category, skip) = !(category in skip)

function run_stress_tests(b::Backend;
                          reset!::Function,
                          verify::Symbol = :envelope,
                          skip::Vector{Symbol} = Symbol[])
  verify in (:none, :envelope) || error("verify must be :none or :envelope (got $verify)")
  @testset "Backend Stress: $(backend_name(b))" begin
    should_run(:curves, skip)       && stress_curves(b, reset!, verify)
    should_run(:surfaces, skip)     && stress_surfaces(b, reset!, verify)
    should_run(:solids, skip)       && stress_solids(b, reset!, verify)
    should_run(:extrusion, skip)    && stress_extrusion(b, reset!, verify)
    should_run(:sweep, skip)        && stress_sweep(b, reset!, verify)
    should_run(:revolve, skip)      && stress_revolve(b, reset!, verify)
    should_run(:loft, skip)         && stress_loft(b, reset!, verify)
    should_run(:csg, skip)          && stress_csg(b, reset!, verify)
    should_run(:transforms, skip)   && stress_transforms(b, reset!, verify)
    should_run(:pathological, skip) && stress_pathological(b, reset!, verify)
    should_run(:exercises, skip)    && stress_exercises(b, reset!, verify)
  end
end

# Stub categories — implemented in dedicated files included below.
stress_curves(b, reset!, verify)       = error("stress_curves not loaded")
stress_surfaces(b, reset!, verify)     = error("stress_surfaces not loaded")
stress_solids(b, reset!, verify)       = error("stress_solids not loaded")
stress_extrusion(b, reset!, verify)    = error("stress_extrusion not loaded")
stress_sweep(b, reset!, verify)        = error("stress_sweep not loaded")
stress_revolve(b, reset!, verify)      = error("stress_revolve not loaded")
stress_loft(b, reset!, verify)         = error("stress_loft not loaded")
stress_csg(b, reset!, verify)          = error("stress_csg not loaded")
stress_transforms(b, reset!, verify)   = error("stress_transforms not loaded")
stress_pathological(b, reset!, verify) = error("stress_pathological not loaded")
stress_exercises(b, reset!, verify)    = error("stress_exercises not loaded")

include("stress/curves.jl")
include("stress/surfaces.jl")
include("stress/solids.jl")
include("stress/extrusion.jl")
include("stress/sweep.jl")
include("stress/revolve.jl")
include("stress/loft.jl")
include("stress/csg.jl")
include("stress/transforms.jl")
include("stress/pathological.jl")
include("stress/exercises.jl")

end # module BackendStressTests
