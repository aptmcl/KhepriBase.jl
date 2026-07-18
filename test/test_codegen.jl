# test_codegen.jl — Headless golden tests for the BIM-model → Julia-program codegen pipeline.
#
# The codegen core (model_to_expr → transform passes → expr_to_string) lives in KhepriBase/src/
# CodeGen.jl and is backend-agnostic: it operates on portable Khepri BIM shapes plus `Expr` and needs
# NO live backend. We build a fixed in-memory model under `with_introspection` (which suppresses
# realization) and lock the emitted source with golden files — the regression net for the pipeline.
#
# Family metadata is carried explicitly in `model.family_meta` (an IdDict family → FamilyMeta), built
# here exactly as KhepriRevit's introspect_model builds it. The two backend seams
# (`add_backend_families`→`b_native_family_expr`, `add_header`→`b_codegen_module`) are driven with the
# Revit backend so the golden matches real KhepriRevit output. Run from the KhepriRevit project.

using Test
using KhepriBase
using KhepriRevit

const KB = KhepriBase              # the codegen pipeline now lives here
const revit = KhepriRevit.revit    # backend supplying b_native_family_expr / b_codegen_module

const GOLDEN_DIR = joinpath(@__DIR__, "golden", "codegen")

# ── Helpers ────────────────────────────────────────────────────────────────

# The single non-deterministic line in the pipeline output is add_header's timestamp.
normalize_src(s::AbstractString) =
  replace(s, r"# Generated on: .*" => "# Generated on: <timestamp>")

# Assemble the model NamedTuple that the pipeline consumes (empty slots default to []).
bim_model(; levels = [], walls = [], floors = [], columns = [], beams = [],
            ceilings = [], roofs = [], fixtures = [], stairs = [], railings = [], groups = [],
            family_meta = IdDict{Any, FamilyMeta}()) =
  (levels = levels, walls = walls, floors = floors, columns = columns, beams = beams,
   ceilings = ceilings, roofs = roofs, fixtures = fixtures, stairs = stairs,
   railings = railings, groups = groups, family_meta = family_meta)

# The full pipeline exactly as generate_khepri_code composes it, minus file IO, normalized.
# codegen_passes is the SAME authoritative list generate_khepri_code uses — the golden locks the
# real pipeline, not a test-local copy that could drift from it.
run_pipeline(model) =
  let raw = KB.model_to_expr(model),
      fmap = KB.family_expr_map(model),
      passes = KB.codegen_passes(revit, fmap)
    normalize_src(KB.expr_to_string(foldl((e, p) -> p(e), passes, init = raw)))
  end

# Compare against a golden; mint it (and pass) on first run.
function check_golden(name::AbstractString, src::AbstractString)
  mkpath(GOLDEN_DIR)
  path = joinpath(GOLDEN_DIR, "$name.jl")
  if isfile(path)
    @test read(path, String) == src
  elseif haskey(ENV, "KHEPRI_REQUIRE_GOLDEN")
    # CI/stress mode: a missing golden is a failure, not an auto-mint — otherwise a deleted or
    # renamed golden file silently turns a regression into a green run.
    @error "golden missing (unset KHEPRI_REQUIRE_GOLDEN to mint)" path
    @test isfile(path)
  else
    write(path, src)
    @info "Minted golden (delete to regenerate)" path
    @test isfile(path)
  end
end

# ── The fixed reference model ────────────────────────────────────────────────
# 3 levels; a rectangular 4-wall room (shared wall family) with a door + window on one wall;
# a 4x3 regular column grid; one slab. Exercises extract_levels, extract_families, the Wall
# door/window meta_program override, add_backend_families (system + file family branches),
# loop_rerolling, and add_header in one composition. Door/window families are reconstructed named by
# family_key (fixing the type-collapse); system families keep their default names for now.
function reference_model()
  fm = IdDict{Any, FamilyMeta}()
  with_introspection(revit) do
    l0, l1, l2 = level(0.0), level(3.0), level(6.0)

    wfam = wall_family()
    fm[wfam] = FamilyMeta(category = :wall, family_name = "Basic Wall",
                          type_name = "Generic 200mm", is_system = true)

    dfam = door_family("Single-Flush:0915 x 2134mm", 0.915, 2.134)   # width, height from Revit
    fm[dfam] = FamilyMeta(category = :door, family_name = "Single-Flush", type_name = "0915 x 2134mm",
                          is_system = false, path = raw"C:\Families\Doors\Single-Flush.rfa")
    wnfam = window_family("Fixed:0915 x 1220mm", 0.915, 1.220)
    fm[wnfam] = FamilyMeta(category = :window, family_name = "Fixed", type_name = "0915 x 1220mm",
                           is_system = false, path = raw"C:\Families\Windows\Fixed.rfa")

    corners = [xyz(0, 0, 0), xyz(6, 0, 0), xyz(6, 4, 0), xyz(0, 4, 0)]
    walls = []
    for i in 1:4
      p0, p1 = corners[i], corners[mod1(i + 1, 4)]
      push!(walls, wall(open_polygonal_path([p0, p1]),
                        bottom_level = l0, top_level = l1, family = wfam))
    end
    push!(walls[1].doors, door(walls[1], xy(1.0, 0.0), family = dfam))
    push!(walls[1].windows, window(walls[1], xy(3.0, 0.9), family = wnfam))

    cfam = column_family()
    fm[cfam] = FamilyMeta(category = :column, family_name = "M_Concrete-Rectangular",
                          type_name = "300 x 450mm", is_system = false,
                          path = raw"C:\Families\Columns\Concrete-Rectangular.rfa")
    columns = []
    for i in 0:3, j in 0:2
      push!(columns, column(xyz(i * 5.0, j * 4.0, 0.0),
                            bottom_level = l0, top_level = l1, family = cfam))
    end

    sfam = slab_family()
    fm[sfam] = FamilyMeta(category = :slab, family_name = "Floor",
                          type_name = "Generic 300mm", is_system = true)
    slabs = [slab(closed_polygonal_path(corners), level = l0, family = sfam)]

    bim_model(levels = [l0, l1, l2], walls = walls, columns = columns, floors = slabs, family_meta = fm)
  end
end

# ── Tests ────────────────────────────────────────────────────────────────────

@testset "codegen" begin

  @testset "reference model (whole pipeline)" begin
    src = run_pipeline(reference_model())
    check_golden("reference_model", src)
    # Structural sanity independent of the golden bytes:
    @test occursin("using KhepriRevit", src)
    @test occursin("level_0 = level(0.0)", src)
    @test occursin("column(", src)                    # 4x3 grid present (rerolled into a nested for-loop)
    @test occursin("add_door(", src)                  # wall opening emitted portably
    @test occursin("set_backend_family(", src)        # backend family mappings emitted
    @test occursin("revit_file_family(", src)         # file-family branch (door/window/column)
    @test occursin("revit_system_family(", src)       # system-family branch (wall/slab)
    # Door family reconstructed named by family_key (collapse fix) and carrying real dims (Phase 3a):
    @test occursin("door_family(\"Single-Flush:0915 x 2134mm\", 0.915, 2.134", src)
  end

  @testset "family-type collapse fixed (collision case)" begin
    # Two DISTINCT door types (different family_key), identical Khepri params, on one wall. Pre-fix,
    # both emitted the identical `door_family()` and collapsed to ONE variable + one backend mapping
    # (last-writer-wins). Named by family_key, they now stay distinct — two vars, two mappings.
    fm = IdDict{Any, FamilyMeta}()
    src = with_introspection(revit) do
      l0, l1 = level(0.0), level(3.0)
      wfam = wall_family()
      fm[wfam] = FamilyMeta(category = :wall, family_name = "W", type_name = "T", is_system = true)
      w = wall(open_polygonal_path([xyz(0, 0, 0), xyz(6, 0, 0)]),
               bottom_level = l0, top_level = l1, family = wfam)
      d1 = door_family("Single-Flush:0915")
      fm[d1] = FamilyMeta(category = :door, family_name = "Single-Flush", type_name = "0915",
                          is_system = false, path = raw"C:\A.rfa")
      d2 = door_family("Double-Flush:1830")
      fm[d2] = FamilyMeta(category = :door, family_name = "Double-Flush", type_name = "1830",
                          is_system = false, path = raw"C:\B.rfa")
      push!(w.doors, door(w, xy(1.0, 0.0), family = d1))
      push!(w.doors, door(w, xy(4.0, 0.0), family = d2))
      run_pipeline(bim_model(levels = [l0, l1], walls = [w], family_meta = fm))
    end
    check_golden("collision_two_door_types", src)
    @test occursin("door_single_flush_0915", src)     # both types survive as distinct variables
    @test occursin("door_double_flush_1830", src)
    @test length(findall("revit_opening_file_family(", src)) >= 2   # distinct backend mappings, not collapsed (doors use the opening-file family)
  end

  @testset "multi-run stair emission + nested family extraction" begin
    # A U-shaped stair carries its 3D walk centerline as a 6th positional argument; the
    # straight form keeps the classic 5-arg emission. The stair family nests a
    # railing_family — pre-fix, bottom-up replacement mutated the outer expr before it
    # was visited, so it stayed inline (unextracted, unguarded) inside the program.
    with_introspection(revit) do
      l0, l1 = level(0.0), level(3.26)
      rfam = railing_family(height = 0.9, post_spacing = 1.0)
      sfam = stair_family(width = 1.2, riser_height = 0.1716, tread_depth = 0.28,
                          railing_family = rfam)
      walk = open_polygonal_path([xyz(0, 0, 0.0), xyz(-1.12, 0, 0.858),
                                  xyz(-1.72, 0.6, 0.858), xyz(-1.72, 3.4, 2.745),
                                  xyz(-1.12, 4.0, 2.745), xyz(-0.56, 4.0, 3.26)])
      st = stair(xyz(0, 0, 0), vx(-1), l0, l1, sfam, walk)
      rl = railing(open_polygonal_path([xyz(0, 0.6, 0), xyz(-1.72, 0.6, 3.26)]),
                   level = l0, family = rfam)
      fm = IdDict{Any, FamilyMeta}(
        sfam => FamilyMeta(category = :stair, family_name = "Cast", type_name = "U",
                           is_system = true),
        rfam => FamilyMeta(category = :railing, family_name = "Glass", type_name = "R",
                           is_system = true))
      src = run_pipeline(bim_model(levels = [l0, l1], stairs = [st], railings = [rl],
                                   family_meta = fm))
      # The walk path is emitted (6-arg form) with climbing z values:
      @test occursin("open_polygonal_path", src)
      @test occursin("0.858", src)
      # Nested-family extraction: the stair references the EXTRACTED family variable —
      # no inline `stair_family(` call remains at any use site:
      @test occursin("stair_cast_u", src)
      @test length(findall("stair_family(", src)) == 1     # the single assignment only
      # The straight form stays 5-arg (no trailing `nothing`):
      st2 = stair(xyz(0, 0, 0), vy(1), l0, l1)
      @test !occursin("nothing", KB._expr_str(KB.meta_program(st2)))
    end
  end

  @testset "beam orientation round-trips (meta_program override)" begin
    with_introspection(revit) do
      hb = beam(xyz(0, 0, 0), xyz(5, 0, 0))   # horizontal beam via the 2-point constructor
      mp = KB.meta_program(hb)
      @test mp.head == :call && mp.args[1] == :beam
      # Two positional Loc endpoints — the far end must be a POINT, not the scalar height h that the
      # auto-generated meta_program would have emitted (collapsing a horizontal beam to vertical).
      @test mp.args[2] isa Expr && mp.args[2].args[1] in (:xy, :xyz)
      @test mp.args[3] isa Expr && mp.args[3].args[1] in (:xy, :xyz)
      src = KB.expr_to_string(Expr(:block, mp))
      # Computed (Float64) coords emit as floats; the far endpoint is a POINT (5,0,0), not a scalar h.
      @test occursin("beam(xy(0.0, 0.0)", src)         # near endpoint (0,0,0)
      @test occursin("xy(5.0, 0.0)", src)              # far endpoint (5,0,0) recovered
    end
  end

  @testset "sanitize_name" begin
    @test KB.sanitize_name("Basic Wall 200mm") == "basic_wall_200mm"
    @test KB.sanitize_name("   ") == "unnamed"
    @test startswith(KB.sanitize_name("200mm"), "_")     # leading digit gets underscore
    @test KB.sanitize_name("a---b") == "a_b"             # runs collapse
  end

  @testset "loop_rerolling threshold" begin
    mk(i) = Expr(:call, :column, Expr(:call, :xyz, Float64(i), 0.0, 0.0))
    three = Expr(:block, [mk(i) for i in 0:2]...)
    @test KB.loop_rerolling(three) == three              # < 4 similar calls: not rerolled
    four = Expr(:block, [mk(i) for i in 0:3]...)
    rerolled = KB.loop_rerolling(four)
    @test any(s -> s isa Expr && s.head == :for, rerolled.args)
  end

  @testset "loop_rerolling: row-major 2D grid rerolls to a nested loop" begin
    # A 4x3 row-major grid of identical calls varying in x (outer) and y (inner). `_try_reroll`
    # now takes ref_paths as the UNION of every varying path across all pairs (not just the first
    # pair, which varies in y alone), so the 2D detector (_try_reroll_2d/_detect_nesting) fires and
    # the grid rerolls into an outer×inner nested for-loop reproducing all 12 placements.
    mk(x, y) = Expr(:call, :column, Expr(:call, :xy, Float64(x), Float64(y)))
    grid = Expr(:block, [mk(x, y) for x in 0:5:15 for y in 0:4:8]...)
    out = KB.loop_rerolling(grid)
    outer = only(filter(s -> s isa Expr && s.head == :for, out.args))
    @test outer.head == :for                                     # outer loop present
    @test any(s -> s isa Expr && s.head == :for, outer.args[2].args)  # nested inner loop
  end

  @testset "_try_make_range" begin
    step1 = KB._try_make_range([0.0, 1.0, 2.0, 3.0])
    @test step1 isa Expr && step1.head == :call && step1.args[1] == :(:) && length(step1.args) == 3
    stepN = KB._try_make_range([0.0, 5.0, 10.0])
    # count-preserving range(first; step, length=n), NOT a float first:step:last (see CodeGen.jl)
    @test stepN isa Expr && stepN.head == :call && stepN.args[1] == :range
    @test collect(eval(stepN)) == [0.0, 5.0, 10.0]   # exact step AND exact count
    irregular = KB._try_make_range([0.0, 1.0, 3.0])
    @test irregular isa Expr && irregular.head == :vect   # non-uniform → literal vector
  end

  @testset "_dedup_slabs" begin
    with_introspection(revit) do
      p = [xyz(0, 0, 0), xyz(4, 0, 0), xyz(4, 4, 0), xyz(0, 4, 0)]
      prot = [xyz(4, 4, 0), xyz(0, 4, 0), xyz(0, 0, 0), xyz(4, 0, 0)]  # same set, rotated order
      s1 = slab(closed_polygonal_path(p), level = level(0.0))
      s2 = slab(closed_polygonal_path(prot), level = level(0.0))
      @test length(KB._dedup_slabs([s1, s2])) == 1
    end
  end

  @testset "detect_level_repetition is identity (stub)" begin
    e = Expr(:block, Expr(:call, :foo, 1))
    @test KB.detect_level_repetition(e) === e
  end

end
