# Headless tests for the codegen abstraction/parametrization passes (sort_statements,
# extract_functions, detect_level_repetition, parametrize_levels, symbolize_ranges,
# unify_constants, wrap_program_in_function) and the printer features they rely on.
using Test
using KhepriBase
using KhepriBase: sort_statements, extract_functions, detect_level_repetition,
  parametrize_levels, parametrize_level_series, extract_coordinate_dimensions,
  derive_parameter_relations, round_parameters, symbolize_ranges, symbolize_angles, extract_shared_dimensions,
  unify_constants, wrap_program_in_function,
  loop_rerolling, extract_levels, extract_families, hoist_opening_frames,
  sectionalize_by_storey, expr_to_string,
  codegen_emit_timestamp, FamilyMeta, _expr_str

@testset "canonical order + rerolling" begin
  # A shuffled two-family grid: sorting must group by family (non-numeric fingerprint), order each
  # group spatially, and rerolling must then produce one nested loop per family.
  cols = Any[]
  for (fam, xs, ys) in ((:fam_a, [0.0, 5.0], [0.0, 4.0, 8.0]),
                        (:fam_b, [20.0, 25.0], [0.0, 4.0]))
    for x in xs, y in ys
      push!(cols, :(column(xy($x, $y), 0.0, level_0, level_1, $fam)))
    end
  end
  shuffled = Expr(:block, cols[[3, 7, 1, 9, 5, 8, 2, 10, 6, 4]]...)
  rolled = loop_rerolling(sort_statements(shuffled))
  @test count(s -> s isa Expr && s.head == :for, rolled.args) == 2
  # Determinism: sorting the same statements in a different arrival order yields identical output.
  shuffled2 = Expr(:block, cols[[10, 2, 8, 4, 6, 1, 9, 3, 5, 7]]...)
  @test expr_to_string(sort_statements(shuffled)) == expr_to_string(sort_statements(shuffled2))
end

@testset "let-block walls participate in rerolling" begin
  lets = [Expr(:let, Expr(:(=), :__wall, :(wall(open_polygonal_path([xy($x, 0.0), xy($(x + 4.0), 0.0)]), level_0, level_1, fam_w))),
               Expr(:block, :(add_door(__wall, xy(1.0, 0.0), fam_d)), :__wall))
          for x in [0.0, 6.0, 12.0, 18.0]]
  rolled = loop_rerolling(Expr(:block, lets...))
  @test count(s -> s isa Expr && s.head == :for, rolled.args) >= 1
  @test Meta.parseall(expr_to_string(rolled)) isa Expr
end

@testset "extract_functions: translated cluster → function + Loc param" begin
  cluster(dx, dy) = Any[
    :(wall(open_polygonal_path([xy($(0.0 + dx), $(0.0 + dy)), xy($(4.0 + dx), $(0.0 + dy))]), level_0, level_1, fam_w)),
    :(slab(rectangular_path(xy($(0.0 + dx), $(0.0 + dy)), 4.0, 3.0), level_0, fam_s))]
  prog = Expr(:block, vcat(cluster(0.0, 0.0), cluster(10.0, 0.0), cluster(20.0, 5.0))...)
  src = expr_to_string(extract_functions(prog))
  @test occursin("function wall_cluster_1(p)", src)
  @test occursin("p + vxy(4.0, 0.0)", src)
  @test occursin("wall_cluster_1(xy(10.0, 0.0))", src)
  @test occursin("wall_cluster_1(xy(20.0, 5.0))", src)
  @test Meta.parseall(src) isa Expr
  # Non-translational variation falls back to scalar parameters.
  varying = Any[]
  for (x, h) in ((0.0, 3.0), (8.0, 4.0), (16.0, 5.0))
    push!(varying, :(box(xyz($x, 0.0, 0.0), 2.0, 2.0, $h)))
    push!(varying, :(sphere(xyz($x, 5.0, $h), 1.0)))
  end
  src2 = expr_to_string(extract_functions(Expr(:block, varying...)))
  @test occursin("function box_cluster_", src2)
  @test Meta.parseall(src2) isa Expr
end

@testset "detect_level_repetition: typical floors" begin
  lvl(i, h) = Expr(:(=), Symbol("level_$i"), :(level($h)))
  floor_stmts(a, b) = Any[
    :(wall(open_polygonal_path([xy(0.0, 0.0), xy(6.0, 0.0)]), $(Symbol("level_$a")), $(Symbol("level_$b")), fam_w)),
    :(slab(rectangular_path(xy(0.0, 0.0), 6.0, 4.0), $(Symbol("level_$a")), fam_s))]
  prog = Expr(:block, lvl(0, 0.0), lvl(1, 3.0), lvl(2, 6.0), lvl(3, 9.0),
              vcat(floor_stmts(0, 1), floor_stmts(1, 2), floor_stmts(2, 3))...)
  src = expr_to_string(detect_level_repetition(prog))
  @test occursin("function typical_floor(lvl_0, lvl_1)", src)
  @test occursin("building_levels = [level_0, level_1, level_2, level_3]", src)
  @test occursin("for i = 1:3", src)
  @test occursin("typical_floor(building_levels[i], building_levels[i + 1])", src)
  @test Meta.parseall(src) isa Expr
  # Two distinct floors (different plans) must NOT be merged.
  prog2 = Expr(:block, lvl(0, 0.0), lvl(1, 3.0), lvl(2, 6.0),
               floor_stmts(0, 1)...,
               :(wall(open_polygonal_path([xy(0.0, 0.0), xy(9.0, 0.0)]), level_1, level_2, fam_w)),
               :(slab(rectangular_path(xy(0.0, 0.0), 9.0, 4.0), level_1, fam_s)))
  @test detect_level_repetition(prog2) === prog2
end

@testset "parametrize_levels + symbolize_ranges + unify_constants" begin
  lvl(i, h) = Expr(:(=), Symbol("level_$i"), :(level($h)))
  prog = Expr(:block, lvl(0, 0.0), lvl(1, 3.048), lvl(2, 6.096), lvl(3, 9.144),
              :(wall(open_polygonal_path([xy(0.0, 0.0), xy(3.048, 0.0)]), level_0, level_1, fam_w)))
  src = expr_to_string(unify_constants(parametrize_levels(prog)))
  @test occursin("floor_height = 3.048", src)
  @test occursin("n_levels = 4", src)
  @test occursin("level(2 * floor_height)", src)
  @test occursin("xy(floor_height, 0.0)", src)        # unified distinctive constant
  @test Meta.parseall(src) isa Expr
  # Irregular level heights: pass is a no-op.
  irregular = Expr(:block, lvl(0, 0.0), lvl(1, 3.0), lvl(2, 7.5), lvl(3, 9.0))
  @test parametrize_levels(irregular) === irregular
  # Loop symbolization names count + spacing after body callee and axis.
  rolled = loop_rerolling(Expr(:block,
    [:(column(xy($x, 0.0), 0.0, level_0, level_1, fam_a)) for x in [0.0, 2.75, 5.5, 8.25]]...))
  src2 = expr_to_string(symbolize_ranges(rolled))
  @test occursin("columns_x_spacing = 2.75", src2)
  @test occursin("n_columns_x = 4", src2)
  @test occursin("range(0.0, step=columns_x_spacing, length=n_columns_x)", src2)
  # Round numbers are not distinctive: no false coupling.
  prog3 = Expr(:block, Expr(:(=), :some_len, 5.0), :(wall(xy(5.0, 0.0), fam_w)))
  @test unify_constants(prog3) === prog3
end

@testset "parametrize_levels: unconnected top level" begin
  lvl(i, h) = Expr(:(=), Symbol("level_$i"), :(level($h)))
  prog = Expr(:block, lvl(0, 0.0), lvl(1, 3.26), lvl(2, 6.52),
              Expr(:(=), :level_3, :(unconnected_level(7.02))))
  src = expr_to_string(parametrize_levels(prog))
  @test occursin("floor_height = 3.26", src)
  @test occursin("level_3 = unconnected_level(2 * floor_height + 0.5)", src)   # 2 storeys + roof rise
  @test Meta.parseall(src) isa Expr
  # An unconnected level exactly on a floor multiple → no remainder term.
  prog2 = Expr(:block, lvl(0, 0.0), lvl(1, 3.0), lvl(2, 6.0),
               Expr(:(=), :level_3, :(unconnected_level(9.0))))
  @test occursin("level_3 = unconnected_level(3 * floor_height)", expr_to_string(parametrize_levels(prog2)))
end

@testset "parametrize_levels: unconnected datum bounds and minus form" begin
  lvl(i, h) = Expr(:(=), Symbol("level_$i"), :(level($h)))
  # Negative remainder emits as a subtraction, never "+ -0.1".
  prog = Expr(:block, lvl(0, 0.0), lvl(1, 3.0), lvl(2, 6.0),
              Expr(:(=), :level_3, :(unconnected_level(8.9))))
  src = expr_to_string(parametrize_levels(prog))
  @test occursin("level_3 = unconnected_level(3 * floor_height - 0.1)", src)
  @test !occursin("+ -", src)
  # Mid-storey datum (|r| > d/4): decomposing would misstate intent — stays literal.
  prog2 = Expr(:block, lvl(0, 0.0), lvl(1, 3.26), lvl(2, 6.52),
               Expr(:(=), :level_3, :(unconnected_level(4.9))))
  @test occursin("unconnected_level(4.9)", expr_to_string(parametrize_levels(prog2)))
  # Far-off site datum (k beyond the stack): stays literal.
  prog3 = Expr(:block, lvl(0, 0.0), lvl(1, 3.0), lvl(2, 6.0),
               Expr(:(=), :level_3, :(unconnected_level(100.0))))
  @test occursin("unconnected_level(100.0)", expr_to_string(parametrize_levels(prog3)))
  # Below-base datum: stays literal.
  prog4 = Expr(:block, lvl(0, 0.0), lvl(1, 3.0), lvl(2, 6.0),
               Expr(:(=), :level_3, :(unconnected_level(-2.0))))
  @test occursin("unconnected_level(-2.0)", expr_to_string(parametrize_levels(prog4)))
end

@testset "extract_coordinate_dimensions: plan dims from rebased deltas" begin
  # Eight 6.0 x-deltas + six 4.0 y-deltas (the reference-room shape) become plan_width/plan_depth.
  w(x1, y1, x2, y2) =
    :(wall(open_polygonal_path([add_xy(p0, $x1, $y1), add_xy(p0, $x2, $y2)]),
           bottom_level=level_0, top_level=level_1, family=wall_fam))
  prog = Expr(:block,
    w(0.0, 0.0, 6.0, 0.0), w(6.0, 0.0, 6.0, 4.0), w(6.0, 4.0, 0.0, 4.0), w(0.0, 4.0, 0.0, 0.0),
    :(slab(region(closed_polygonal_path([add_xy(p0, 0.0, 0.0), add_xy(p0, 6.0, 0.0),
                                         add_xy(p0, 6.0, 4.0), add_xy(p0, 0.0, 4.0)])),
           level_0, slab_fam)))
  src = expr_to_string(extract_coordinate_dimensions(prog))
  @test occursin("plan_width = 6.0", src)
  @test occursin("plan_depth = 4.0", src)
  @test occursin("add_xy(p0, plan_width, plan_depth)", src)
  @test !occursin("add_xy(p0, 6.0", src)
  @test Meta.parseall(src) isa Expr
  # Loop weighting: one statement in a length-6 loop reaches the threshold.
  prog2 = Expr(:block,
    Expr(:for, Expr(:(=), :x, Expr(:call, :range, 0.0, Expr(:kw, :step, 2.0), Expr(:kw, :length, 6))),
         Expr(:block, :(column(add_xy(p0, x, 7.5), 0, level_0, level_1, col_fam)))))
  @test occursin("plan_depth = 7.5", expr_to_string(extract_coordinate_dimensions(prog2)))
  # Below threshold: untouched.
  prog3 = Expr(:block, w(0.0, 0.0, 9.0, 0.0))
  @test extract_coordinate_dimensions(prog3) === prog3
  # Position spam is capped: a real floor plan has MANY recurring aligned-wall offsets — only
  # the top-2 dominant knobs per axis are lifted, the rest stay literal (live-corpus moradia3
  # regression: plan_width..plan_width_30).
  spam = Expr(:block,
    vcat([[:(wall(open_polygonal_path([add_xy(p0, $v, 0.0), add_xy(p0, $v, 9.0)]),
              bottom_level=level_0, top_level=level_1, family=wall_fam)) for _ in 1:4]
          for v in [1.1, 2.2, 3.3, 4.4, 5.5]]...)...)
  srcs = expr_to_string(extract_coordinate_dimensions(spam))
  @test occursin("plan_width = 1.1", srcs)
  @test occursin("plan_width_2 = 2.2", srcs)
  @test !occursin("plan_width_3", srcs)
  @test occursin("add_xy(p0, 3.3, 0.0)", srcs)          # beyond the cap: literal
end

@testset "derive_parameter_relations: strong forms only, structural targets excluded" begin
  prog = Expr(:block,
    Expr(:(=), :slab_thickness, 0.163),
    Expr(:(=), :wall_top_offset, -0.163),          # = -slab_thickness
    Expr(:(=), :parapet_rise, 0.326),              # = 2 * slab_thickness
    Expr(:(=), :plan_width, 6.0),                  # mined position: never a target
    Expr(:(=), :floor_height, 3.0))                # 6.0 = 2*3.0 must NOT couple
  src = expr_to_string(derive_parameter_relations(prog))
  @test occursin("wall_top_offset = -slab_thickness", src)
  @test occursin("parapet_rise = 2 * slab_thickness", src)
  @test occursin("plan_width = 6.0", src)
  # Identity values stay independent knobs.
  prog2 = Expr(:block, Expr(:(=), :a_dim, 0.163), Expr(:(=), :b_dim, 0.163))
  @test derive_parameter_relations(prog2) === prog2
  # SUM relations are deliberately not derived (live moradia T4: floor_height was expressed
  # from wall offsets via a coincidence sum). Values that only combine additively stay literal.
  prog3 = Expr(:block,
    Expr(:(=), :inner_radius, 10.163),
    Expr(:(=), :building_width, 5.163),
    Expr(:(=), :outer_radius, 15.326))
  @test derive_parameter_relations(prog3) === prog3
  # Structural params are never relation targets, even when a strong form matches.
  prog4 = Expr(:block,
    Expr(:(=), :wall_base_offset, 1.63),
    Expr(:(=), :floor_height, 3.26),               # exactly 2 * wall_base_offset — stays literal
    Expr(:(=), :columns_x_spacing, 3.26))          # *_spacing likewise
  @test derive_parameter_relations(prog4) === prog4
end

@testset "extract_shared_dimensions: reroll weighting, bucketing, :parameters kwargs" begin
  # A single rerolled statement in a length-8 loop reaches the threshold.
  prog = Expr(:block,
    Expr(:(=), :n_walls, 8),
    Expr(:for, Expr(:(=), :x, Expr(:call, :range, 0.0, Expr(:kw, :step, 3.0), Expr(:kw, :length, :n_walls))),
         Expr(:block, Expr(:call, :wall, :p, Expr(:kw, :top_offset, -0.26)))))
  src = expr_to_string(extract_shared_dimensions(prog))
  @test occursin("wall_top_offset = -0.26", src)
  @test occursin("top_offset=wall_top_offset", src)
  # Introspection wobble merges into one knob (snap bucketing).
  prog2 = Expr(:block,
    [Expr(:call, :wall, :p, Expr(:kw, :top_offset, -0.2)) for _ in 1:3]...,
    [Expr(:call, :wall, :p, Expr(:kw, :top_offset, -0.19999999)) for _ in 1:2]...)
  src2 = expr_to_string(extract_shared_dimensions(prog2))
  @test occursin("wall_top_offset = -0.2", src2)
  @test !occursin("-0.19999999", src2)
  @test length(findall("wall_top_offset", src2)) == 6   # 1 assign + 5 uses
  # kwargs inside an Expr(:parameters, ...) block (pbr_material emission form) are mined too.
  prog3 = Expr(:block,
    [Expr(:call, :pbr_material, Expr(:parameters, Expr(:kw, :roughness, 0.30000001)), "m$(i)")
     for i in 1:4]...)
  src3 = expr_to_string(extract_shared_dimensions(prog3))
  @test occursin("pbr_material_roughness = 0.30000001", src3)
  @test occursin("roughness=pbr_material_roughness", src3)
end

@testset "named levels: identity, emission, extraction, parametrization" begin
  # Same-elevation twins with distinct names are DISTINCT (real models carry them; height-only
  # identity silently merged 9 of a 97-level corpus model); unnamed keeps the height dedupe.
  @test level(1.0) == level(1.0)
  @test level(1.0, name="A") != level(1.0, name="B")
  @test level(1.0, name="A") == level(1.0, name="A")
  @test occursin("name=\"A\"", _expr_str(KhepriBase.meta_program(level(2.0, name="A"))))
  @test !occursin("name", _expr_str(KhepriBase.meta_program(level(2.0))))
  # extract_levels handles named calls; same-height twins get deterministic numbering by name.
  prog = Expr(:block,
    :(wall(p, level(3.33, name="B"), fam)),
    :(wall(q, level(3.33, name="A"), fam)))
  src = expr_to_string(extract_levels(prog))
  @test occursin("level_0 = level(3.33, name=\"A\")", src)
  @test occursin("level_1 = level(3.33, name=\"B\")", src)
  # parametrize_levels preserves the name kwarg when rebuilding the uniform ladder…
  lvlN(i, h, n) = Expr(:(=), Symbol("level_$i"), Expr(:call, :level, h, Expr(:kw, :name, n)))
  prog2 = Expr(:block, lvlN(0, 0.0, "P0"), lvlN(1, 3.0, "P1"), lvlN(2, 6.0, "P2"))
  src2 = expr_to_string(parametrize_levels(prog2))
  @test occursin("level(floor_height, name=\"P1\")", src2)
  # …and the series rewrite drops names deliberately (uniform ladder ⇒ unique elevations ⇒
  # elevation-keyed replay is exact; names only matter for same-elevation twins).
  prog3 = Expr(:block,
    Expr(:(=), :floor_height, 3.0), Expr(:(=), :n_levels, 3),
    lvlN(0, 0.0, "P0"),
    Expr(:(=), :level_1, Expr(:call, :level, :floor_height, Expr(:kw, :name, "P1"))),
    Expr(:(=), :level_2, Expr(:call, :level, Expr(:call, :*, 2, :floor_height),
                              Expr(:kw, :name, "P2"))))
  src3 = expr_to_string(parametrize_level_series(prog3))
  @test occursin("levels = [", src3)
  @test !occursin("name=", src3)
end

@testset "resilient realization contains member failures" begin
  # A generated factory/storey body replays as ONE statement: without containment, one
  # throwing member silently drops every later member (live goldennugget lost ~half the
  # model this way). Under resilient_realization, the bad member records an error + a void
  # ref and the rest realize.
  b = KhepriBase.measure_backend()
  prev = KhepriBase.current_backends()
  KhepriBase.current_backend(b)
  try
    # Coincident spline points make the default b_spline throw deterministically.
    @test_throws Exception spline([xy(0, 0), xy(0, 0)])
    with(KhepriBase.resilient_realization, true) do
      with(KhepriBase.realization_errors, Any[]) do
        box(xyz(0, 0, 0), 1, 1, 1)
        bad = spline([xy(5, 5), xy(5, 5)])          # contained, not fatal
        box(xyz(10, 0, 0), 1, 1, 1)
        @test length(KhepriBase.realization_errors()) == 1
        @test KhepriBase.realization_errors()[1][1] === bad
        ms = [m.measurement for m in KhepriBase.measured_shapes(b) if m.measurement.n_trigs > 0]
        @test length(ms) == 2                       # both boxes realized after the failure
      end
    end
  finally
    KhepriBase.current_backends(prev)
  end
end

@testset "model_to_expr: railings precede failure-prone categories" begin
  # Order pin for the storey-abort shield: railings must be emitted before columns, roofs,
  # fixtures and stairs, so an abort in those classes cannot take the railings down with it.
  fake(catsym, n) = [Expr(:call, catsym, i) for i in 1:n]
  # Use meta_program-free stand-ins through a NamedTuple model with empty complex slots and
  # verify via the emission loops' relative order on a real introspection-shaped model instead:
  # simplest faithful check — emit a model with one element of each ordered category through
  # with_introspection-style proxies is Revit-dependent, so pin the SOURCE order directly.
  src = read(joinpath(dirname(pathof(KhepriBase)), "CodeGen.jl"), String)
  railings_at = findfirst("for rl in model.railings", src)
  columns_at = findfirst("for c in model.columns", src)
  stairs_at = findfirst("for st in model.stairs", src)
  @test railings_at !== nothing && columns_at !== nothing && stairs_at !== nothing
  @test railings_at[1] < columns_at[1] < stairs_at[1]
end

@testset "round_parameters: architectural values with provenance comments" begin
  # Introspection noise (-0.19999999, 3.2599999) becomes what a designer would write, with
  # the original preserved in a trailing comment; already-round and non-round-neighbor values
  # stay verbatim and uncommented.
  prog = Expr(:block,
    Expr(:(=), :wall_top_offset, -0.19999999),
    Expr(:(=), :floor_height, 3.2599999),
    Expr(:(=), :plan_depth, 7.574999),
    Expr(:(=), :odd_dim, 0.241434),
    Expr(:(=), :n_levels, 3),
    :(level_0 = level(0.0)))
  src = expr_to_string(round_parameters(prog))
  @test occursin("wall_top_offset = -0.2  # rounded from the introspected -0.19999999", src)
  @test occursin("floor_height = 3.26  # rounded from the introspected 3.2599999", src)
  @test occursin("plan_depth = 7.575  # rounded from the introspected 7.574999", src)
  @test occursin("odd_dim = 0.241434\n", src)
  @test Meta.parseall(src) isa Expr
  # 3.048 (10 ft) IS the architectural number — untouched, no comment.
  clean = Expr(:block, Expr(:(=), :floor_height, 3.048))
  @test round_parameters(clean) === clean
  # wrap still collects rounded params as kwargs (comment dropped in kwarg form).
  wrapped = expr_to_string(wrap_program_in_function(round_parameters(Expr(:block,
    Expr(:(=), :floor_height, 3.2599999),
    :(wall(open_polygonal_path([xy(0.0, 0.0), xy(6.0, 0.0)]), level_0, level_0, fam_w))))))
  @test occursin("function building(; floor_height=3.26)", wrapped)
end

@testset "symbolize_angles: tolerance and extended coverage" begin
  prog = Expr(:block,
    :(family_element(xy(0.0, 0.0), 1.0471879, level_0, f)),     # 9.7e-6 off pi/3: survey angle
    :(column(xy(1.0, 2.0), 1.5707963, level_0, level_1, cf)),   # positional column angle
    Expr(:call, :beam, :p, :q, Expr(:kw, :angle, 3.1415927), Expr(:kw, :family, :bf)),
    Expr(:call, :free_column, :p, :q, Expr(:kw, :angle, 0.7853982), Expr(:kw, :family, :ff)))
  src = expr_to_string(symbolize_angles(prog))
  @test occursin("1.0471879", src)                              # near-miss NOT symbolized
  @test occursin("column(xy(1.0, 2.0), pi / 2, level_0, level_1, cf)", src)
  @test occursin("angle=pi,", src)                              # beam kw symbolized
  @test occursin("angle=pi / 4", src)                           # free_column kw symbolized
  @test Meta.parseall(src) isa Expr
end

@testset "extract_shared_dimensions: angle kwargs are never mined" begin
  prog = Expr(:block, [Expr(:call, :beam, :p, Expr(:kw, :angle, 0.7853982)) for _ in 1:5]...)
  @test extract_shared_dimensions(prog) === prog
end

@testset "naming: collision suffixes are numeric everywhere" begin
  frame = :(frame_family("frame_family", rectangular_path(xy(-0.08, -0.08), 0.16, 0.16)))
  prog = Expr(:block,
    Expr(:(=), :opening_frame, :(something_else())),
    Expr(:(=), :door_a, Expr(:call, :door_family, "A", 1.4, 2.0, 0.05, frame)),
    Expr(:(=), :door_b, Expr(:call, :door_family, "B", 0.7, 2.0, 0.05, frame)))
  src = expr_to_string(hoist_opening_frames(prog))
  @test occursin("opening_frame_2 = frame_family(", src)
  @test !occursin("opening_frame_x", src)
  prog2 = Expr(:block,
    Expr(:(=), :wall_top_offset, 9.9),
    [Expr(:call, :wall, :p, Expr(:kw, :top_offset, -0.163)) for _ in 1:4]...)
  src2 = expr_to_string(extract_shared_dimensions(prog2))
  @test occursin("wall_top_offset_2 = -0.163", src2)
  @test occursin("top_offset=wall_top_offset_2", src2)
  @test !occursin("wall_top_offset_x", src2)
end

@testset "parametrize_level_series: n_levels is live" begin
  lvl(i, rhs) = Expr(:(=), Symbol("level_$i"), rhs)
  prog = Expr(:block,
    Expr(:(=), :floor_height, 3.0), Expr(:(=), :n_levels, 3),
    lvl(0, :(level(0.0))), lvl(1, :(level(floor_height))), lvl(2, :(level(2 * floor_height))),
    Expr(:(=), :building_levels, Expr(:vect, :level_0, :level_1, :level_2)),
    Expr(:for, Expr(:(=), :i, Expr(:call, :(:), 1, 2)),
         Expr(:block, :(typical_floor(building_levels[i], building_levels[i + 1])))))
  src = expr_to_string(parametrize_level_series(prog))
  @test occursin("levels = [", src) && occursin("for i = 0:n_levels - 1", src)
  @test occursin("level(i * floor_height)", src)
  @test occursin("level_0 = levels[1]", src)
  @test occursin("level_2 = levels[3]", src)
  @test occursin("building_levels = levels", src)
  @test occursin("for i = 1:n_levels - 1", src)
  @test Meta.parseall(src) isa Expr
  # base_level_height variant.
  prog2 = Expr(:block,
    Expr(:(=), :base_level_height, 1.2), Expr(:(=), :floor_height, 3.0), Expr(:(=), :n_levels, 2),
    lvl(0, :(level(base_level_height))), lvl(1, :(level(base_level_height + floor_height))))
  @test occursin("level(base_level_height + i * floor_height)",
                 expr_to_string(parametrize_level_series(prog2)))
  # No n_levels parameter: identity (parametrize_levels bailed on irregular spacing).
  prog3 = Expr(:block, lvl(0, :(level(0.0))), lvl(1, :(level(3.0))))
  @test parametrize_level_series(prog3) === prog3
  # Unconnected markers stay outside the series.
  prog4 = Expr(:block,
    Expr(:(=), :floor_height, 3.0), Expr(:(=), :n_levels, 2),
    lvl(0, :(level(0.0))), lvl(1, :(level(floor_height))),
    Expr(:(=), :level_2, :(unconnected_level(2 * floor_height + 0.5))))
  @test occursin("level_2 = unconnected_level(2 * floor_height + 0.5)",
                 expr_to_string(parametrize_level_series(prog4)))
end

@testset "n_levels drives storey replication (measure)" begin
  # The cardinality knob must be REAL: patching n_levels 3 -> 5 in the emitted program adds two
  # levels AND two typical storeys of geometry. Dead-parameter regression (the review's I2).
  lvl(i, rhs) = Expr(:(=), Symbol("level_$i"), rhs)
  tf = Expr(:function, Expr(:call, :typical_floor, :lvl_0, :lvl_1),
            Expr(:block,
              :(wall(open_polygonal_path([xy(0.0, 0.0), xy(6.0, 0.0)]),
                     bottom_level=lvl_0, top_level=lvl_1, family=wall_fam))))
  prog = Expr(:block,
    Expr(:(=), :floor_height, 3.0), Expr(:(=), :n_levels, 3),
    lvl(0, :(level(0.0))), lvl(1, :(level(floor_height))), lvl(2, :(level(2 * floor_height))),
    :(wall_fam = wall_family("wall_family", 0.2, 0.0, 0.0)),
    tf,
    Expr(:(=), :building_levels, Expr(:vect, :level_0, :level_1, :level_2)),
    Expr(:for, Expr(:(=), :i, Expr(:call, :(:), 1, 2)),
         Expr(:block, :(typical_floor(building_levels[i], building_levels[i + 1])))))
  ser = parametrize_level_series(prog)
  patch(e, n) =
    Expr(:block, [s isa Expr && s.head == :(=) && s.args[1] == :n_levels ?
                    Expr(:(=), :n_levels, n) : s
                  for s in KhepriBase.stmts_of(e)]...)
  measure_count(e) =
    let mb = KhepriBase.measure_backend(),
        rows = KhepriBase.measured_statements(e; b=mb)
      @test all(r -> r.error === nothing, rows)
      length([m for m in KhepriBase.measured_shapes(mb) if m.measurement.n_trigs > 0])
    end
  @test measure_count(ser) == 2
  @test measure_count(patch(ser, 5)) == 4
end

@testset "wrap_program_in_function" begin
  prog = Expr(:block,
              Expr(:(=), :floor_height, 3.048), Expr(:(=), :n_levels, 4),
              Expr(:(=), :level_0, :(level(0.0))),
              :(wall(open_polygonal_path([xy(0.0, 0.0), xy(6.0, 0.0)]), level_0, level_0, fam_w)))
  src = expr_to_string(wrap_program_in_function(prog))
  @test occursin("function building(; floor_height=3.048, n_levels=4)", src)
  @test occursin("building()", src)
  @test Meta.parseall(src) isa Expr
end

@testset "family variable name collisions get a counter" begin
  fmap = Dict{Expr, FamilyMeta}()
  f1 = :(door_family("A:0915 x 2134mm", 0.915, 2.134))
  f2 = :(door_family("A:0915-X-2134MM", 0.915, 2.135))
  fmap[f1] = FamilyMeta(category=:door, family_name="A", type_name="0915 x 2134mm", is_system=false)
  fmap[f2] = FamilyMeta(category=:door, family_name="A", type_name="0915-X-2134MM", is_system=false)
  src = expr_to_string(extract_families(fmap)(Expr(:block, :(door(w, xy(1, 0), $f1)), :(door(w, xy(2, 0), $f2)))))
  @test occursin("door_a_0915_x_2134mm = ", src)
  @test occursin("door_a_0915_x_2134mm_2 = ", src)
end

@testset "timestamp parameter" begin
  @test codegen_emit_timestamp() == true
  with(codegen_emit_timestamp, false) do
    @test codegen_emit_timestamp() == false
  end
end

@testset "sectionalize_by_storey: one building origin, total rebase" begin
  lvl(i, h) = Expr(:(=), Symbol("level_$i"), :(level($h)))
  wall(x1, y1, x2, y2, lo, hi) =
    Expr(:call, :wall, :(open_polygonal_path([xy($x1, $y1), xy($x2, $y2)])),
         Expr(:kw, :bottom_level, Symbol("level_$lo")), Expr(:kw, :top_level, Symbol("level_$hi")),
         Expr(:kw, :family, :wall_fam))
  wA = Expr(:let, Expr(:(=), :__wall, wall(-23.0, -81.0, -20.0, -81.0, 0, 1)),
            Expr(:block, :(add_door(__wall, xy(1.5, 0.0), door_fam)), :__wall))
  prog = Expr(:block, lvl(0, 0.0), lvl(1, 3.0), lvl(2, 6.0),
              wA, wall(-23.0, -75.0, -20.0, -75.0, 0, 1),
              wall(-22.0, -80.0, -19.0, -80.0, 1, 2), wall(-22.0, -78.0, -19.0, -78.0, 1, 2))
  src = expr_to_string(sectionalize_by_storey()(prog))
  @test occursin("building_origin = xy(-23.0, -81.0)", src)        # global SW corner
  @test occursin("function storey_0(p0=building_origin)", src)
  @test occursin("function storey_1(p0=building_origin)", src)     # ONE knob for every storey
  @test occursin("add_xy(p0, 0.0, 0.0)", src)                      # SW vertex → zero offset
  @test occursin("add_xy(p0, 3.0, 6.0)", src)                      # far vertex → clean delta
  @test occursin("add_xy(p0, 1.0, 1.0)", src)                      # storey_1 shares the origin
  @test occursin("add_door(__wall, xy(1.5, 0.0), door_fam)", src)  # local door coord NOT rebased
  @test occursin("storey_0()", src)                                # called with default anchor
  @test occursin("  let __wall = wall(", src)                      # :let prints via the printer arm
  @test !occursin("p0 + vxy", src)
  @test Meta.parseall(src) isa Expr
end

@testset "sectionalize_by_storey: loop and symbol coordinates rebase too" begin
  lvl(i, h) = Expr(:(=), Symbol("level_$i"), :(level($h)))
  prog = Expr(:block, Expr(:(=), :floor_height, 3.0), lvl(0, 0.0), lvl(1, 3.0), lvl(2, 6.0),
    :(wall(open_polygonal_path([xy(0.0, 0.0), xy(floor_height, 0.0)]),
           bottom_level=level_0, top_level=level_1, family=wall_fam)),
    Expr(:for, Expr(:(=), :x, Expr(:call, :range, 2.0, Expr(:kw, :step, 5.0), Expr(:kw, :length, 4))),
         Expr(:block, :(column(xy(x, 12.0), 0, level_0, level_1, col_fam)))),
    :(family_element(xyz(-5.0, -3.0, floor_height), pi / 2, level_1, fix_fam)))
  src = expr_to_string(sectionalize_by_storey()(prog))
  @test occursin("building_origin = xy(-5.0, -3.0)", src)
  @test occursin("add_xy(p0, x + 5.0, 15.0)", src)                 # loop var: symbolic delta
  @test occursin("add_xy(p0, floor_height + 5.0, 3.0)", src)       # param symbol: symbolic delta
  @test occursin("add_xyz(p0, 0.0, 0.0, floor_height)", src)       # z stays level-relative
  @test !occursin("xy(x, 12.0)", src)                              # nothing absolute survives
  @test Meta.parseall(src) isa Expr
end

@testset "sectionalize_by_storey: typical_floor rebased and anchor forwarded" begin
  lvl(i, h) = Expr(:(=), Symbol("level_$i"), :(level($h)))
  tf = Expr(:function, Expr(:call, :typical_floor, :lvl_0, :lvl_1),
            Expr(:block,
              :(wall(open_polygonal_path([xy(10.0, 20.0), xy(16.0, 20.0)]),
                     bottom_level=lvl_0, top_level=lvl_1, family=wall_fam))))
  prog = Expr(:block, lvl(0, 0.0), lvl(1, 3.0), lvl(2, 6.0),
              tf,
              :(typical_floor(level_0, level_1)),
              :(typical_floor(level_1, level_2)))
  src = expr_to_string(sectionalize_by_storey()(prog))
  @test occursin("building_origin = xy(10.0, 20.0)", src)
  @test occursin("function typical_floor(lvl_0, lvl_1, p0=building_origin)", src)
  @test occursin("add_xy(p0, 0.0, 0.0)", src)                      # def body rebased
  @test occursin("typical_floor(level_0, level_1, p0)", src)       # storey forwards its anchor
  @test occursin("function storey_0(p0=building_origin)", src)     # coord-less storey still anchored
  @test Meta.parseall(src) isa Expr
end

@testset "sectionalize_by_storey: factories untouched, instances rebased, finalize last" begin
  lvl(i, h) = Expr(:(=), Symbol("level_$i"), :(level($h)))
  factory = Expr(:function, Expr(:call, :group_desk_factory),
                 Expr(:block, :(slab(rectangular_path(xy(0.0, 0.0), 1.0, 0.5), level_0, slab_fam))))
  prog = Expr(:block, lvl(0, 0.0), lvl(1, 3.0), lvl(2, 6.0),
              :(wall(open_polygonal_path([xy(5.0, 5.0), xy(11.0, 5.0)]),
                     bottom_level=level_0, top_level=level_1, family=wall_fam)),
              factory,
              Expr(:(=), :group_desk,
                   Expr(:call, :group, Expr(:parameters, Expr(:kw, :factory, :group_desk_factory)), "desk")),
              :(group_instance(group_desk, xyz(7.0, 6.0, 0.0))),
              :(wall(open_polygonal_path([xy(5.0, 9.0), xy(11.0, 9.0)]),
                     bottom_level=level_1, top_level=level_2, family=wall_fam)),
              :(finalize_groups()))
  src = expr_to_string(sectionalize_by_storey()(prog))
  @test occursin("building_origin = xy(5.0, 5.0)", src)
  @test occursin("rectangular_path(xy(0.0, 0.0), 1.0, 0.5)", src)  # factory body stays relative
  @test occursin("group_instance(group_desk, add_xyz(p0, 2.0, 1.0, 0.0))", src)
  @test endswith(rstrip(src), "finalize_groups()")
  @test Meta.parseall(src) isa Expr
end

@testset "sectionalize_by_storey: single-storey program is untouched" begin
  prog = Expr(:block, Expr(:(=), :level_0, :(level(0.0))),
              :(wall(open_polygonal_path([xy(0.0, 0.0), xy(6.0, 0.0)]),
                     bottom_level=level_0, top_level=level_0, family=wall_fam)))
  @test sectionalize_by_storey()(prog) === prog
end

@testset "building_origin rigid translation (measure oracle)" begin
  # The keystone test for the anchor contract: shifting building_origin must move EVERY shape
  # of the design — literal walls, a symbol-bearing endpoint, a rerolled column grid, and
  # typical-floor content — by exactly the same vector, with sizes unchanged. This fails on the
  # partial (literal-only) rewrite the first attempt shipped.
  lvl(i, h) = Expr(:(=), Symbol("level_$i"), :(level($h)))
  tf = Expr(:function, Expr(:call, :typical_floor, :lvl_0, :lvl_1),
            Expr(:block,
              :(wall(open_polygonal_path([xy(3.0, 7.0), xy(9.0, 7.0)]),
                     bottom_level=lvl_0, top_level=lvl_1, family=wall_fam)),
              :(slab(region(closed_polygonal_path([xy(3.0, 7.0), xy(9.0, 7.0), xy(9.0, 9.0), xy(3.0, 9.0)])),
                     lvl_0, slab_fam))))
  prog = Expr(:block,
    Expr(:(=), :floor_height, 3.0),
    lvl(0, 0.0), lvl(1, 3.0), lvl(2, 6.0),
    :(wall_fam = wall_family("wall_family", 0.2, 0.0, 0.0)),
    :(slab_fam = slab_family("slab_family", 0.2, 0.0)),
    :(col_fam = column_family("column_family", rectangular_path(xy(-0.1, -0.1), 0.2, 0.2))),
    tf,
    :(wall(open_polygonal_path([xy(0.0, 0.0), xy(6.0, 0.0)]),
           bottom_level=level_0, top_level=level_1, family=wall_fam)),
    :(slab(region(closed_polygonal_path([xy(0.0, 0.0), xy(6.0, 0.0), xy(6.0, 4.0), xy(0.0, 4.0)])),
           level_0, slab_fam)),
    :(wall(open_polygonal_path([xy(0.0, floor_height), xy(6.0, floor_height)]),
           bottom_level=level_0, top_level=level_1, family=wall_fam)),
    Expr(:for, Expr(:(=), :x, Expr(:call, :range, 1.0, Expr(:kw, :step, 2.0), Expr(:kw, :length, 3))),
         Expr(:block, :(column(xy(x, 2.0), 0, level_0, level_1, col_fam)))),
    :(typical_floor(level_0, level_1)),
    :(typical_floor(level_1, level_2)))
  sec = sectionalize_by_storey()(prog)
  shift_origin(e, dx, dy) =
    Expr(:block, [s isa Expr && s.head == :(=) && s.args[1] == :building_origin ?
                    Expr(:(=), :building_origin,
                         Expr(:call, :xy, s.args[2].args[2] + dx, s.args[2].args[3] + dy)) : s
                  for s in KhepriBase.stmts_of(sec)]...)
  measure(e) =
    let mb = KhepriBase.measure_backend(),
        rows = KhepriBase.measured_statements(e; b=mb)
      @test all(r -> r.error === nothing, rows)
      [r.measurement for r in KhepriBase.measured_shapes(mb) if r.measurement.n_trigs > 0]
    end
  ms1 = measure(sec)
  ms2 = measure(shift_origin(sec, 50.0, -30.0))
  @test length(ms1) > 0
  @test length(ms1) == length(ms2)
  let v = vxyz(50.0, -30.0, 0.0, world_cs),
      back = [KhepriBase.Measurement(m.n_trigs, m.aabb_min - v, m.aabb_max - v,
                                     m.centroid - v, m.area, m.signed_volume)
              for m in ms2],
      res = KhepriBase.match_measurements(ms1, back; tol_pos=1e-6, tol_size=1e-6)
    @test isempty(res.unmatched_a)
    @test isempty(res.unmatched_b)
  end
end

@testset "hoist_opening_frames" begin
  frame = :(frame_family("frame_family", rectangular_path(xy(-0.08, -0.08), 0.16, 0.16)))
  prog = Expr(:block,
    Expr(:(=), :level_0, :(level(0.0))),
    Expr(:(=), :door_a, Expr(:call, :door_family, "A", 1.4, 2.0, 0.05, frame)),
    Expr(:(=), :door_b, Expr(:call, :door_family, "B", 0.7, 2.0, 0.05, frame)),
    Expr(:(=), :window_c, Expr(:call, :window_family, "C", 1.0, 1.0, 0.05, frame)),
    :(add_door(w, xy(1.0, 0.0), door_a)))
  src = expr_to_string(hoist_opening_frames(prog))
  # The symmetric standard casing additionally gets its width lifted (Isenberg frame_width idiom).
  @test occursin("frame_width = 0.16", src)
  @test occursin("opening_frame = frame_family(\"frame_family\", rectangular_path(xy(-frame_width / 2, -frame_width / 2), frame_width, frame_width))", src)
  @test occursin("door_a = door_family(\"A\", 1.4, 2.0, 0.05, opening_frame)", src)
  @test occursin("window_c = window_family(\"C\", 1.0, 1.0, 0.05, opening_frame)", src)
  @test !occursin("0.05, frame_family(", src)             # no inline frame left behind
  @test Meta.parseall(src) isa Expr
  # An ASYMMETRIC casing keeps its literal profile (no width knob to name).
  aframe = :(frame_family("frame_family", rectangular_path(xy(-0.08, -0.08), 0.16, 0.2)))
  prog2 = Expr(:block,
    Expr(:(=), :door_a, Expr(:call, :door_family, "A", 1.4, 2.0, 0.05, aframe)),
    Expr(:(=), :door_b, Expr(:call, :door_family, "B", 0.7, 2.0, 0.05, aframe)))
  @test occursin("rectangular_path(xy(-0.08, -0.08), 0.16, 0.2)",
                 expr_to_string(hoist_opening_frames(prog2)))
  # A frame used only once stays inline (no-op, same object).
  single = Expr(:block, Expr(:(=), :door_x, Expr(:call, :door_family, "X", 1.0, 2.0, 0.05, frame)))
  @test hoist_opening_frames(single) === single
end

@testset "printing polish: negated fractions, loop-order params" begin
  @test _expr_str(Expr(:call, :-, Expr(:call, :/, :pi, 2))) == "-pi / 2"
  @test _expr_str(Expr(:call, :-, Expr(:call, :*, 2, :pi))) == "-2 * pi"
  @test _expr_str(Expr(:call, :-, Expr(:call, :+, :a, :b))) == "-(a + b)"
  # Nested reroll: outer-axis knobs emitted before inner-axis knobs, the way an author writes them.
  grid = Expr(:for, Expr(:(=), :x, Expr(:call, :range, 0.0, Expr(:kw, :step, 5.0), Expr(:kw, :length, 4))),
           Expr(:block,
             Expr(:for, Expr(:(=), :y, Expr(:call, :range, 0.0, Expr(:kw, :step, 4.0), Expr(:kw, :length, 3))),
               Expr(:block, :(column(xy(x, y), 0, level_0, level_1, fam))))))
  src = expr_to_string(symbolize_ranges(Expr(:block, grid)))
  @test findfirst("n_columns_x", src)[1] < findfirst("n_columns_y", src)[1]
end

@testset "protected constructs are never aliased or mined" begin
  # Composition (hoist -> dims -> unify), the codegen_passes relative order: the hoisted
  # frame_family loses the transitive protection it had INSIDE door_family, so it must be
  # protected by name — a wall dimension equal to the casing width must not be aliased into
  # the casing profile.
  frame = :(frame_family("frame_family", rectangular_path(xy(-0.08, -0.08), 0.16, 0.2)))
  prog = Expr(:block,
    Expr(:(=), :door_a, Expr(:call, :door_family, "A", 1.4, 2.0, 0.05, frame)),
    Expr(:(=), :door_b, Expr(:call, :door_family, "B", 0.7, 2.0, 0.05, frame)),
    [Expr(:call, :wall, :p, Expr(:kw, :top_offset, 0.16)) for _ in 1:5]...)
  src = expr_to_string(unify_constants(extract_shared_dimensions(hoist_opening_frames(prog))))
  @test occursin("wall_top_offset = 0.16", src)
  @test occursin("rectangular_path(xy(-0.08, -0.08), 0.16, 0.2)", src)   # casing NOT aliased
  # Dim params must not leak into unconnected_level heights or group_instance placements.
  prog2 = Expr(:block,
    [Expr(:call, :wall, :p, Expr(:kw, :top_offset, -0.163)) for _ in 1:4]...,
    Expr(:(=), :level_3, :(unconnected_level(-0.163))),
    :(group_instance(grp, xyz(-0.163, 0.0, 0.0))))
  src2 = expr_to_string(unify_constants(extract_shared_dimensions(prog2)))
  @test occursin("top_offset=wall_top_offset", src2)
  @test occursin("unconnected_level(-0.163)", src2)
  @test occursin("group_instance(grp, xyz(-0.163, 0.0, 0.0))", src2)
end

@testset "extract_shared_dimensions: Int kwargs are counts, not dimensions" begin
  # Ints recurring as kwargs must survive untouched (lifting them as Float64 params
  # changes call-site types: range(length=7.0) throws).
  prog = Expr(:block, [Expr(:call, :truss, :p, Expr(:kw, :n_panels, 6)) for _ in 1:5]...)
  @test extract_shared_dimensions(prog) === prog
  # An Int and a Float of the same numeric value must not merge.
  prog2 = Expr(:block,
    [Expr(:call, :wall, :p, Expr(:kw, :top_offset, 3.0)) for _ in 1:4]...,
    Expr(:call, :truss, :p, Expr(:kw, :n_panels, 3)))
  src2 = expr_to_string(extract_shared_dimensions(prog2))
  @test occursin("n_panels=3", src2)
  @test occursin("top_offset=wall_top_offset", src2)
end

@testset "extract_shared_dimensions" begin
  # A value recurring >= threshold under one (callee, keyword) → named <callee>_<keyword>.
  walls = [Expr(:call, :wall, :(open_polygonal_path([xy($x, 0.0), xy($(x + 1.0), 0.0)])),
                Expr(:kw, :top_level, :level_1),
                Expr(:kw, :top_offset, -0.19999999))
           for x in 0.0:1.0:4.0]                          # 5 walls, all top_offset = -0.19999999
  src = expr_to_string(extract_shared_dimensions(Expr(:block, walls...)))
  @test occursin("wall_top_offset = -0.19999999", src)
  @test occursin("top_offset=wall_top_offset", src)
  @test !occursin("top_offset=-0.19999999", src)
  @test Meta.parseall(src) isa Expr
  # Below-threshold occurrences are left untouched (no-op, same object).
  few = Expr(:block, [Expr(:call, :wall, :p, Expr(:kw, :top_offset, -0.5)) for _ in 1:2]...)
  @test extract_shared_dimensions(few) === few
  # Values inside protected constructs (family constructors) are never lifted.
  fams = Expr(:block, [Expr(:call, :door_family, "d", 1.0, 2.0, Expr(:kw, :inset, 0.05)) for _ in 1:6]...)
  @test extract_shared_dimensions(fams) === fams
  # Zero is never lifted even if frequent.
  zeros = Expr(:block, [Expr(:call, :wall, :p, Expr(:kw, :base_offset, 0.0)) for _ in 1:6]...)
  @test extract_shared_dimensions(zeros) === zeros
end

@testset "symbolize_angles" begin
  prog = Expr(:block,
    :(family_element(xy(1.0, 2.0), 1.5707963, level_0, fam_a)),
    :(family_element(xy(3.0, 4.0), 3.1415927, level_0, fam_b)),
    :(family_element(xy(5.0, 6.0), -1.5707963, level_0, fam_c)),
    :(family_element(xy(7.0, 8.0), 0.0, level_0, fam_d)),
    :(box(xyz(0.0, 0.0, 0.0), 3.1415927, 1.0, 1.0)))   # arg 3 is a size, must NOT be touched
  src = expr_to_string(symbolize_angles(prog))
  @test occursin("family_element(xy(1.0, 2.0), pi / 2, level_0, fam_a)", src)
  @test occursin("family_element(xy(3.0, 4.0), pi, level_0, fam_b)", src)
  @test occursin("family_element(xy(5.0, 6.0), -pi / 2, level_0, fam_c)", src)
  @test occursin("family_element(xy(7.0, 8.0), 0.0, level_0, fam_d)", src)   # 0 left as-is
  @test occursin("box(xyz(0.0, 0.0, 0.0), 3.1415927", src)                   # box size untouched
  @test Meta.parseall(src) isa Expr
end

@testset "infix arithmetic printing" begin
  @test _expr_str(:(a + 2 * b)) == "a + 2 * b"
  @test _expr_str(:((a + b) * c)) == "(a + b) * c"
  @test _expr_str(:(a - (b + c))) == "a - (b + c)"
  @test _expr_str(:(xs[i + 1])) == "xs[i + 1]"
end
