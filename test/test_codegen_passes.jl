# Headless tests for the codegen abstraction/parametrization passes (sort_statements,
# extract_functions, detect_level_repetition, parametrize_levels, symbolize_ranges,
# unify_constants, wrap_program_in_function) and the printer features they rely on.
using Test
using KhepriBase
using KhepriBase: sort_statements, extract_functions, detect_level_repetition,
  parametrize_levels, symbolize_ranges, symbolize_angles, extract_shared_dimensions,
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

@testset "sectionalize_by_storey: per-storey anchor" begin
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
  @test occursin("function storey_0(p0=xy(-23.0, -81.0))", src)   # SW plan corner of storey 0
  @test occursin("p0 + vxy(0.0, 0.0)", src)                       # SW vertex → zero offset
  @test occursin("p0 + vxy(3.0, 6.0)", src)                       # far vertex → clean local delta
  @test occursin("add_door(__wall, xy(1.5, 0.0), door_fam)", src) # local door coord NOT rebased
  @test occursin("function storey_1(p0=xy(-22.0, -80.0))", src)
  @test occursin("storey_0()", src)                               # called with default anchor
  @test Meta.parseall(src) isa Expr
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
  @test occursin("opening_frame = frame_family(\"frame_family\", rectangular_path(xy(-0.08, -0.08), 0.16, 0.16))", src)
  @test occursin("door_a = door_family(\"A\", 1.4, 2.0, 0.05, opening_frame)", src)
  @test occursin("window_c = window_family(\"C\", 1.0, 1.0, 0.05, opening_frame)", src)
  @test !occursin("0.05, frame_family(", src)             # no inline frame left behind
  @test Meta.parseall(src) isa Expr
  # A frame used only once stays inline (no-op, same object).
  single = Expr(:block, Expr(:(=), :door_x, Expr(:call, :door_family, "X", 1.0, 2.0, 0.05, frame)))
  @test hoist_opening_frames(single) === single
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
  @test occursin("family_element(xy(5.0, 6.0), -(pi / 2), level_0, fam_c)", src)
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
