# Headless tests for the codegen abstraction/parametrization passes (sort_statements,
# extract_functions, detect_level_repetition, parametrize_levels, symbolize_ranges,
# unify_constants, wrap_program_in_function) and the printer features they rely on.
using Test
using KhepriBase
using KhepriBase: sort_statements, extract_functions, detect_level_repetition,
  parametrize_levels, symbolize_ranges, unify_constants, wrap_program_in_function,
  loop_rerolling, extract_levels, extract_families, expr_to_string, codegen_emit_timestamp,
  FamilyMeta, _expr_str

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

@testset "infix arithmetic printing" begin
  @test _expr_str(:(a + 2 * b)) == "a + 2 * b"
  @test _expr_str(:((a + b) * c)) == "(a + b) * c"
  @test _expr_str(:(a - (b + c))) == "a - (b + c)"
  @test _expr_str(:(xs[i + 1])) == "xs[i + 1]"
end
