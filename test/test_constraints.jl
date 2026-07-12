# test_constraints.jl — regression tests for the constraint-algebra fixes (rank 5).
# The constraint/layout engine previously had no test coverage, which is why the
# inverted-severity and empty-input bugs survived.
using Test
using KhepriBase

@testset "Constraints" begin
  @testset "combine/either pick the MOST-severe level (CONSTRAINTS-1)" begin
    # The @enum is HARD=0 < SOFT=1 < PREFERENCE=2, so combining must use `min`,
    # not `max`, to return the most-severe level.
    hard = min_area(:office, 10)             # default severity HARD
    soft = max_aspect_ratio(:office, 2)      # default severity SOFT
    @test hard.severity == KhepriBase.HARD
    @test soft.severity == KhepriBase.SOFT
    @test KhepriBase.combine(hard, soft).severity == KhepriBase.HARD
    @test KhepriBase.either(hard, soft).severity == KhepriBase.HARD
    @test KhepriBase.combine(soft, soft).severity == KhepriBase.SOFT
  end

  @testset "combine/merge_constraints reject empty input (CONSTRAINTS-9)" begin
    @test_throws ErrorException KhepriBase.combine()
    @test_throws ErrorException KhepriBase.merge_constraints()
  end

  @testset "either() stamps combined severity onto emitted violations (T21)" begin
    # The chosen child's violations must carry the either-constraint's declared (most-severe)
    # severity, not the child's own — while KEEPING v.constraint_name (ConstraintFixer substring-
    # matches on it, so renaming would break a fixer keyed on the child's name).
    cat = first(instances(KhepriBase.ConstraintCategory))
    viol(nm) = KhepriBase.Violation(nm, KhepriBase.SOFT, cat, "tgt", "msg", 1.0, 0.0)
    a = KhepriBase.Constraint("a", KhepriBase.HARD, cat, ctx -> [viol("a")])
    b = KhepriBase.Constraint("b", KhepriBase.SOFT, cat, ctx -> [viol("b")])
    let vs = KhepriBase.either(a, b).check(nothing)
      @test length(vs) == 1
      @test vs[1].severity == KhepriBase.HARD          # combined, not the child's SOFT
      @test vs[1].constraint_name == "a"               # child name preserved
    end
  end
end
