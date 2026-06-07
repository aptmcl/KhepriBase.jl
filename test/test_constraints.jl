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
end
