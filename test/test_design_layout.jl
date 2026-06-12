# test_design_layout.jl — Level 2 (SpaceDesc) → Level 1 (Layout) compilation.
using Test
using KhepriBase

@testset "Design layout" begin
  #=
  Regression: the above()/^ docstrings used to claim "`a` sits on top of
  `b`", inverted relative to the implementation (Above stores below=a,
  above=b and _layout! places `above` at z + height(below)).  Pin the
  actual semantics — arguments read bottom-to-top — so docs and code can
  never drift apart silently again.
  =#
  @testset "above(a, b) places b on top of a" begin
    let a = room(:a, :living_room, 4.0, 3.0, height=3.0),
        b = room(:b, :bedroom, 4.0, 3.0, height=2.8),
        l = layout(above(a, b)),
        za = find_space(l, :a).origin_z,
        zb = find_space(l, :b).origin_z
      @test za == 0.0
      @test zb ≈ 3.0     # b starts where a's height ends
      @test zb > za      # b is the upper storey
    end
  end

  @testset "^ is right-associative and reads bottom-to-top" begin
    let ground = room(:ground, :living_room, 4.0, 3.0, height=3.0),
        first_floor = room(:first, :bedroom, 4.0, 3.0, height=3.0),
        second_floor = room(:second, :bedroom, 4.0, 3.0, height=3.0),
        l = layout(ground ^ first_floor ^ second_floor)
      @test find_space(l, :ground).origin_z == 0.0
      @test find_space(l, :first).origin_z ≈ 3.0
      @test find_space(l, :second).origin_z ≈ 6.0
    end
  end
end
