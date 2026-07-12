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

  #=
  Regression (T16): the declarative engine used three unsafe Dict idioms —
  bare inserts that silently overwrote colliding ids, a HeightOverride that
  re-stamped EVERY space at its z (clobbering sibling heights placed by an
  outer combinator), and Scaled/Mirrored that flattened polar arcs to their
  bounding rectangle. Pin the corrected behaviour.
  =#
  @testset "duplicate space id warns instead of silently dropping" begin
    # Two rooms resolving to the same scoped id: last-writer-wins is kept, but
    # the collision is now surfaced with a @warn rather than a silent drop.
    let dup = beside_x(room(:dup, :living_room, 4.0, 3.0, height=3.0),
                       room(:dup, :bedroom, 2.0, 2.0, height=2.5))
      @test_logs (:warn,) match_mode=:any layout(dup)
      let l = (@test_logs (:warn,) match_mode=:any layout(dup))
        @test find_space(l, :dup).kind == :bedroom   # last writer survives
      end
    end
  end

  @testset "with_height overrides only its own subtree, not siblings" begin
    # `a` is placed first by beside_x at z=0; the with_height wrapper around `b`
    # (also at z=0) must NOT re-stamp `a`'s height.
    let a = room(:a, :living_room, 4.0, 3.0, height=3.0),
        b = room(:b, :bedroom, 4.0, 3.0, height=2.8),
        l = layout(beside_x(a, with_height(b, 9.0)))
      @test find_space(l, :a).height ≈ 3.0    # sibling untouched (was 9.0 before)
      @test find_space(l, :b).height ≈ 9.0    # target overridden
    end
  end

  @testset "scale/mirror reject polar spaces instead of flattening them" begin
    let sector = polar_envelope(xy(0, 0), 2.0, 5.0, 0.0, pi/2, 3.0; id=:sector)
      @test_throws ErrorException layout(scale(sector, 2.0))
      @test_throws ErrorException layout(mirror_x(sector))
      # A rectangular child still scales/mirrors fine.
      @test layout(scale(room(:r, :zone, 4.0, 3.0, height=3.0), 2.0)) isa Layout
    end
  end
end
