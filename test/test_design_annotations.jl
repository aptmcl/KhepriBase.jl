# test_design_annotations.jl — Level-2 design annotations lowered by build(l::Layout).
#
# The declarative annotations (connect_spaces, connect_exterior, disconnect,
# no_windows) must produce real openings when the layout is built, the
# combinator options (align, slab_between) must be honored by the layout
# engine, the tag_*_family combinators must carry Family objects through to
# built elements, and everything still unimplemented must error loudly.
using Test
using KhepriBase

@testset "Design annotations" begin

  #=== connect_spaces ===#

  @testset "connect_spaces door between :a and :b" begin
    let desc = connect_spaces(room(:a, :living_room, 4.0, 4.0) |
                              room(:b, :kitchen, 4.0, 4.0), :a, :b),
        l = layout(desc),
        r = build(l),
        sa = find_space(l, :a),
        sb = find_space(l, :b)
      @test length(r.doors) == 1
      @test !isempty(space_doors(r[1], sa))
      @test !isempty(space_doors(r[1], sb))
      # Lowering is pure: rebuilding the same Layout must not
      # accumulate connections (validate(l) rebuilds repeatedly).
      @test length(build(l).doors) == 1
      @test isempty(l.storeys[1].connections)  # storey untouched
    end
  end

  @testset "connect_spaces width override reflected in the door family" begin
    let desc = connect_spaces(room(:a, :living_room, 4.0, 4.0) |
                              room(:b, :kitchen, 4.0, 4.0), :a, :b;
                              width=1.2),
        r = build(layout(desc)),
        door = r.doors[1]
      @test door.family.width ≈ 1.2
      # Unset dims default from the default family.
      @test door.family.height ≈ default_door_family().height
    end
  end

  @testset "connect_spaces arch suppresses the shared wall" begin
    let base = room(:a, :living_room, 4.0, 4.0) |
               room(:b, :kitchen, 4.0, 4.0),
        r_plain = build(layout(base)),
        r_arch = build(layout(connect_spaces(base, :a, :b; kind=:arch)))
      # The interior wall is elided, so the arch build has fewer walls
      # (the exterior loop merges into a single chain).
      @test length(r_arch.walls) < length(r_plain.walls)
      # And the shared edge becomes a virtual (element-less) boundary.
      @test any(b -> b.kind == :virtual && isnothing(b.element),
                r_arch[1].boundaries)
      @test isempty(r_arch.doors) && isempty(r_arch.windows)
    end
  end

  #=== connect_exterior ===#

  @testset "connect_exterior face=:south places a door on the y≈0 edge" begin
    let desc = connect_exterior(room(:a, :living_room, 6.0, 4.0), :a;
                                kind=:door, face=:south),
        r = build(layout(desc))
      @test length(r.doors) == 1
      let door = r.doors[1],
          b = only(filter(b -> b.element === door, r[1].boundaries))
        @test abs(cy(b.p1)) < 1e-6
        @test abs(cy(b.p2)) < 1e-6
      end
    end
  end

  @testset "connect_exterior count=2 places two windows" begin
    let desc = connect_exterior(room(:a, :living_room, 6.0, 4.0), :a;
                                kind=:window, face=:north, count=2),
        r = build(layout(desc))
      @test length(r.windows) == 2
    end
  end

  @testset "connect_exterior face=:auto picks an exterior edge" begin
    let desc = connect_exterior(room(:a, :living_room, 6.0, 4.0), :a;
                                kind=:window),
        r = build(layout(desc))
      @test length(r.windows) == 1
    end
  end

  #=== disconnect / no_windows (phase-2 removal) ===#

  @testset "disconnect removes a connect_spaces door (any nesting order)" begin
    let base = room(:a, :living_room, 4.0, 4.0) |
               room(:b, :kitchen, 4.0, 4.0),
        # disconnect is applied OUTSIDE the connect annotation...
        d1 = disconnect(connect_spaces(base, :a, :b), :a, :b),
        # ...and also INSIDE (wrapped first); both must win.
        d2 = connect_spaces(disconnect(base, :a, :b), :a, :b)
      @test isempty(build(layout(d1)).doors)
      @test isempty(build(layout(d2)).doors)
    end
  end

  @testset "disconnect of a never-connected pair is a no-op" begin
    let desc = disconnect(room(:a, :living_room, 4.0, 4.0) |
                          room(:b, :kitchen, 4.0, 4.0), :a, :b),
        r = build(layout(desc))
      @test isempty(r.doors) && isempty(r.windows)
    end
  end

  @testset "no_windows cancels a connect_exterior window" begin
    let base = connect_exterior(room(:a, :living_room, 6.0, 4.0), :a;
                                kind=:window, face=:south),
        r = build(layout(no_windows(base, :a)))
      @test isempty(r.windows)
    end
  end

  #=== loud errors ===#

  @testset "unknown ids and invalid annotation arguments error" begin
    let base = room(:a, :living_room, 4.0, 4.0) |
               room(:b, :kitchen, 4.0, 4.0)
      @test_throws ArgumentError build(layout(connect_spaces(base, :a, :zzz)))
      @test_throws ArgumentError build(layout(disconnect(base, :zzz, :b)))
      @test_throws ArgumentError build(layout(no_windows(base, :zzz)))
      # Unsupported kinds.
      @test_throws ArgumentError build(layout(connect_spaces(base, :a, :b; kind=:vent)))
      @test_throws ArgumentError build(layout(connect_exterior(base, :a; kind=:arch)))
      # Arches carry no family, so width/height are meaningless.
      @test_throws ArgumentError build(layout(connect_spaces(base, :a, :b; kind=:arch, width=1.8)))
      # Bad face / count on exterior connections.
      @test_throws ArgumentError build(layout(connect_exterior(base, :a; face=:up)))
      @test_throws ArgumentError build(layout(connect_exterior(base, :a; count=0)))
    end
  end

  @testset "cross-storey connect_spaces errors" begin
    let desc = connect_spaces(room(:a, :living_room, 4.0, 4.0) ^
                              room(:b, :bedroom, 4.0, 4.0), :a, :b)
      @test_throws ArgumentError build(layout(desc))
    end
  end

  #=== align ===#

  @testset "beside_x align offsets the shallower room" begin
    let a = room(:a, :living_room, 4.0, 4.0),
        b = room(:b, :kitchen, 3.0, 2.0)
      # Default :start keeps the historical flush-at-y=0 placement.
      @test find_space(layout(beside_x(a, b)), :b).origin_y ≈ 0.0
      let l = layout(beside_x(a, b; align=:center))
        @test find_space(l, :a).origin_y ≈ 0.0
        @test find_space(l, :b).origin_y ≈ 1.0   # (4 - 2) / 2
      end
      let l = layout(beside_x(a, b; align=:end))
        @test find_space(l, :b).origin_y ≈ 2.0   # 4 - 2
      end
    end
  end

  @testset "beside_y align offsets the narrower room" begin
    let a = room(:a, :living_room, 4.0, 3.0),
        b = room(:b, :kitchen, 2.0, 3.0),
        l = layout(beside_y(a, b; align=:center))
      @test find_space(l, :b).origin_x ≈ 1.0     # (4 - 2) / 2
      @test find_space(l, :b).origin_y ≈ 3.0     # behind a
    end
  end

  @testset "unimplemented/invalid combinator options error" begin
    let a = room(:a, :living_room, 4.0, 4.0),
        b = room(:b, :kitchen, 4.0, 4.0)
      @test_throws ArgumentError beside_x(a, b; shared_wall=false)
      @test_throws ArgumentError beside_y(a, b; shared_wall=false)
      @test_throws ArgumentError beside(a, b; axis=:y, shared_wall=false)
      @test_throws ArgumentError beside_x(a, b; align=:middle)
    end
  end

  #=== slab_between ===#

  @testset "above(...; slab_between=false) suppresses the interface slab" begin
    let g = room(:ground, :living_room, 4.0, 4.0, height=3.0),
        u = room(:upper, :bedroom, 4.0, 4.0, height=3.0)
      # Default: both storeys get their floor slab.
      let r = build(layout(above(g, u)))
        @test length(r[1].slabs) == 1
        @test length(r[2].slabs) == 1
      end
      let l = layout(above(g, u; slab_between=false)),
          r = build(l)
        @test length(r[1].slabs) == 1
        @test length(r[2].slabs) == 0
        # The Storey flag stays truthful when the whole level opts out.
        @test !l.storeys[2].generate_slabs
      end
    end
  end

  #=== tag_wall_family / tag_slab_family ===#

  @testset "tag_slab_family with a SlabFamily object reaches the built slab" begin
    let fam = slab_family(thickness=0.5),
        desc = tag_slab_family(room(:a, :living_room, 4.0, 4.0), fam),
        r = build(layout(desc))
      @test length(r.slabs) == 1
      @test r.slabs[1].family === fam
    end
  end

  @testset "tag_wall_family with a WallFamily object reaches the built walls" begin
    let fam = wall_family(thickness=0.4),
        base = room(:a, :living_room, 4.0, 4.0) |
               room(:b, :kitchen, 4.0, 4.0)
      # Both rooms tagged with the SAME family: every wall (exterior
      # and the shared interior one) carries it.
      let r = build(layout(tag_wall_family(base, fam)))
        @test all(w -> w.family === fam, r.walls)
      end
      # Only one room tagged: its exterior walls carry the tag, while
      # the interior wall (tags disagree) and the untagged room's
      # exterior fall back to the storey default.
      let desc = tag_wall_family(room(:a, :living_room, 4.0, 4.0), fam) |
                 room(:b, :kitchen, 4.0, 4.0),
          r = build(layout(desc))
        @test any(w -> w.family === fam, r.walls)
        @test any(w -> w.family !== fam, r.walls)
      end
    end
  end

  @testset "Symbol-form family tags error loudly" begin
    let a = room(:a, :living_room, 4.0, 4.0)
      @test_throws ArgumentError tag_wall_family(a, :brick)
      @test_throws ArgumentError tag_slab_family(a, :concrete)
      @test_throws ArgumentError tag_wall_family(a, "brick")
    end
  end

end
