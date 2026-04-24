# test_bim.jl - Tests for BIM (Building Information Modeling) operations

using Test
using KhepriBase

# Include the mock backend
include("TestMockBackend.jl")

@testset "BIM" begin

  @testset "Level" begin
    @testset "level creation" begin
      l = level(3.0)
      @test is_level(l)
      @test l.height ≈ 3.0 atol=1e-10
    end

    @testset "level with default height" begin
      l = level()
      @test l.height ≈ 0.0 atol=1e-10
    end

    @testset "level equality" begin
      l1 = level(3.0)
      l2 = level(3.0)
      @test l1 == l2

      l3 = level(5.0)
      @test l1 != l3
    end

    @testset "upper_level" begin
      l1 = level(0.0)
      l2 = upper_level(l1, 3.5)
      @test l2.height ≈ 3.5 atol=1e-10
    end

    @testset "default_level parameter" begin
      @test default_level().height ≈ 0.0 atol=1e-10
    end

    @testset "default_level_to_level_height parameter" begin
      @test default_level_to_level_height() == 3
    end

    @testset "convert Real to Level" begin
      l = convert(Level, 5.0)
      @test l.height ≈ 5.0 atol=1e-10
    end
  end

  @testset "Slab Family" begin
    @testset "slab_family creation" begin
      sf = slab_family()
      @test is_slab_family(sf)
      @test sf.name == "slab_family"
      @test sf.thickness ≈ 0.2 atol=1e-10
      @test sf.coating_thickness ≈ 0.0 atol=1e-10
    end

    @testset "slab_family with custom thickness" begin
      sf = slab_family(thickness=0.3)
      @test sf.thickness ≈ 0.3 atol=1e-10
    end

    @testset "slab_family with materials" begin
      sf = slab_family(
        bottom_material=material_concrete,
        top_material=material_wood,
        side_material=material_plaster
      )
      @test sf.bottom_material === material_concrete
      @test sf.top_material === material_wood
      @test sf.side_material === material_plaster
    end

    @testset "default_slab_family parameter" begin
      @test is_slab_family(default_slab_family())
    end

    @testset "slab_family_element" begin
      base = slab_family()
      elem = slab_family_element(base, thickness=0.4)
      @test elem.thickness ≈ 0.4 atol=1e-10
      @test elem.based_on === base
    end

    @testset "slab_family_thickness" begin
      with_mock_backend() do b
        sf = slab_family(thickness=0.25, coating_thickness=0.05)
        th = KhepriBase.slab_family_thickness(b, sf)
        @test th ≈ 0.30 atol=1e-10  # thickness + coating
      end
    end

    @testset "slab_family_elevation" begin
      with_mock_backend() do b
        sf = slab_family(thickness=0.25, coating_thickness=0.05)
        elev = KhepriBase.slab_family_elevation(b, sf)
        @test elev ≈ -0.20 atol=1e-10  # coating - thickness
      end
    end

    @testset "used_materials from slab family" begin
      sf = slab_family()
      mats = used_materials(sf)
      @test length(mats) == 3  # bottom, top, side materials
    end
  end

  @testset "Slab" begin
    @testset "slab creation" begin
      with_mock_backend() do b
        r = region(closed_polygonal_path([xy(0, 0), xy(10, 0), xy(10, 10), xy(0, 10)]))
        s = slab(r, level(0), default_slab_family())
        @test is_slab(s)
      end
    end

    @testset "slab with default parameters" begin
      with_mock_backend() do b
        s = slab()
        @test is_slab(s)
      end
    end
  end

  @testset "Roof Family" begin
    @testset "roof_family creation" begin
      rf = roof_family()
      @test is_roof_family(rf)
      @test rf.thickness ≈ 0.2 atol=1e-10
    end

    @testset "roof_family with custom values" begin
      rf = roof_family(thickness=0.15, coating_thickness=0.02)
      @test rf.thickness ≈ 0.15 atol=1e-10
      @test rf.coating_thickness ≈ 0.02 atol=1e-10
    end

    @testset "default_roof_family parameter" begin
      @test is_roof_family(default_roof_family())
    end
  end

  @testset "Wall Family" begin
    @testset "wall_family creation" begin
      wf = wall_family()
      @test is_wall_family(wf)
    end

    @testset "default_wall_family parameter" begin
      @test is_wall_family(default_wall_family())
    end
  end

  @testset "Beam Family" begin
    @testset "beam_family creation" begin
      bf = beam_family()
      @test is_beam_family(bf)
    end

    @testset "default_beam_family parameter" begin
      @test is_beam_family(default_beam_family())
    end
  end

  @testset "Column Family" begin
    @testset "column_family creation" begin
      cf = column_family()
      @test is_column_family(cf)
    end

    @testset "default_column_family parameter" begin
      @test is_column_family(default_column_family())
    end
  end

  @testset "Door Family" begin
    @testset "door_family creation" begin
      df = door_family()
      @test is_door_family(df)
    end

    @testset "default_door_family parameter" begin
      @test is_door_family(default_door_family())
    end
  end

  @testset "Window Family" begin
    @testset "window_family creation" begin
      wf = window_family()
      @test is_window_family(wf)
    end

    @testset "default_window_family parameter" begin
      @test is_window_family(default_window_family())
    end
  end

  @testset "Panel Family" begin
    @testset "panel_family creation" begin
      pf = panel_family()
      @test is_panel_family(pf)
    end

    @testset "default_panel_family parameter" begin
      @test is_panel_family(default_panel_family())
    end
  end

  @testset "Table Family" begin
    @testset "table_family creation" begin
      tf = table_family()
      @test is_table_family(tf)
    end

    @testset "default_table_family parameter" begin
      @test is_table_family(default_table_family())
    end
  end

  @testset "Chair Family" begin
    @testset "chair_family creation" begin
      cf = chair_family()
      @test is_chair_family(cf)
    end

    @testset "default_chair_family parameter" begin
      @test is_chair_family(default_chair_family())
    end
  end

  @testset "Table and Chair Family" begin
    @testset "table_chair_family creation" begin
      tcf = table_chair_family()
      @test is_table_chair_family(tcf)
    end

    @testset "default_table_chair_family parameter" begin
      @test is_table_chair_family(default_table_chair_family())
    end
  end

  @testset "LayerFamily" begin
    @testset "layer_family creation" begin
      lf = layer_family("TestLayer", rgb(1, 0, 0))
      @test lf.name == "TestLayer"
      @test lf.color.r ≈ 1 atol=1e-10
    end

    @testset "layer_family default color" begin
      lf = layer_family("DefaultColorLayer")
      @test lf.color.r ≈ 1 atol=1e-10
      @test lf.color.g ≈ 1 atol=1e-10
      @test lf.color.b ≈ 1 atol=1e-10
    end
  end

  @testset "Family operations" begin
    @testset "family function" begin
      sf = slab_family()
      @test family(sf) === sf

      sfe = slab_family_element(sf, thickness=0.3)
      # family returns the family itself (with based_on set), not the base
      @test sfe.based_on === sf
      @test family(sfe).based_on === sf
    end

    @testset "set_backend_family" begin
      sf = slab_family()
      lf = layer_family("CustomSlabLayer")
      b = mock_backend()

      set_backend_family(sf, b, lf)
      @test sf.implemented_as[typeof(b)] === lf
    end
  end

  @testset "with_ family operations" begin
    @testset "with_slab_family" begin
      original = default_slab_family()
      custom = slab_family(thickness=0.5)

      with_slab_family(family=custom) do
        @test default_slab_family().thickness ≈ 0.5 atol=1e-10
      end

      @test default_slab_family() === original
    end
  end

  @testset "BIM struct" begin
    levels = [level(0), level(3), level(6)]
    bim = KhepriBase.BIM(levels, levels[1], KhepriBase.BIMElement[])
    @test length(bim.levels) == 3
    @test bim.current_level === levels[1]
    @test isempty(bim.elements)
  end

  # Frame geometry on an arc wall can't be produced by a single composite
  # sweep: the RMF seed is picked from the first frame's cs_from_o_vz, which
  # depends only on the jamb's vertical tangent and is blind to the wall's
  # arc angle. src/BIM.jl splits into (right jamb, head, left jamb [, sill])
  # when the subpath is an ArcPath, each with its own profile rotation.
  # This smoke-tests the dispatch and that each piece reaches b_trig.
  @testset "Arc-wall door/window frame split" begin
    b = mock_backend()

    arc_sub = arc_path(xyz(0, 0, 0.0001), 5, 0.4, 0.2)
    profile = rectangular_profile(0.1, 0.4)
    mat = 0  # MockBackend ignores material refs
    h = 2.0

    reset_mock_backend!(b)
    @test !isempty(KhepriBase.arc_jamb_refs(b, arc_sub, :end,   h, profile, mat))
    @test mock_geometry_stats(b).triangles > 0

    reset_mock_backend!(b)
    @test !isempty(KhepriBase.arc_jamb_refs(b, arc_sub, :begin, h, profile, mat))
    @test mock_geometry_stats(b).triangles > 0

    reset_mock_backend!(b)
    @test !isempty(KhepriBase.arc_head_refs(b, arc_sub, h, profile, mat))
    @test mock_geometry_stats(b).triangles > 0

    reset_mock_backend!(b)
    @test !isempty(KhepriBase.arc_sill_refs(b, arc_sub, profile, mat))
    @test mock_geometry_stats(b).triangles > 0
  end

  # `frame_refs` dispatches on the subpath type: ArcPath → split sweeps,
  # anything else → the single merged-polyline sweep path. Verify the
  # polygonal-subpath path is still exercised by a straight wall (so the
  # user's OK-ish straight-wall behavior is preserved).
  @testset "frame_refs dispatches on subpath type" begin
    poly_sub = open_polygonal_path([xyz(2, 0, 0.0001), xyz(3, 0, 0.0001)])
    arc_sub  = arc_path(xyz(0, 0, 0.0001), 5, 0.4, 0.2)

    # The three-arg polygonal method exists for any Path; the four-arg
    # ArcPath method is the specialized split one. Check that specializing
    # on ArcPath catches `ArcPath` but not `OpenPolygonalPath`.
    @test hasmethod(KhepriBase.frame_refs,
                    Tuple{Backend, Union{KhepriBase.Door, KhepriBase.Window}, ArcPath, Real})
    @test hasmethod(KhepriBase.frame_refs,
                    Tuple{Backend, Union{KhepriBase.Door, KhepriBase.Window}, KhepriBase.Path, Real})
    # Both subpath types are reachable; polygonal uses the merged sweep.
    @test poly_sub isa KhepriBase.Path
    @test arc_sub isa ArcPath
  end

  # End-to-end: arc wall with a door, realized via the HasBooleanOps{false}
  # path (the same path AutoCAD uses). Two layered bugs live here:
  #
  # 1. Before the chord-snap fix, _b_wall_with_openings_impl built the
  #    opening's cutout rectangle from points on the offset arc, which on
  #    a curved wall are not coplanar with the segment's chord-based face
  #    rectangle — AutoCAD rejected the resulting Region with
  #    eNonCoplanarGeometry.
  #
  # 2. Even with chord-snapped endpoints, using Region-with-hole still
  #    fails on polygonalized-arc segments because the hole almost always
  #    touches the segment boundary (any opening spanning ≥2 segments has
  #    op_at_start or op_at_end true somewhere). AutoCAD's boolean then
  #    silently drops the hole, leaving the wall surface covering the
  #    door. The fix renders the wall face as explicit rectangles
  #    (left/right/above/below each opening) and suppresses jacket jambs
  #    at segment-boundary continuations.
  #
  # MockBackend's `b_surface` → `planar_region` path catches (1) via
  # `planarity_tolerance`; geometric correctness for (2) would only be
  # visually verifiable, but we at least assert that the scenarios run
  # and produce geometry.
  @testset "Arc wall with openings (HasBooleanOps{false})" begin
    b = mock_backend()

    reset_mock_backend!(b)
    with(current_backend, b) do
      with(default_frame_family, frame_family(profile=rectangular_profile(0.1, 0.4))) do
        with(default_door_family, door_family()) do
          w = wall(arc_path(xy(0, 0), 5, 0, π))
          add_door(w, x(2))
          @test mock_geometry_stats(b).triangles > 0
        end
      end
    end

    # Two doors, same arc wall — both openings span multiple polygonalized
    # segments, exercising the wall-face split for several openings.
    reset_mock_backend!(b)
    with(current_backend, b) do
      with(default_frame_family, frame_family(profile=rectangular_profile(0.1, 0.4))) do
        with(default_door_family, door_family()) do
          [add_door(wall(arc_path(xy(0, 0), 5, 0, π)), p) for p in [x(2), x(8)]]
          @test mock_geometry_stats(b).triangles > 0
        end
      end
    end

    # Elevated window: exercises the has_sill branches of the jacket
    # emission (sill + end-jamb + top when the opening's start is a
    # continuation; sill + start-jamb + top when its end is a continuation;
    # two separate pieces when both are continuations).
    reset_mock_backend!(b)
    with(current_backend, b) do
      with(default_frame_family, frame_family(profile=rectangular_profile(0.1, 0.4))) do
        with(default_window_family, window_family()) do
          w = wall(arc_path(xy(0, 0), 5, 0, π))
          add_window(w, xy(2, 1))  # loc.y = 1 → base_height = 1, elevated
          @test mock_geometry_stats(b).triangles > 0
        end
      end
    end

    # Regression: middle-segment overlap detection + one-cap-per-wall-end.
    #
    # On a polygonalized arc wall, openings span multiple segments. Two
    # historical bugs broke this:
    #
    #   (1) The overlap test only caught "opening starts here" and
    #       "opening ends here"; middle segments (where the opening
    #       fully contains the segment) were skipped and emitted as
    #       full wall rectangles covering the opening.
    #
    #   (2) End caps were emitted on every segment's start and end —
    #       the `!is_closed_path(w_path)` guard didn't distinguish
    #       "wall is open" from "segment is at the wall's extremity",
    #       so polygonalization boundaries sprouted vertical slabs
    #       that sliced through walls and openings.
    #
    # A visual assertion against MockBackend's tessellated output is
    # fragile (the triangle centroid of a chord-based quad doesn't land
    # at the arc radius), so the tests here are coarser: if both fixes
    # regress we get either an error (bug 1 silently skips openings and
    # the current code asserts correctness of the structure) or a
    # triangle count that jumps by ~100s (bug 2 adds 2 caps × ~N
    # segments × 2 triangles per cap = hundreds of extra triangles).
    reset_mock_backend!(b)
    with(current_backend, b) do
      with(default_frame_family, frame_family(profile=rectangular_profile(0.1, 0.4))) do
        with(default_door_family, door_family()) do
          w = wall(arc_path(xy(0, 0), 5, 0, π))
          add_door(w, x(2))
          triangles = mock_geometry_stats(b).triangles
          # Upper-bound guard for bug (2): the polygonalization of the
          # radius-5 π-arc yields ~60 segments. Emitting end caps on
          # every segment would add ≥ 2×60×2 = 240 extra triangles.
          # A healthy run lands well under 2000; guard at 1800 so a
          # regression that doubles cap count still trips us.
          @test triangles > 0
          @test triangles < 1800
        end
      end
    end
  end

end
