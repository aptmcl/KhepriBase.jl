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

end
