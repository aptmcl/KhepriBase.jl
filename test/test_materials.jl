# test_materials.jl - Tests for material and layer system

using Test
using KhepriBase

# Include the mock backend
include("TestMockBackend.jl")

@testset "Materials" begin

  @testset "Layer creation" begin
    @testset "base_layer" begin
      bl = base_layer()
      @test is_base_layer(bl)
    end

    @testset "layer with defaults" begin
      with_mock_backend() do b
        l = layer()
        @test is_layer(l)
        @test layer_name(l) == "Layer"
        @test layer_active(l) == true
      end
    end

    @testset "layer with custom name" begin
      with_mock_backend() do b
        l = layer("MyLayer")
        @test layer_name(l) == "MyLayer"
      end
    end

    @testset "layer with custom color" begin
      with_mock_backend() do b
        l = layer("ColorLayer", true, rgba(1, 0, 0, 1))
        @test layer_name(l) == "ColorLayer"
        @test layer_color(l).r ≈ 1 atol=1e-10
        @test layer_color(l).g ≈ 0 atol=1e-10
        @test layer_color(l).b ≈ 0 atol=1e-10
      end
    end

    @testset "inactive layer" begin
      with_mock_backend() do b
        l = layer("InactiveLayer", false)
        @test layer_active(l) == false
      end
    end
  end

  @testset "Material creation" begin
    @testset "material_in_layer" begin
      with_mock_backend() do b
        l = layer("TestLayer")
        m = material_in_layer(l)
        @test is_material(m)
        @test m.layer === l
      end
    end

    @testset "material with name" begin
      with_mock_backend() do b
        m = material("TestMaterial")
        @test is_material(m)
        @test m.layer.name == "TestMaterial"
      end
    end
  end

  @testset "Predefined materials" begin
    @testset "material_point" begin
      @test is_material(material_point)
      @test material_point.layer.name == "Points"
    end

    @testset "material_curve" begin
      @test is_material(material_curve)
      @test material_curve.layer.name == "Curves"
    end

    @testset "material_surface" begin
      @test is_material(material_surface)
      @test material_surface.layer.name == "Surfaces"
    end

    @testset "material_basic" begin
      @test is_material(material_basic)
      @test material_basic.layer.name == "Basic"
    end

    @testset "material_glass" begin
      @test is_material(material_glass)
      @test material_glass.layer.name == "Glass"
    end

    @testset "material_metal" begin
      @test is_material(material_metal)
      @test material_metal.layer.name == "Metal"
    end

    @testset "material_wood" begin
      @test is_material(material_wood)
      @test material_wood.layer.name == "Wood"
    end

    @testset "material_concrete" begin
      @test is_material(material_concrete)
      @test material_concrete.layer.name == "Concrete"
    end

    @testset "material_plaster" begin
      @test is_material(material_plaster)
      @test material_plaster.layer.name == "Plaster"
    end

    @testset "material_grass" begin
      @test is_material(material_grass)
      @test material_grass.layer.name == "Grass"
    end

    @testset "material_clay" begin
      @test is_material(material_clay)
      @test material_clay.layer.name == "Clay"
    end
  end

  @testset "Default material parameters" begin
    @testset "default_point_material" begin
      @test default_point_material() === material_point
    end

    @testset "default_curve_material" begin
      @test default_curve_material() === material_curve
    end

    @testset "default_surface_material" begin
      @test default_surface_material() === material_surface
    end

    @testset "default_material" begin
      @test default_material() === material_basic
    end
  end

  @testset "Changing default materials" begin
    @testset "with default_material" begin
      original = default_material()
      with(default_material, material_wood) do
        @test default_material() === material_wood
      end
      @test default_material() === original
    end
  end

  @testset "Material merging" begin
    @testset "merge_materials" begin
      m1 = material("Material1")
      m2 = material("Material2")
      merged = merge_materials(m1, m2)
      @test is_material(merged)
      @test occursin("Material1", merged.layer.name)
      @test occursin("Material2", merged.layer.name)
    end
  end

  @testset "Material assignment to shapes" begin
    with_mock_backend() do b
      @testset "default material for points" begin
        p = point(u0())
        @test p.material === default_point_material()
      end

      @testset "default material for curves" begin
        l = line([u0(), ux()])
        @test l.material === default_curve_material()

        c = circle(u0(), 5)
        @test c.material === default_curve_material()
      end

      @testset "default material for surfaces" begin
        sc = surface_circle(u0(), 5)
        @test sc.material === default_surface_material()
      end

      @testset "default material for solids" begin
        s = sphere(u0(), 5)
        @test s.material === default_material()
      end

      @testset "custom material assignment" begin
        c = circle(u0(), 5, material=material_metal)
        @test c.material === material_metal
      end
    end
  end

  @testset "used_materials from shapes" begin
    with_mock_backend() do b
      @testset "single material" begin
        c = circle(u0(), 5)
        mats = used_materials(c)
        @test length(mats) == 1
        @test mats[1] === c.material
      end
    end
  end

  @testset "material_ref" begin
    with_mock_backend() do b
      m = material("TestMat")
      ref = material_ref(b, m)
      # Should return a reference (in mock backend, it's 0 for materials)
      @test ref isa Integer
    end
  end

  @testset "BackendParameter" begin
    @testset "empty BackendParameter" begin
      bp = BackendParameter()
      @test bp isa BackendParameter
    end
  end

  @testset "material_as_layer parameter" begin
    @test material_as_layer() == false
    with(material_as_layer, true) do
      @test material_as_layer() == true
    end
    @test material_as_layer() == false
  end

  @testset "standard_material" begin
    with_mock_backend() do b
      @testset "creation with defaults" begin
        sm = standard_material()
        @test sm isa Material
        @test is_standard_material(sm)
        @test standard_material_name(sm) == "Material"
      end

      @testset "creation with custom base_color" begin
        sm = standard_material(base_color=rgba(1, 0, 0, 1))
        @test standard_material_base_color(sm) == rgba(1, 0, 0, 1)
      end

      @testset "assignment to shape" begin
        sm = standard_material(base_color=rgba(1, 0, 0, 1))
        s = sphere(u0(), 5, material=sm)
        @test s.material === sm
      end

      @testset "material_ref returns valid reference" begin
        sm = standard_material(base_color=rgba(0, 1, 0, 1))
        ref = material_ref(b, sm)
        @test ref isa Integer
        @test ref != 0  # not void_ref
      end
    end
  end

end
