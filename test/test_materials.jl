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
        @test layer_visible(l) == true
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

    @testset "invisible layer" begin
      with_mock_backend() do b
        l = layer("InvisibleLayer", false)
        @test layer_visible(l) == false
      end
    end
  end

  @testset "Material creation" begin
    @testset "material from layer" begin
      with_mock_backend() do b
        l = layer("TestLayer")
        m = material(l)
        @test is_layer_material(m)
        @test m.layer === l
      end
    end

    @testset "material from name (PBR)" begin
      m = material("TestMaterial")
      @test is_pbr_material(m)
      @test pbr_material_name(m) == "TestMaterial"
    end

    @testset "layer_material from name" begin
      with_mock_backend() do b
        m = layer_material("TestMaterial")
        @test is_layer_material(m)
        @test m.layer.name == "TestMaterial"
      end
    end
  end

  @testset "Material accessor functions" begin
    @testset "material_color" begin
      @testset "PbrMaterial" begin
        @test material_color(material_glass) == rgba(0.95, 0.95, 1.0, 0.3)
        @test material_color(material_metal) == rgba(0.8, 0.8, 0.85, 1.0)
      end

      @testset "MaterialInLayer" begin
        with_mock_backend() do b
          l = layer("TestLayer", true, rgba(1, 0, 0, 1))
          m = material(l)
          @test material_color(m) == rgba(1, 0, 0, 1)
        end
      end
    end

    @testset "material_name" begin
      @testset "PbrMaterial" begin
        @test material_name(material_glass) == "Glass"
        @test material_name(material_point) == "Points"
      end

      @testset "MaterialInLayer" begin
        with_mock_backend() do b
          m = material("TestMaterial")
          @test material_name(m) == "TestMaterial"
        end
      end
    end

    @testset "layer(material)" begin
      with_mock_backend() do b
        @testset "PbrMaterial derives layer from name" begin
          l = layer(material_glass)
          @test is_layer(l)
          @test layer_name(l) == "Glass"
        end

        @testset "MaterialInLayer returns its layer" begin
          orig_layer = layer("TestLayer")
          m = material(orig_layer)
          @test layer(m) === orig_layer
        end
      end
    end
  end

  @testset "Predefined materials" begin
    # Pre-defined materials are PbrMaterial instances (not MaterialInLayer).
    # They no longer have a layer field; use material_name instead.
    @testset "material_point" begin
      @test is_pbr_material(material_point)
      @test material_name(material_point) == "Points"
    end

    @testset "material_curve" begin
      @test is_pbr_material(material_curve)
      @test material_name(material_curve) == "Curves"
    end

    @testset "material_surface" begin
      @test is_pbr_material(material_surface)
      @test material_name(material_surface) == "Surfaces"
    end

    @testset "material_basic" begin
      @test is_pbr_material(material_basic)
      @test material_name(material_basic) == "Basic"
    end

    @testset "material_glass" begin
      @test is_pbr_material(material_glass)
      @test material_name(material_glass) == "Glass"
    end

    @testset "material_metal" begin
      @test is_pbr_material(material_metal)
      @test material_name(material_metal) == "Metal"
    end

    @testset "material_wood" begin
      @test is_pbr_material(material_wood)
      @test material_name(material_wood) == "Wood"
    end

    @testset "material_concrete" begin
      @test is_pbr_material(material_concrete)
      @test material_name(material_concrete) == "Concrete"
    end

    @testset "material_plaster" begin
      @test is_pbr_material(material_plaster)
      @test material_name(material_plaster) == "Plaster"
    end

    @testset "material_grass" begin
      @test is_pbr_material(material_grass)
      @test material_name(material_grass) == "Grass"
    end

    @testset "material_clay" begin
      @test is_pbr_material(material_clay)
      @test material_name(material_clay) == "Clay"
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
      @test is_layer_material(merged)
      @test occursin("Material1", material_name(merged))
      @test occursin("Material2", material_name(merged))
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

    @testset "default field" begin
      bp_default = BackendParameter()
      @test bp_default.default === nothing
      @test bp_default(MockBackend) === nothing

      bp_custom = BackendParameter(default=backend_default)
      @test bp_custom.default === backend_default
      @test bp_custom(MockBackend) === backend_default

      # Backend-specific value overrides default
      bp_override = BackendParameter(default=backend_default)
      bp_override(MockBackend, "custom_value")
      @test bp_override(MockBackend) == "custom_value"
    end

    @testset "copy preserves default" begin
      bp = BackendParameter(default=backend_default)
      bp2 = copy(bp)
      @test bp2.default === backend_default
    end
  end

  @testset "three-level material defaults" begin
    with_mock_backend() do b
      @testset "structural materials use backend default" begin
        # Structural materials should return BackendDefault() when no override is set
        @test material_point.data(MockBackend) === backend_default
        @test material_curve.data(MockBackend) === backend_default
        @test material_surface.data(MockBackend) === backend_default
        @test material_basic.data(MockBackend) === backend_default
      end

      @testset "visual materials use PBR path" begin
        # Visual materials should return nothing (PBR path)
        @test material_glass.data(MockBackend) === nothing
        @test material_metal.data(MockBackend) === nothing
      end

      @testset "structural materials produce void_ref" begin
        ref = material_ref(b, material_curve)
        @test ref == 0  # void_ref for MockBackend
      end

      @testset "visual materials produce non-void refs (PBR)" begin
        ref = material_ref(b, material_glass)
        @test ref isa Integer
        @test ref != 0
      end

      @testset "backend override takes precedence over default" begin
        # Create a fresh structural material to avoid mutating the global one
        m = material(
          name="TestStruct",
          base_color=rgba(1.0, 1.0, 0.0, 1.0),
          data=BackendParameter(default=backend_default))
        @test m.data(MockBackend) === backend_default
        set_material(MockBackend, m, "CustomOverride")
        @test m.data(MockBackend) == "CustomOverride"
      end
    end
  end

  @testset "use_layers_for_materials parameter" begin
    @test use_layers_for_materials() == false
    with(use_layers_for_materials, true) do
      @test use_layers_for_materials() == true
    end
    @test use_layers_for_materials() == false
  end

  @testset "material" begin
    with_mock_backend() do b
      @testset "creation with defaults" begin
        sm = material()
        @test sm isa Material
        @test is_pbr_material(sm)
        @test material_name(sm) == "Material"
      end

      @testset "creation with custom base_color" begin
        sm = material(base_color=rgba(1, 0, 0, 1))
        @test pbr_material_base_color(sm) == rgba(1, 0, 0, 1)
      end

      @testset "pbr_material has no layer field" begin
        sm = material()
        @test !hasfield(PbrMaterial, :layer)
      end

      @testset "assignment to shape" begin
        sm = material(base_color=rgba(1, 0, 0, 1))
        s = sphere(u0(), 5, material=sm)
        @test s.material === sm
      end

      @testset "material_ref returns valid reference" begin
        sm = material(base_color=rgba(0, 1, 0, 1))
        ref = material_ref(b, sm)
        @test ref isa Integer
        @test ref != 0  # not void_ref
      end
    end
  end

  @testset "b_set_layer_material default no-op" begin
    with_mock_backend() do b
      # Default implementation returns nothing
      @test b_set_layer_material(b, 0, 0) === nothing
    end
  end

  @testset "set_material with PbrMaterial" begin
    sm = material(name="TestSetOn", base_color=rgba(1, 0, 0, 1))
    set_material(MockBackend, sm, "SomeBackendRef")
    @test sm.data(MockBackend) == "SomeBackendRef"
  end

end
