# test_family_lifecycle.jl - Tests for family creation → mapping → realization → caching lifecycle

using Test
using KhepriBase

include("TestMockBackend.jl")

@testset "Family Lifecycle" begin

  @testset "family_ref caching" begin
    with_mock_backend() do b
      sf = slab_family()
      lf = layer_family("TestSlabLayer")
      set_backend_family(sf, b, lf)

      r1 = family_ref(b, sf)
      r2 = family_ref(b, sf)
      @test r1 === r2  # same cached value
    end
  end

  @testset "set_backend_family clears cache" begin
    with_mock_backend() do b
      sf = slab_family()
      lf1 = layer_family("Layer1")
      lf2 = layer_family("Layer2")

      set_backend_family(sf, b, lf1)
      r1 = family_ref(b, sf)

      set_backend_family(sf, b, lf2)
      r2 = family_ref(b, sf)

      @test r1 !== r2  # cache was cleared, re-realized
    end
  end

  @testset "resolution chain: family_element delegates to parent's implemented_as" begin
    with_mock_backend() do b
      sf = slab_family()
      lf = layer_family("BaseSlabLayer")
      set_backend_family(sf, b, lf)

      sfe = slab_family_element(sf, thickness=0.4)
      # sfe.based_on === sf, and sf has the implemented_as entry
      bf = backend_family(b, sfe)
      @test bf === lf
    end
  end

  @testset "error when no implemented_as and no based_on" begin
    with_mock_backend() do b
      sf = slab_family()  # no implemented_as set for this backend
      @test_throws ErrorException backend_family(b, sf)
    end
  end

  @testset "LayerFamily through family_ref calls b_layer" begin
    with_mock_backend() do b
      sf = slab_family()
      lf = layer_family("MockLayer", rgb(1, 0, 0))
      set_backend_family(sf, b, lf)

      r = family_ref(b, sf)
      # MockBackend's b_layer returns next_ref! which is an Int
      @test r isa Integer
    end
  end

  @testset "invalidate_family_refs clears default family refs" begin
    with_mock_backend() do b
      # Use the global default family, which is registered in _family_defaults
      sf = default_slab_family()
      lf = layer_family("InvalidationTest")
      set_backend_family(sf, b, lf)

      family_ref(b, sf)  # populate cache
      @test haskey(sf.ref, b)

      invalidate_family_refs(b)
      @test !haskey(sf.ref, b)
    end
  end

  @testset "force_realize for Family delegates to family_ref" begin
    with_mock_backend() do b
      sf = slab_family()
      lf = layer_family("ForceRealizeTest")
      set_backend_family(sf, b, lf)

      force_realize(b, sf)

      # Both caches should be populated
      @test haskey(sf.ref, b)  # family_ref cache
      @test haskey(b.refs.families, sf)  # proxy system cache
    end
  end

  @testset "b_delete_all_shape_refs preserves family refs" begin
    with_mock_backend() do b
      sf = slab_family()
      lf = layer_family("PreserveTest")
      set_backend_family(sf, b, lf)

      family_ref(b, sf)  # populate f.ref cache
      @test haskey(sf.ref, b)

      # Simulate delete_all by clearing shapes ref storage
      empty!(b.refs.shapes)

      # Family ref should survive
      @test haskey(sf.ref, b)
    end
  end

  @testset "implemented_as is type-keyed" begin
    with_mock_backend() do b
      sf = slab_family()
      lf = layer_family("TypeKeyTest")
      set_backend_family(sf, b, lf)

      # Key should be the type, not the instance
      @test haskey(sf.implemented_as, typeof(b))
      @test sf.implemented_as[typeof(b)] === lf
    end
  end

  @testset "meta_program round-trip" begin
    @testset "simple family" begin
      sf = slab_family(thickness=0.3, coating_thickness=0.05)
      expr = meta_program(sf)
      sf2 = eval(expr)
      @test sf2.thickness ≈ sf.thickness atol=1e-10
      @test sf2.coating_thickness ≈ sf.coating_thickness atol=1e-10
    end

    @testset "composite family (table_chair_family)" begin
      tf = table_family(length=2.0, width=1.0)
      cf = chair_family(length=0.5)
      tcf = table_chair_family(table_family=tf, chair_family=cf, chairs_top=2, chairs_bottom=2)
      expr = meta_program(tcf)
      tcf2 = eval(expr)
      @test tcf2.table_family.length ≈ 2.0 atol=1e-10
      @test tcf2.chair_family.length ≈ 0.5 atol=1e-10
      @test tcf2.chairs_top == 2
      @test tcf2.chairs_bottom == 2
    end

    @testset "default family parameter" begin
      expr = meta_program(default_slab_family)
      @test expr isa Expr
    end
  end

  @testset "family_element_element inherits implemented_as" begin
    with_mock_backend() do b
      sf = slab_family()
      lf = layer_family("InheritTest")
      set_backend_family(sf, b, lf)

      sfe = slab_family_element(sf, thickness=0.5)
      # family_element copies parent's implemented_as
      @test haskey(sfe.implemented_as, typeof(b))
    end
  end

end
