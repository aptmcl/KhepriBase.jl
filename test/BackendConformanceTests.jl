# BackendConformanceTests.jl - Conformance test suite for Khepri backends
#
# Verifies that a backend correctly implements the KhepriBase contract.
# Any backend can run these tests to validate its implementation.
#
# Usage:
#   include("BackendConformanceTests.jl")
#   using .BackendConformanceTests
#   run_conformance_tests(my_backend,
#     reset! = () -> reset_my_backend!(my_backend),
#     skip = [:advanced])

module BackendConformanceTests

using Test
using KhepriBase
KhepriBase.@import_backend_api

export run_conformance_tests

should_test(tier, skip) = !(tier in skip)

function run_conformance_tests(b::Backend; reset!::Function, skip::Vector{Symbol}=Symbol[])
  @testset "Backend Conformance: $(backend_name(b))" begin
    should_test(:structural, skip) && test_structural(b, reset!)
    should_test(:curves, skip)     && test_curves(b, reset!)
    should_test(:triangles, skip)  && test_triangles(b, reset!)
    should_test(:surfaces, skip)   && test_surfaces(b, reset!)
    should_test(:solids, skip)     && test_solids(b, reset!)
    should_test(:materials, skip)  && test_materials(b, reset!)
    should_test(:layers, skip)     && test_layers(b, reset!)
    should_test(:refs, skip)       && test_refs(b, reset!)
    should_test(:highlevel, skip)  && test_highlevel(b, reset!)
    should_test(:delete, skip)     && test_delete(b, reset!)
    should_test(:advanced, skip)   && test_advanced(b, reset!)
  end
end

# Helper: extract type parameters K,T from a Backend{K,T}
backend_K(::Backend{K,T}) where {K,T} = K
backend_T(::Backend{K,T}) where {K,T} = T

# ================================================================
# :structural -- Backend struct validation
# ================================================================
function test_structural(b, reset!)
  @testset "Structural" begin
    reset!()

    @testset "has refs field" begin
      @test hasfield(typeof(b), :refs)
      @test b.refs isa KhepriBase.References
    end

    @testset "has transaction field" begin
      @test hasfield(typeof(b), :transaction)
    end

    @testset "backend_name returns non-empty String" begin
      name = backend_name(b)
      @test name isa String
      @test !isempty(name)
    end

    @testset "void_ref returns a value" begin
      vr = void_ref(b)
      @test !isnothing(vr)
    end

    @testset "void_ref type matches T parameter" begin
      T = backend_T(b)
      vr = void_ref(b)
      @test vr isa T
    end

    @testset "new_refs returns empty vector" begin
      nr = new_refs(b)
      @test nr isa AbstractVector
      @test isempty(nr)
    end
  end
end

# ================================================================
# :curves -- Tier 0: curves (b_* level)
# ================================================================
function test_curves(b, reset!)
  @testset "Curves (Tier 0)" begin
    reset!()
    vr = void_ref(b)

    @testset "b_point" begin
      r = b_point(b, u0(), vr)
      @test r != vr
    end

    @testset "b_line" begin
      r = b_line(b, [u0(), ux()], vr)
      @test r != vr
    end

    @testset "b_polygon" begin
      r = b_polygon(b, [u0(), ux(), uxy()], vr)
      @test r != vr
    end

    @testset "b_circle" begin
      r = b_circle(b, u0(), 5.0, vr)
      @test !isnothing(r)
    end

    @testset "b_arc" begin
      r = b_arc(b, u0(), 5.0, 0.0, pi/2, vr)
      @test !isnothing(r)
    end

    @testset "b_spline" begin
      r = b_spline(b, [u0(), ux(), uxy()], false, false, vr)
      @test !isnothing(r)
    end

    @testset "b_closed_spline" begin
      r = b_closed_spline(b, [u0(), ux(), uxy(), uy()], vr)
      @test !isnothing(r)
    end

    @testset "b_rectangle" begin
      r = b_rectangle(b, u0(), 10.0, 5.0, vr)
      @test r != vr
    end
  end
end

# ================================================================
# :triangles -- Tier 1: triangles
# ================================================================
function test_triangles(b, reset!)
  @testset "Triangles (Tier 1)" begin
    reset!()
    vr = void_ref(b)

    @testset "b_trig" begin
      r = b_trig(b, u0(), ux(), uy(), vr)
      @test !isnothing(r)
    end

    @testset "b_quad" begin
      r = b_quad(b, u0(), ux(), uxy(), uy(), vr)
      @test !isnothing(r)
    end

    @testset "b_ngon" begin
      pts = [ux(), uxy(), uy()]
      rs = b_ngon(b, pts, u0(), false, vr)
      @test !isnothing(rs)
    end

    @testset "b_quad_strip" begin
      ps = [u0(), ux()]
      qs = [uy(), uxy()]
      rs = b_quad_strip(b, ps, qs, false, vr)
      @test !isnothing(rs)
    end
  end
end

# ================================================================
# :surfaces -- Tier 2: surfaces
# ================================================================
function test_surfaces(b, reset!)
  @testset "Surfaces (Tier 2)" begin
    reset!()
    vr = void_ref(b)

    @testset "b_surface_polygon" begin
      rs = b_surface_polygon(b, [u0(), ux(), uxy(), uy()], vr)
      @test !isnothing(rs)
    end

    @testset "b_surface_circle" begin
      rs = b_surface_circle(b, u0(), 5.0, vr)
      @test !isnothing(rs)
    end

    @testset "b_surface_rectangle" begin
      rs = b_surface_rectangle(b, u0(), 10.0, 5.0, vr)
      @test !isnothing(rs)
    end

    @testset "b_surface_regular_polygon" begin
      rs = b_surface_regular_polygon(b, 6, u0(), 5.0, 0.0, true, vr)
      @test !isnothing(rs)
    end
  end
end

# ================================================================
# :solids -- Tier 3: solids
# ================================================================
function test_solids(b, reset!)
  @testset "Solids (Tier 3)" begin
    reset!()
    vr = void_ref(b)

    @testset "b_box" begin
      r = b_box(b, u0(), 5.0, 5.0, 5.0, vr)
      @test !isnothing(r)
    end

    @testset "b_sphere" begin
      r = b_sphere(b, u0(), 5.0, vr)
      @test !isnothing(r)
    end

    @testset "b_cylinder" begin
      r = b_cylinder(b, u0(), 3.0, 10.0, vr, vr, vr)
      @test !isnothing(r)
    end

    @testset "b_cone" begin
      r = b_cone(b, u0(), 5.0, 10.0, vr, vr)
      @test !isnothing(r)
    end

    @testset "b_torus" begin
      r = b_torus(b, u0(), 10.0, 3.0, vr)
      @test !isnothing(r)
    end

    @testset "b_cuboid" begin
      r = b_cuboid(b,
        u0(), ux(), uxy(), uy(),
        uz(), xyz(1,0,1), xyz(1,1,1), xyz(0,1,1),
        vr)
      @test !isnothing(r)
    end

    @testset "b_prism" begin
      r = b_prism(b, [u0(), ux(), uxy()], vz(5.0), vr, vr, vr)
      @test !isnothing(r)
    end

    @testset "b_regular_pyramid" begin
      r = b_regular_pyramid(b, 4, u0(), 5.0, 0.0, 10.0, true, vr, vr)
      @test !isnothing(r)
    end

    @testset "b_regular_pyramid_frustum" begin
      r = b_regular_pyramid_frustum(b, 4, u0(), 5.0, 0.0, 10.0, 3.0, true, vr, vr, vr)
      @test !isnothing(r)
    end

    @testset "b_regular_prism" begin
      r = b_regular_prism(b, 6, u0(), 5.0, 0.0, 10.0, true, vr, vr, vr)
      @test !isnothing(r)
    end
  end
end

# ================================================================
# :materials -- Material operations
# ================================================================
function test_materials(b, reset!)
  @testset "Materials" begin
    reset!()

    @testset "b_material tier 3" begin
      r = b_material(b, "TestMat",
        rgba(1,0,0,1),    # base_color
        0.0,              # metallic
        0.5,              # roughness
        0.5,              # specular
        1.5,              # ior
        0.0,              # transmission
        0.0,              # transmission_roughness
        0.0,              # clearcoat
        0.0,              # clearcoat_roughness
        rgba(0,0,0,0),    # emission_color
        1.0)              # emission_strength
      @test !isnothing(r)
    end

    @testset "b_get_material with nothing" begin
      r = b_get_material(b, nothing)
      @test r == void_ref(b)
    end
  end
end

# ================================================================
# :layers -- Layer operations
# ================================================================
function test_layers(b, reset!)
  @testset "Layers" begin
    reset!()

    @testset "b_layer" begin
      r = b_layer(b, "TestLayer", true, rgb(1, 0, 0))
      @test !isnothing(r)
    end

    @testset "b_current_layer_ref" begin
      # Some backends (FrontendView) return nothing here; just verify no error
      b_current_layer_ref(b)
      @test true
    end
  end
end

# ================================================================
# :refs -- Reference management (uses high-level API)
# ================================================================
function test_refs(b, reset!)
  @testset "Reference Management" begin
    reset!()
    backend(b)

    @testset "shape ref created after sphere" begin
      s = sphere(u0(), 5)
      refs = shape_refs_storage(b)
      @test haskey(refs, s)
    end

    @testset "ref count matches shape count" begin
      reset!()
      backend(b)
      s1 = sphere(u0(), 1)
      s2 = box(xyz(10,0,0), 2, 2, 2)
      s3 = circle(xyz(20,0,0), 3)
      refs = shape_refs_storage(b)
      @test length(refs) >= 3
    end

    @testset "ref returns GenericRef" begin
      reset!()
      backend(b)
      K = backend_K(b)
      T = backend_T(b)
      s = sphere(u0(), 5)
      r = ref(b, s)
      @test r isa GenericRef{K,T}
    end
  end
end

# ================================================================
# :highlevel -- High-level shape API
# ================================================================
function test_highlevel(b, reset!)
  @testset "High-Level API" begin
    reset!()
    backend(b)

    @testset "point" begin
      s = point(u0())
      @test is_point(s)
    end

    @testset "line" begin
      s = line([u0(), ux()])
      @test !isnothing(s)
    end

    @testset "circle" begin
      s = circle(u0(), 5)
      @test !isnothing(s)
    end

    @testset "polygon" begin
      s = polygon([u0(), ux(), uxy()])
      @test !isnothing(s)
    end

    @testset "rectangle" begin
      s = rectangle(u0(), 10, 5)
      @test !isnothing(s)
    end

    @testset "surface_polygon" begin
      s = surface_polygon([u0(), ux(), uxy()])
      @test !isnothing(s)
    end

    @testset "surface_circle" begin
      s = surface_circle(u0(), 5)
      @test !isnothing(s)
    end

    @testset "sphere" begin
      s = sphere(u0(), 5)
      @test !isnothing(s)
    end

    @testset "box" begin
      s = box(xyz(20,0,0), 5, 5, 5)
      @test !isnothing(s)
    end

    @testset "cylinder" begin
      s = cylinder(xyz(30,0,0), 3, 10)
      @test !isnothing(s)
    end

    @testset "cone" begin
      s = cone(xyz(40,0,0), 5, 10)
      @test !isnothing(s)
    end

    @testset "torus" begin
      s = torus(xyz(50,0,0), 10, 3)
      @test !isnothing(s)
    end

    @testset "shape count in refs" begin
      refs = shape_refs_storage(b)
      # We created 12 shapes above (point through torus)
      @test length(refs) >= 12
    end
  end
end

# ================================================================
# :delete -- Deletion operations
# ================================================================
function test_delete(b, reset!)
  @testset "Deletion" begin
    @testset "delete_all_shapes clears refs" begin
      reset!()
      backend(b)
      sphere(u0(), 5)
      box(xyz(10,0,0), 3, 3, 3)
      @test !isempty(shape_refs_storage(b))
      delete_all_shapes()
      @test isempty(shape_refs_storage(b))
    end

    @testset "delete_shape removes one shape" begin
      reset!()
      backend(b)
      s1 = sphere(u0(), 5)
      s2 = box(xyz(10,0,0), 3, 3, 3)
      @test length(shape_refs_storage(b)) >= 2
      delete_shape(s1)
      refs = shape_refs_storage(b)
      @test !haskey(refs, s1)
      @test haskey(refs, s2)
    end
  end
end

# ================================================================
# :advanced -- Extrusion, sweep, revolve
# ================================================================
function test_advanced(b, reset!)
  @testset "Advanced (Tier 4)" begin
    reset!()
    backend(b)

    @testset "extrusion" begin
      s = extrusion(circular_path(), vz(5))
      @test !isnothing(s)
    end

    @testset "sweep" begin
      s = sweep(
        open_polygonal_path([u0(), xyz(0,0,10)]),
        circular_path(u0(), 1))
      @test !isnothing(s)
    end

    @testset "revolve" begin
      s = revolve(line([xyz(5,0,0), xyz(5,0,10)]))
      @test !isnothing(s)
    end
  end
end

end # module BackendConformanceTests
