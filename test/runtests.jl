using KhepriBase
KhepriBase.@import_backend_api  # bring dev API (b_*, realize, NativeRef, ...) into Main for tests
using Test

@testset "KhepriBase.jl" begin
  # Include the mock backend first (used by several test files)
  include("TestMockBackend.jl")

  @testset "API surface" begin
    include("test_api_surface.jl")
    include("test_socket_shadow_guard.jl")
  end

  @testset "Render view options" begin
    include("test_render_view_options.jl")
  end

  @testset "Architectural materials" begin
    include("test_arch_materials.jl")
  end

  # Phase 1: Foundation tests
  @testset "Foundation" begin
    include("test_coords.jl")
    include("test_utils.jl")
  end

  # Phase 2: Core Geometry tests
  @testset "Core Geometry" begin
    include("test_paths.jl")
    include("test_surfaces.jl")
    include("test_geometry.jl")
    include("test_geometry_operations.jl")
    include("test_tolerances.jl")
    include("test_review_fixes.jl")
    include("test_degenerate_inputs.jl")
    include("test_geometry_kernel.jl")
    # Tier 0 correctness oracles (see Docs/TestingStrategy.md)
    include("test_metamorphic.jl")
    include("test_topology.jl")
    include("test_mock_record.jl")
  end

  # Phase 3: Shape System tests
  @testset "Shape System" begin
    include("test_shapes.jl")
    include("test_backend.jl")
    include("test_default_ops.jl")
  end

  # Phase 4: Extended Features tests
  @testset "Extended Features" begin
    include("test_materials.jl")
    include("test_bim.jl")
    include("test_family_lifecycle.jl")
    include("test_wall_graph.jl")
    include("test_constraints.jl")
    include("test_design_layout.jl")
  end

  # Legacy tests (from original Test.jl)
  @testset "Legacy Tests" begin
    include("Test.jl")
  end

  # Phase 5: Fallback chain tests (MinimalTriangleBackend)
  include("test_fallback_chain.jl")

  # Phase 6: Conformance tests (MockBackend as reference)
  @testset "Backend Conformance (MockBackend)" begin
    include("BackendConformanceTests.jl")
    using .BackendConformanceTests

    b = mock_backend()
    run_conformance_tests(b,
      reset! = () -> begin
        reset_mock_backend!(b)
        backend(b)
      end,
      skip = Symbol[]
    )
  end

end
