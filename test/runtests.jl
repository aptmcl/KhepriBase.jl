using KhepriBase
using Test

@testset "KhepriBase.jl" begin
  # Include the mock backend first (used by several test files)
  include("TestMockBackend.jl")

  # Phase 1: Foundation tests
  @testset "Foundation" begin
    include("test_coords.jl")
    include("test_utils.jl")
  end

  # Phase 2: Core Geometry tests
  @testset "Core Geometry" begin
    include("test_paths.jl")
    include("test_geometry.jl")
  end

  # Phase 3: Shape System tests
  @testset "Shape System" begin
    include("test_shapes.jl")
    include("test_backend.jl")
  end

  # Phase 4: Extended Features tests
  @testset "Extended Features" begin
    include("test_materials.jl")
    include("test_bim.jl")
  end

  # Legacy tests (from original Test.jl)
  @testset "Legacy Tests" begin
    include("Test.jl")
  end
end
