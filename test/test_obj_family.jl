# test_obj_family.jl — OBJ-family realization (Milestone 2, Phase 4).
#
# A single portable family realizes as an OBJ mesh on ANY backend via the default b_mesh_obj_fmt (no
# native OBJ loader required), and the furniture/fixture realizers now resolve OBJ mappings through
# maybe_backend_family — so a family whose OBJ is reachable only via based_on/default no longer falls
# back to a placeholder box. This is the enabling change for cross-backend BIM-element families
# (a family carrying both a Revit .rfa and an extracted OBJ/MTL).

using Test
using KhepriBase
if !@isdefined(MockBackend)
  include("TestMockBackend.jl")
end

@testset "OBJ family realization (Phase 4)" begin
  dir = mktempdir()
  objpath = joinpath(dir, "tri.obj")
  write(objpath, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")   # one triangle

  @testset "read_obj_mesh parses vertices/faces" begin
    (verts, faces) = KhepriBase.read_obj_mesh(objpath)
    @test length(verts) == 3
    @test faces == [[1, 2, 3]]
  end

  @testset "obj_file_path passes through .obj / absolute paths" begin
    @test KhepriBase.obj_file_path(objpath) == objpath                    # absolute .obj → verbatim
    @test KhepriBase.obj_file_path("Porta/Porta") ==
          joinpath("resources", "models", "obj", "Porta/Porta.obj")       # bare name → resource dir
  end

  @testset "default b_mesh_obj_fmt emits a surface mesh" begin
    mb = MockBackend()
    KhepriBase.b_mesh_obj_fmt(mb, objpath, u0())
    @test length(mb.triangles) == 1        # a mesh…
    @test length(mb.boxes) == 0            # …not a box
  end

  @testset "b_sink resolves a delegated OBJ family (was: 80cm box)" begin
    base = sink_family()
    set_backend_family(base, MockBackend, obj_family(objpath))
    derived = sink_family(based_on = base)   # OBJ reachable only via based_on, not directly
    @test get(derived.implemented_as, MockBackend, nothing) === nothing            # old raw get: miss
    @test KhepriBase.maybe_backend_family(MockBackend(), derived) isa KhepriBase.OBJFamily
    mb = MockBackend()
    KhepriBase.b_sink(mb, u0(), nothing, derived)
    @test length(mb.triangles) == 1 && length(mb.boxes) == 0
  end

  @testset "b_family_element resolves OBJ family (was: always a 1m box)" begin
    fam = family_element_family()
    set_backend_family(fam, MockBackend, obj_family(objpath))
    mb = MockBackend()
    KhepriBase.b_family_element(mb, u0(), 0, default_level(), fam)
    @test length(mb.triangles) == 1 && length(mb.boxes) == 0
  end

  @testset "obj_model places a scaled mesh at its frame" begin
    mb = MockBackend()
    KhepriBase.b_obj_model(mb, objpath, xyz(10, 0, 0), 2.0, nothing)
    @test length(mb.triangles) == 1 && length(mb.boxes) == 0
    t = mb.triangles[1]
    @test t.p1.x ≈ 10.0 atol = 1e-9      # near vertex (0,0,0) at the frame origin (10,0,0)
    @test t.p2.x ≈ 12.0 atol = 1e-9      # far vertex (1,0,0)·scale2 → (12,0,0)
    # The proxy constructs with fields intact (realization suppressed under introspection):
    with_introspection(mb) do
      om = obj_model(path = objpath, location = xyz(1, 2, 3), scale = 1.5)
      @test is_obj_model(om) && om.path == objpath && om.scale == 1.5
    end
  end
end
