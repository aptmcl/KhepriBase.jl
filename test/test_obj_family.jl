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

  @testset "read_obj_mesh resolves negative indices + pads short v lines (T21)" begin
    # A short "v 0 1" line was silently skipped, shifting every later 1-based face index; and OBJ's
    # negative (relative) indices were parsed as literal negatives. Harden both.
    p2 = joinpath(dir, "hardening.obj")
    write(p2, "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1\nf 1 2 3\nf -1 -2 -3\n")
    local verts, faces
    @test_logs (:warn,) match_mode=:any ((verts, faces) = KhepriBase.read_obj_mesh(p2))
    @test length(verts) == 4                 # short "v 0 1" padded, not dropped
    @test verts[4] == [0.0, 1.0, 0.0]        # missing z → 0.0
    @test faces[1] == [1, 2, 3]
    @test faces[2] == [4, 3, 2]              # -1,-2,-3 resolved against the 4 vertices
  end

  @testset "read_obj_mesh tracks per-face usemtl group (T21)" begin
    p3 = joinpath(dir, "twomat.obj")
    write(p3, "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nusemtl red\nf 1 2 3\nusemtl blue\nf 2 4 3\n")
    (verts, faces, face_mats) = KhepriBase.read_obj_mesh(p3)
    @test length(faces) == 2
    @test face_mats == ["red", "blue"]       # active usemtl captured per face
  end

  @testset "b_mesh_obj_fmt groups a 2-material OBJ (0-based faces, no loss)" begin
    p3 = joinpath(dir, "twomat.obj")          # (written above)
    write(joinpath(dir, "twomat.mtl"),
          "newmtl red\nKd 1 0 0\nnewmtl blue\nKd 0 0 1\n")
    mb = MockBackend()
    KhepriBase.b_mesh_obj_fmt(mb, p3, u0())
    @test length(mb.triangles) == 2          # one per face, split across two material groups, none lost
    @test length(mb.boxes) == 0
    # 0-based decrement correct: face "f 2 4 3" (1-based) → verts[2,4,3] = (1,0,0),(1,1,0),(0,1,0).
    let t = mb.triangles[2]
      @test (t.p1.x, t.p1.y) == (1.0, 0.0)
      @test (t.p2.x, t.p2.y) == (1.0, 1.0)
      @test (t.p3.x, t.p3.y) == (0.0, 1.0)
    end
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

@testset "parse_mtl PBR keywords" begin
  mtl = tempname() * ".mtl"
  write(mtl, """
  newmtl Painted_Metal
  Kd 0.8000 0.2000 0.1000
  d 0.9000
  Pr 0.3500
  Pm 0.2500
  Ke 0.1000 0.0000 0.0000
  Ns 400.0
  illum 2

  newmtl Legacy_Phong
  Kd 0.1 0.6 0.2
  Ns 198.0
  """)
  (mats, order) = KhepriBase.parse_mtl(mtl)
  @test order == ["Painted_Metal", "Legacy_Phong"]
  let m = mats["Painted_Metal"]
    @test m.roughness == 0.35          # Pr wins over Ns
    @test m.metallic == 0.25
    @test red(m.emission_color) ≈ 0.1
    @test alpha(m.base_color) ≈ 0.9
  end
  # Without Pr, roughness derives from the Blinn-Phong exponent: sqrt(2/(Ns+2)).
  @test mats["Legacy_Phong"].roughness ≈ sqrt(2 / 200)
  rm(mtl; force=true)
end
