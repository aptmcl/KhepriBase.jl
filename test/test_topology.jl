# test_topology.jl — Tier 0 surface tessellation topology & winding correctness.
#
# A correct mesh is more than the right vertex/face COUNT: indices must be in
# range, faces non-degenerate, and (for a planar patch) winding consistent so
# every face normal agrees. A count-only oracle certifies backwards-wound,
# non-manifold, or sliver-laden meshes as "correct". See Docs/TestingStrategy.md.

using KhepriBase
using Test
isdefined(Main, :Oracles) || include("Oracles.jl")
using .Oracles

@testset "Surface tessellation topology (Tier 0)" begin

  # A flat (z = 0) bilinear patch: its tessellation must be planar and
  # consistently wound, with a quad grid whose size the sampler determines.
  planar = bezier_surface([xy(0, 0) xy(3, 0); xy(0, 2) xy(3, 2)])

  @testset "planar patch: indices, non-degenerate, consistent normals, count" begin
    for (u, v) in ((2, 2), (2, 3), (5, 4))
      m = KhepriBase.tessellate_surface(planar, u_count=u, v_count=v)
      @test face_indices_valid(m)
      @test !has_repeated_face_indices(m)
      @test !has_degenerate_faces(m)
      @test planar_normals_consistent(m)
      # Face count is the (nu-1)·(nv-1) quad grid derived from the sampler,
      # not a hard-coded magic number (which would rot with the default density).
      (nu, nv) = size(KhepriBase.sample_surface_points(planar, u, v))
      @test length(m.faces) == (nu - 1) * (nv - 1)
    end
  end

  @testset "curved patch: still valid, non-degenerate connectivity" begin
    curved = bezier_surface([xyz(0, 0, 0) xyz(1, 0, 1) xyz(2, 0, 0);
                             xyz(0, 1, 1) xyz(1, 1, 2) xyz(2, 1, 1);
                             xyz(0, 2, 0) xyz(1, 2, 1) xyz(2, 2, 0)])
    m = KhepriBase.tessellate_surface(curved, u_count=6, v_count=6)
    @test face_indices_valid(m)
    @test !has_repeated_face_indices(m)
    @test !has_degenerate_faces(m)
  end
end
