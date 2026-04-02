# test_wall_graph.jl - Tests for WallGraph junction-aware wall networks

using Test
using KhepriBase

const TOL = 1e-10

# Helper: extract cx/cy from a Loc for readable comparisons
pt(loc) = (cx(loc), cy(loc))

@testset "WallGraph" begin

  #=== Construction API ===#

  @testset "Construction" begin
    @testset "wall_graph creates empty graph" begin
      wg = wall_graph()
      @test isempty(wg.junctions)
      @test isempty(wg.segments)
    end

    @testset "junction! adds junctions" begin
      wg = wall_graph()
      j1 = junction!(wg, xy(0, 0))
      j2 = junction!(wg, xy(5, 0))
      @test j1 == 1
      @test j2 == 2
      @test length(wg.junctions) == 2
    end

    @testset "segment! connects junctions" begin
      wg = wall_graph()
      j1 = junction!(wg, xy(0, 0))
      j2 = junction!(wg, xy(5, 0))
      s = segment!(wg, j1, j2)
      @test s == 1
      @test length(wg.segments) == 1
      @test s in wg.junctions[j1].segments
      @test s in wg.junctions[j2].segments
    end

    @testset "wall_path! creates junctions and segments" begin
      wg = wall_graph()
      sids = wall_path!(wg, xy(0,0), xy(5,0), xy(5,4))
      @test length(sids) == 2
      @test length(wg.junctions) == 3
      @test length(wg.segments) == 2
    end

    @testset "wall_path! closed creates loop" begin
      wg = wall_graph()
      sids = wall_path!(wg, xy(0,0), xy(5,0), xy(5,4), xy(0,4), closed=true)
      @test length(sids) == 4
      @test length(wg.junctions) == 4
    end

    @testset "wall_path! auto-merges junctions at same position" begin
      wg = wall_graph()
      wall_path!(wg, xy(0,0), xy(5,0), xy(10,0))
      wall_path!(wg, xy(5,0), xy(5,5))
      # xy(5,0) should be shared, not duplicated
      @test length(wg.junctions) == 4
      j_idx = KhepriBase.find_or_create_junction!(wg, xy(5,0), 0.01)
      @test length(wg.junctions[j_idx].segments) == 3  # T-junction
    end

    @testset "segment_length" begin
      wg = wall_graph()
      wall_path!(wg, xy(0,0), xy(3,4))
      @test segment_length(wg, 1) ≈ 5.0 atol=TOL
    end
  end

  #=== Openings ===#

  @testset "Openings" begin
    @testset "add_wall_door! centered by default" begin
      wg = wall_graph()
      sids = wall_path!(wg, xy(0,0), xy(10,0))
      add_wall_door!(wg, sids[1])
      op = wg.segments[sids[1]].openings[1]
      @test op.kind == :door
      @test op.sill ≈ 0.0 atol=TOL
      # Centered: (10 - 1.0) / 2 = 4.5  (default door width = 1.0)
      @test op.distance ≈ 4.5 atol=TOL
    end

    @testset "add_wall_door! at specific position" begin
      wg = wall_graph()
      sids = wall_path!(wg, xy(0,0), xy(10,0))
      add_wall_door!(wg, sids[1], at=2.0)
      @test wg.segments[sids[1]].openings[1].distance ≈ 2.0 atol=TOL
    end

    @testset "add_wall_window! with sill" begin
      wg = wall_graph()
      sids = wall_path!(wg, xy(0,0), xy(10,0))
      add_wall_window!(wg, sids[1], at=3.0, sill=1.2)
      op = wg.segments[sids[1]].openings[1]
      @test op.kind == :window
      @test op.sill ≈ 1.2 atol=TOL
      @test op.distance ≈ 3.0 atol=TOL
    end

    @testset "add_wall_door! by junction pair" begin
      wg = wall_graph()
      j1 = junction!(wg, xy(0, 0))
      j2 = junction!(wg, xy(8, 0))
      segment!(wg, j1, j2)
      add_wall_door!(wg, j1, j2, at=2.0)
      @test length(wg.segments[1].openings) == 1
      @test wg.segments[1].openings[1].distance ≈ 2.0 atol=TOL
    end
  end

  #=== Chain Detection ===#

  @testset "Chain Detection" begin
    @testset "single segment = single chain" begin
      wg = wall_graph()
      wall_path!(wg, xy(0,0), xy(10,0))
      chains = resolve(wg)
      @test length(chains) == 1
      @test length(chains[1].source_segments) == 1
    end

    @testset "L-corner merges into one chain" begin
      wg = wall_graph()
      wall_path!(wg, xy(0,0), xy(5,0), xy(5,4))
      chains = resolve(wg)
      @test length(chains) == 1
      @test length(chains[1].source_segments) == 2
      @test length(path_vertices(chains[1].path)) == 3
    end

    @testset "closed rectangle = one closed chain" begin
      wg = wall_graph()
      wall_path!(wg, xy(0,0), xy(5,0), xy(5,4), xy(0,4), closed=true)
      chains = resolve(wg)
      @test length(chains) == 1
      @test chains[1].path isa KhepriBase.ClosedPolygonalPath
      @test length(path_vertices(chains[1].path)) == 4
    end

    @testset "T-junction: through-pair merges, abutting is separate" begin
      wg = wall_graph()
      wall_path!(wg, xy(0,0), xy(5,0), xy(10,0))  # through wall
      wall_path!(wg, xy(5,0), xy(5,5))             # abutting wall
      chains = resolve(wg)
      @test length(chains) == 2
      # One chain has 2 segments (through), other has 1 (abutting)
      lens = sort([length(c.source_segments) for c in chains])
      @test lens == [1, 2]
    end

    @testset "different families prevent merging" begin
      wg = wall_graph()
      f1 = wall_family(thickness=0.2)
      f2 = wall_family(thickness=0.3)
      j1 = junction!(wg, xy(0, 0))
      j2 = junction!(wg, xy(5, 0))
      j3 = junction!(wg, xy(10, 0))
      segment!(wg, j1, j2, family=f1)
      segment!(wg, j2, j3, family=f2)
      chains = resolve(wg)
      @test length(chains) == 2
    end

    @testset "opening positions adjusted after chain merge" begin
      wg = wall_graph()
      j1 = junction!(wg, xy(0, 0))
      j2 = junction!(wg, xy(5, 0))
      j3 = junction!(wg, xy(5, 4))
      segment!(wg, j1, j2)
      segment!(wg, j2, j3)
      add_wall_door!(wg, 2, at=1.0)  # 1m from j2 on second segment
      chains = resolve(wg)
      @test length(chains) == 1
      # After merge, opening should be at 5 (first seg length) + 1.0 = 6.0
      @test chains[1].openings[1].distance ≈ 6.0 atol=TOL
    end
  end

  #=== Junction Corner Computation ===#

  @testset "Junction Corners" begin
    @testset "valence 1: flat perpendicular cap" begin
      wg = wall_graph()
      wall_path!(wg, xy(0,0), xy(10,0))
      jc = KhepriBase.compute_junction_corners(wg)
      # Junction at xy(0,0): free end, wall goes east
      right, left = jc[1][1]
      @test cy(right) ≈ -0.1 atol=TOL  # right side below centerline
      @test cy(left) ≈ 0.1 atol=TOL    # left side above centerline
      @test cx(right) ≈ 0.0 atol=TOL   # perpendicular cap at x=0
      @test cx(left) ≈ 0.0 atol=TOL
    end

    @testset "valence 2 (L-corner): proper miter" begin
      wg = wall_graph()
      wall_path!(wg, xy(0,0), xy(5,0), xy(5,4))
      jc = KhepriBase.compute_junction_corners(wg)
      # Junction 2 at xy(5,0): elbow where east wall meets north wall
      # At junction, seg1's outgoing direction is west (-1,0), seg2's is north (0,1)
      r1, l1 = jc[2][1]  # seg1 corners (in outgoing=west frame)
      r2, l2 = jc[2][2]  # seg2 corners (in outgoing=north frame)
      # Miter diagonal: (4.9, 0.1) — (5.1, -0.1)
      # seg1: right=(4.9,0.1), left=(5.1,-0.1) (right of west = north, left of west = south)
      @test cx(r1) ≈ 4.9 atol=TOL
      @test cy(r1) ≈ 0.1 atol=TOL
      @test cx(l1) ≈ 5.1 atol=TOL
      @test cy(l1) ≈ -0.1 atol=TOL
      # Corners are shared: seg1's right = seg2's left, seg1's left = seg2's right
      @test cx(r1) ≈ cx(l2) atol=TOL
      @test cy(r1) ≈ cy(l2) atol=TOL
    end

    @testset "valence 3 (T-junction): no overlap" begin
      wg = wall_graph()
      wall_path!(wg, xy(0,0), xy(5,0), xy(10,0))  # through (east)
      wall_path!(wg, xy(5,0), xy(5,5))             # abutting (north)
      jc = KhepriBase.compute_junction_corners(wg)
      j_idx = KhepriBase.find_or_create_junction!(wg, xy(5,0), 0.01)
      corners = jc[j_idx]

      # Through-wall segments (1=west part, 2=east part)
      # Abutting segment (3=north)
      # The through wall's right side (south face) should be at y = -0.1
      # The abutting wall should start at y = 0.1 (through wall's left face)
      # No wall should overlap another

      # Collect all corner points
      all_corners = Loc[]
      for (s, (r, l)) in corners
        push!(all_corners, r, l)
      end
      # With 3 segments at the junction, there are 3 corner points (each shared by 2 segments)
      unique_corners = unique(pt.(all_corners))
      @test length(unique_corners) == 3

      # Verify the abutting wall's right and left edges span the through-wall's thickness
      r3, l3 = corners[3]  # abutting segment
      @test cy(r3) ≈ 0.1 atol=TOL  # starts at through-wall's left face
      @test cy(l3) ≈ 0.1 atol=TOL
      @test cx(r3) ≈ 5.1 atol=TOL  # width = wall thickness = 0.2
      @test cx(l3) ≈ 4.9 atol=TOL
    end

    @testset "valence 3 (T-junction): different thicknesses" begin
      ext = wall_family(thickness=0.3)
      int_w = wall_family(thickness=0.1)
      wg = wall_graph()
      j1 = junction!(wg, xy(0, 0))
      j2 = junction!(wg, xy(5, 0))
      j3 = junction!(wg, xy(10, 0))
      j4 = junction!(wg, xy(5, 5))
      segment!(wg, j1, j2, family=ext)
      segment!(wg, j2, j3, family=ext)
      segment!(wg, j2, j4, family=int_w)
      jc = KhepriBase.compute_junction_corners(wg)
      corners = jc[j2]
      # Through-wall (ext, 0.3m thick): half = 0.15
      # Abutting wall (int, 0.1m thick): half = 0.05
      r_abut, l_abut = corners[3]
      # Abutting wall starts at y = 0.15 (ext wall's left face)
      @test cy(r_abut) ≈ 0.15 atol=TOL
      @test cy(l_abut) ≈ 0.15 atol=TOL
      # Abutting wall's width at junction = 0.1 (its own thickness)
      @test abs(cx(r_abut) - cx(l_abut)) ≈ 0.1 atol=TOL
    end

    @testset "valence 4 (cross junction)" begin
      wg = wall_graph()
      j = junction!(wg, xy(5, 5))
      j_n = junction!(wg, xy(5, 10))
      j_s = junction!(wg, xy(5, 0))
      j_e = junction!(wg, xy(10, 5))
      j_w = junction!(wg, xy(0, 5))
      s_n = segment!(wg, j, j_n)
      s_s = segment!(wg, j, j_s)
      s_e = segment!(wg, j, j_e)
      s_w = segment!(wg, j, j_w)
      jc = KhepriBase.compute_junction_corners(wg)
      corners = jc[j]
      # 4 segments, 4 corner points
      all_corners = Loc[]
      for (s, (r, l)) in corners
        push!(all_corners, r, l)
      end
      unique_corners = unique(pt.(all_corners))
      @test length(unique_corners) == 4
      # Each corner should be at distance ~0.1*sqrt(2) from center (miter of 90° corner)
      for c in unique_corners
        @test sqrt((c[1] - 5)^2 + (c[2] - 5)^2) ≈ 0.1 * sqrt(2) atol=TOL
      end
    end
  end

  #=== 2D Wall Quads (No Overlap) ===#

  @testset "Wall Quads — No Overlap" begin
    @testset "L-corner: adjacent quads share edge" begin
      wg = wall_graph()
      wall_path!(wg, xy(0,0), xy(5,0), xy(5,4))
      jc = KhepriBase.compute_junction_corners(wg)
      ra1, rb1, lb1, la1 = KhepriBase.segment_quad_2d(wg, 1, jc)
      ra2, rb2, lb2, la2 = KhepriBase.segment_quad_2d(wg, 2, jc)
      # At the shared junction, both quads have the same corner points
      # (right of seg1's B-end = right of seg2's A-end, same for left)
      @test cx(rb1) ≈ cx(ra2) atol=TOL
      @test cy(rb1) ≈ cy(ra2) atol=TOL
      @test cx(lb1) ≈ cx(la2) atol=TOL
      @test cy(lb1) ≈ cy(la2) atol=TOL
    end

    @testset "T-junction: abutting wall fits flush against through-wall" begin
      wg = wall_graph()
      wall_path!(wg, xy(0,0), xy(5,0), xy(10,0))
      wall_path!(wg, xy(5,0), xy(5,5))
      jc = KhepriBase.compute_junction_corners(wg)

      # Through-wall left side should have a gap where the abutting wall connects
      ra1, rb1, lb1, la1 = KhepriBase.segment_quad_2d(wg, 1, jc)  # west half
      ra2, rb2, lb2, la2 = KhepriBase.segment_quad_2d(wg, 2, jc)  # east half
      ra3, rb3, lb3, la3 = KhepriBase.segment_quad_2d(wg, 3, jc)  # abutting

      # West half's left side at B-end should stop at abutting wall's left edge
      @test cx(lb1) ≈ cx(la3) atol=TOL
      @test cy(lb1) ≈ cy(la3) atol=TOL
      # East half's left side at A-end should start at abutting wall's right edge
      @test cx(la2) ≈ cx(ra3) atol=TOL
      @test cy(la2) ≈ cy(ra3) atol=TOL
      # Abutting wall's A-end should sit exactly on the through-wall's left face line
      @test cy(ra3) ≈ 0.1 atol=TOL
      @test cy(la3) ≈ 0.1 atol=TOL
    end
  end

  #=== Mesh Generation ===#

  @testset "Mesh Generation" begin
    @testset "simple wall: 6 faces (4 sides + top + bottom + 2 end caps)" begin
      wg = wall_graph()
      wall_path!(wg, xy(0,0), xy(10,0))
      meshes = resolve_geometry(wg)
      @test length(meshes) == 1
      m = meshes[1]
      # 2 long faces + top + bottom + 2 end caps = 6
      @test length(m.quads) == 6
      @test length(m.quad_materials) == 6
    end

    @testset "connected walls: no end caps at shared junctions" begin
      wg = wall_graph()
      wall_path!(wg, xy(0,0), xy(5,0), xy(10,0))
      meshes = resolve_geometry(wg)
      @test length(meshes) == 2
      # Each segment: 2 faces + top + bottom = 4
      # Plus end caps only at free ends: seg1 has cap at xy(0,0), seg2 at xy(10,0)
      @test meshes[1].quads |> length == 5  # 4 + 1 cap at start
      @test meshes[2].quads |> length == 5  # 4 + 1 cap at end
    end

    @testset "T-junction: no end cap on abutting wall at junction" begin
      wg = wall_graph()
      wall_path!(wg, xy(0,0), xy(5,0), xy(10,0))
      wall_path!(wg, xy(5,0), xy(5,5))
      meshes = resolve_geometry(wg)
      @test length(meshes) == 3
      # Abutting wall (seg 3): 2 faces + top + bottom + 1 cap at free end (xy(5,5))
      # No cap at T-junction end (valence 3)
      @test meshes[3].quads |> length == 5
    end

    @testset "wall with door: face subdivision + 4 reveals" begin
      wg = wall_graph()
      sids = wall_path!(wg, xy(0,0), xy(10,0))
      add_wall_door!(wg, sids[1])
      meshes = resolve_geometry(wg)
      m = meshes[1]
      # 3 strips: before, door, after
      # Strip 1 (before): left + right = 2
      # Strip 2 (door, sill=0): no "below" quads, above(2) + 4 reveals = 6
      # Strip 3 (after): left + right = 2
      # Plus: top + bottom + 2 end caps = 4
      # Total: 2 + 6 + 2 + 4 = 14
      @test length(m.quads) == 14
    end

    @testset "wall with window: sill and lintel subdivisions" begin
      wg = wall_graph()
      sids = wall_path!(wg, xy(0,0), xy(10,0))
      add_wall_window!(wg, sids[1], at=3.0, sill=0.9,
        family=window_family(width=1.5, height=1.2))
      meshes = resolve_geometry(wg)
      m = meshes[1]
      # Same structure as door: 3 strips, opening strip has below+above+reveals
      @test length(m.quads) == 16
    end

    @testset "mesh vertices are 3D" begin
      wg = wall_graph(level=level(0), height=3.0)
      wall_path!(wg, xy(0,0), xy(5,0))
      meshes = resolve_geometry(wg)
      m = meshes[1]
      zs = [cz(v) for v in m.vertices]
      @test minimum(zs) ≈ 0.0 atol=TOL
      @test maximum(zs) ≈ 3.0 atol=TOL
    end

    @testset "materials assigned correctly" begin
      f = wall_family(thickness=0.2,
                      left_material=material_plaster,
                      right_material=material_concrete,
                      side_material=material_metal)
      wg = wall_graph()
      wall_path!(wg, xy(0,0), xy(10,0), family=f)
      meshes = resolve_geometry(wg)
      m = meshes[1]
      # First quad should be left face → left_material
      @test m.quad_materials[1] === material_plaster
      # Second quad should be right face → right_material
      @test m.quad_materials[2] === material_concrete
      # Top/bottom/caps → side_material
      @test m.quad_materials[end] === material_metal
    end
  end

  #=== Miter Limit ===#

  @testset "Miter Limit" begin
    @testset "acute angle uses bevel fallback" begin
      wg = wall_graph()
      # Very acute angle (nearly parallel walls meeting at a point)
      wall_path!(wg, xy(0,0), xy(10,0), xy(10.5, 10))
      jc = KhepriBase.compute_junction_corners(wg)
      # Junction at xy(10,0) has a very acute angle
      # Miter point would be very far; should be clamped
      for (s, (r, l)) in jc[2]
        # Neither corner should be farther than MITER_LIMIT * half_thickness from junction
        @test distance(xy(10, 0), r) <= KhepriBase.MITER_LIMIT * 0.1 + TOL
        @test distance(xy(10, 0), l) <= KhepriBase.MITER_LIMIT * 0.1 + TOL
      end
    end
  end

  #=== End-to-End: resolve_geometry ===#

  @testset "End-to-End" begin
    @testset "two-room house produces correct mesh count" begin
      wg = wall_graph(height=2.8)
      wall_path!(wg, xy(0,0), xy(5,0), xy(10,0), xy(10,5),
                     xy(5,5), xy(0,5), closed=true)
      wall_path!(wg, xy(5,0), xy(5,5))
      add_wall_door!(wg, 7)  # door on interior wall
      meshes = resolve_geometry(wg)
      @test length(meshes) == 7  # 6 perimeter + 1 interior
      # All meshes should have vertices and quads
      for m in meshes
        @test !isempty(m.vertices)
        @test !isempty(m.quads)
        @test length(m.quads) == length(m.quad_materials)
      end
    end

    @testset "mixed families at T-junction" begin
      ext = wall_family(thickness=0.3)
      int_w = wall_family(thickness=0.15)
      wg = wall_graph()
      wall_path!(wg, xy(0,0), xy(5,0), xy(10,0), family=ext)
      wall_path!(wg, xy(5,0), xy(5,5), family=int_w)
      meshes = resolve_geometry(wg)
      # ext walls form one chain (2 segs), int wall is separate (1 seg)
      # But resolve_geometry works per-segment, not per-chain
      @test length(meshes) == 3
      # Interior wall mesh should have narrower cross-section
      jc = KhepriBase.compute_junction_corners(wg)
      ra, rb, lb, la = KhepriBase.segment_quad_2d(wg, 3, jc)
      @test abs(cx(ra) - cx(la)) ≈ 0.15 atol=TOL  # interior wall thickness
    end

    @testset "all quad vertex indices are valid" begin
      wg = wall_graph()
      wall_path!(wg, xy(0,0), xy(10,0), xy(10,8), xy(0,8), closed=true)
      wall_path!(wg, xy(5,0), xy(5,8))
      add_wall_door!(wg, 5)
      add_wall_window!(wg, 1, at=2.0)
      meshes = resolve_geometry(wg)
      for m in meshes
        nv = length(m.vertices)
        for (i1, i2, i3, i4) in m.quads
          @test 1 <= i1 <= nv
          @test 1 <= i2 <= nv
          @test 1 <= i3 <= nv
          @test 1 <= i4 <= nv
        end
      end
    end

    @testset "no degenerate quads (all vertices distinct)" begin
      wg = wall_graph()
      wall_path!(wg, xy(0,0), xy(5,0), xy(10,0), xy(10,5),
                     xy(5,5), xy(0,5), closed=true)
      wall_path!(wg, xy(5,0), xy(5,5))
      meshes = resolve_geometry(wg)
      for m in meshes
        for (i1, i2, i3, i4) in m.quads
          vs = [m.vertices[i1], m.vertices[i2], m.vertices[i3], m.vertices[i4]]
          # All 4 vertices should be distinct
          for a in 1:4, b in a+1:4
            @test distance(vs[a], vs[b]) > 1e-6
          end
        end
      end
    end
  end

end
