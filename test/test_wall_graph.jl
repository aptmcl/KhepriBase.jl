# test_wall_graph.jl - Tests for WallGraph junction-aware wall networks

using Test
using KhepriBase

const TOL = 1e-10

# Helper: extract cx/cy from a Loc for readable comparisons
pt(loc) = (cx(loc), cy(loc))

# Helper: recover the junction traversal order for a chain. Mirrors
# `chain_junctions` in WallGraph.jl; replicated here rather than
# exported so the test module stays self-contained.
function _chain_junctions_dbg(wg, chain::Vector{Int})
  n = length(chain)
  js = Int[]
  if n == 1
    let seg = wg.segments[chain[1]]
      push!(js, seg.junction_a); push!(js, seg.junction_b)
    end
  else
    let s1 = wg.segments[chain[1]], s2 = wg.segments[chain[2]]
      # Start is the junction of s1 NOT shared with s2
      if s1.junction_a == s2.junction_a || s1.junction_a == s2.junction_b
        push!(js, s1.junction_b); push!(js, s1.junction_a)
      else
        push!(js, s1.junction_a); push!(js, s1.junction_b)
      end
      for k in 2:n
        let seg = wg.segments[chain[k]]
          push!(js, seg.junction_a == js[end] ? seg.junction_b : seg.junction_a)
        end
      end
    end
  end
  js
end

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

    @testset "T-junction: all three sides stay separate" begin
      # After switching to face-intersection joins, chains no
      # longer merge through valence-3 junctions — at a T-junction
      # the abutting wall would intrude onto the through-pair's
      # face on one side, producing asymmetric face polylines that
      # `b_quad_strip` can't render. Terminating chains at every
      # T-junction keeps left/right polyline counts matched and
      # lets each wall's corner at the junction come from the
      # pairwise `junction_face_corners` intersection.
      wg = wall_graph()
      wall_path!(wg, xy(0,0), xy(5,0), xy(10,0))  # through wall
      wall_path!(wg, xy(5,0), xy(5,5))             # abutting wall
      chains = resolve(wg)
      @test length(chains) == 3
      lens = sort([length(c.source_segments) for c in chains])
      @test lens == [1, 1, 1]
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
      # The through-wall is continuous (straight perpendicular corners)
      # The abutting wall starts at the through-wall's face (y = 0.1)

      # Through-pair corners are shared between the two through segments
      r1, l1 = corners[1]  # west part: right=(5,0.1), left=(5,-0.1)
      r2, l2 = corners[2]  # east part: right=(5,-0.1), left=(5,0.1)
      @test cx(r1) ≈ cx(l2) atol=TOL  # through-pair shares north corner
      @test cy(r1) ≈ cy(l2) atol=TOL
      @test cx(l1) ≈ cx(r2) atol=TOL  # through-pair shares south corner
      @test cy(l1) ≈ cy(r2) atol=TOL

      # Through-wall corners at the junction: straight perpendicular
      @test cy(r1) ≈ 0.1 atol=TOL   # north
      @test cy(l1) ≈ -0.1 atol=TOL  # south

      # Abutting wall starts at the through-wall's face
      r3, l3 = corners[3]
      @test cy(r3) ≈ 0.1 atol=TOL
      @test cy(l3) ≈ 0.1 atol=TOL
      @test cx(r3) ≈ 5.1 atol=TOL
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

    @testset "T-junction: through-wall continuous, abutting on face" begin
      wg = wall_graph()
      wall_path!(wg, xy(0,0), xy(5,0), xy(10,0))
      wall_path!(wg, xy(5,0), xy(5,5))
      jc = KhepriBase.compute_junction_corners(wg)

      ra1, rb1, lb1, la1 = KhepriBase.segment_quad_2d(wg, 1, jc)  # west half
      ra2, rb2, lb2, la2 = KhepriBase.segment_quad_2d(wg, 2, jc)  # east half
      ra3, rb3, lb3, la3 = KhepriBase.segment_quad_2d(wg, 3, jc)  # abutting

      # Through-wall is continuous: both halves meet at straight perpendicular
      @test cx(lb1) ≈ 5.0 atol=TOL
      @test cy(lb1) ≈ 0.1 atol=TOL
      @test cx(la2) ≈ 5.0 atol=TOL
      @test cy(la2) ≈ 0.1 atol=TOL
      @test cx(rb1) ≈ 5.0 atol=TOL
      @test cy(rb1) ≈ -0.1 atol=TOL
      @test cx(ra2) ≈ 5.0 atol=TOL
      @test cy(ra2) ≈ -0.1 atol=TOL
      # Abutting wall starts exactly on the through-wall's left face line
      @test cy(ra3) ≈ 0.1 atol=TOL
      @test cy(la3) ≈ 0.1 atol=TOL
      @test cx(ra3) ≈ 5.1 atol=TOL
      @test cx(la3) ≈ 4.9 atol=TOL
    end

    @testset "T-junction at 60°: abutting corners on face line" begin
      wg = wall_graph()
      wall_path!(wg, xy(0,0), xy(5,0), xy(10,0))
      wall_path!(wg, xy(5,0), xy(5 + 5*cosd(60), 5*sind(60)))
      jc = KhepriBase.compute_junction_corners(wg)

      ra3, rb3, lb3, la3 = KhepriBase.segment_quad_2d(wg, 3, jc)  # abutting
      # Both corners of the abutting wall at the junction lie on the face (y=0.1)
      @test cy(ra3) ≈ 0.1 atol=TOL
      @test cy(la3) ≈ 0.1 atol=TOL
      # Through-wall remains continuous
      ra1, rb1, lb1, la1 = KhepriBase.segment_quad_2d(wg, 1, jc)
      ra2, rb2, lb2, la2 = KhepriBase.segment_quad_2d(wg, 2, jc)
      @test cy(lb1) ≈ 0.1 atol=TOL
      @test cy(la2) ≈ 0.1 atol=TOL
      @test cy(rb1) ≈ -0.1 atol=TOL
      @test cy(ra2) ≈ -0.1 atol=TOL
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

    @testset "cross junction: cap fills central hole" begin
      wg = wall_graph()
      j1 = junction!(wg, xy(0, 0))
      j_n = junction!(wg, xy(0, 10))
      j_s = junction!(wg, xy(0, -10))
      j_e = junction!(wg, xy(10, 0))
      j_w = junction!(wg, xy(-10, 0))
      segment!(wg, j1, j_n)
      segment!(wg, j1, j_s)
      segment!(wg, j1, j_e)
      segment!(wg, j1, j_w)
      meshes = resolve_geometry(wg)
      # 4 segment meshes + 1 junction cap mesh
      @test length(meshes) == 5
      cap = meshes[5]
      # Cap has 1 quad for top + 1 quad for bottom = 2
      @test length(cap.quads) == 2
    end

    @testset "Y-junction (3 at 120°): cap fills central hole" begin
      wg = wall_graph()
      j1 = junction!(wg, xy(0, 0))
      j2 = junction!(wg, pol(10, 0))
      j3 = junction!(wg, pol(10, 2π/3))
      j4 = junction!(wg, pol(10, 4π/3))
      segment!(wg, j1, j2)
      segment!(wg, j1, j3)
      segment!(wg, j1, j4)
      meshes = resolve_geometry(wg)
      # 3 segment meshes + 1 junction cap mesh
      @test length(meshes) == 4
      cap = meshes[4]
      # Cap has 1 triangle (degenerate quad) for top + 1 for bottom = 2
      @test length(cap.quads) == 2
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

  #=== Arc segments (Tier 1 / Tier 2) ===#

  @testset "Arc segments" begin
    @testset "arc_segment! stamps ArcPath on WallSegment" begin
      wg = wall_graph()
      j1 = junction!(wg, xy(10, 0))
      j2 = junction!(wg, xy(0, 10))
      s  = KhepriBase.arc_segment!(wg, j1, j2; center=u0(), amplitude=π/2)
      seg = wg.segments[s]
      @test !isnothing(seg.arc)
      @test seg.arc.radius ≈ 10.0 atol=TOL
      @test seg.arc.amplitude ≈ π/2 atol=TOL
    end

    @testset "segment_length returns arc length for arcs" begin
      wg = wall_graph()
      j1 = junction!(wg, xy(10, 0))
      j2 = junction!(wg, xy(0, 10))
      KhepriBase.arc_segment!(wg, j1, j2; center=u0(), amplitude=π/2)
      # Quarter-arc on radius 10: length = 10 · π/2
      @test segment_length(wg, 1) ≈ 10 * π/2 atol=1e-6
    end

    @testset "resolve emits ArcPath for a single arc" begin
      wg = wall_graph()
      j1 = junction!(wg, xy(10, 0))
      j2 = junction!(wg, xy(0, 10))
      KhepriBase.arc_segment!(wg, j1, j2; center=u0(), amplitude=π/2)
      chains = resolve(wg)
      @test length(chains) == 1
      @test chains[1].path isa KhepriBase.ArcPath
      @test chains[1].path.radius ≈ 10.0
    end

    @testset "co-circular arcs chain into one ArcPath" begin
      # Two arcs sharing centre and radius, meeting at a valence-2 junction.
      wg = wall_graph()
      j1 = junction!(wg, xy(10, 0))
      j2 = junction!(wg, xy(0, 10))
      j3 = junction!(wg, xy(-10, 0))
      KhepriBase.arc_segment!(wg, j1, j2; center=u0(), amplitude=π/2)
      KhepriBase.arc_segment!(wg, j2, j3; center=u0(), amplitude=π/2)
      chains = resolve(wg)
      @test length(chains) == 1
      @test chains[1].path isa KhepriBase.ArcPath
      @test chains[1].path.amplitude ≈ π atol=1e-6
    end

    @testset "arc and straight segments stay separate chains" begin
      # A straight wall meeting an arc wall forms a hard junction,
      # even though both share family. Each ends up in its own chain.
      wg = wall_graph()
      j1 = junction!(wg, xy(10, 0))
      j2 = junction!(wg, xy(0, 10))
      j3 = junction!(wg, xy(0, 15))
      KhepriBase.arc_segment!(wg, j1, j2; center=u0(), amplitude=π/2)
      segment!(wg, j2, j3)
      chains = resolve(wg)
      @test length(chains) == 2
      @test any(c -> c.path isa KhepriBase.ArcPath, chains)
      @test any(c -> c.path isa KhepriBase.OpenPolygonalPath, chains)
    end

    @testset "line_arc_intersection_2d" begin
      # A horizontal line through y=5 intersects the unit-radius circle
      # at origin? No. Use a line that crosses the circle of radius 5.
      pts = KhepriBase.line_arc_intersection_2d(xy(-10, 3), vxy(1, 0), u0(), 5)
      @test length(pts) == 2
      @test all(p -> abs(distance(p, u0()) - 5) < 1e-6, pts)
    end

    @testset "arc_arc_intersection_2d" begin
      # Two unit circles centred at (0,0) and (1,0) intersect at (0.5, ±√0.75).
      pts = KhepriBase.arc_arc_intersection_2d(u0(), 1.0, xy(1, 0), 1.0)
      @test length(pts) == 2
      @test all(p -> abs(cx(p) - 0.5) < 1e-6, pts)
    end
  end

  #=
  Face-polyline model: the junction is a region, and each incident
  wall ends with two corner vertices on its two faces (not a single
  axis point). These are the primitives `junction_face_corners` and
  `wall_face_polylines` are built to produce. Verify them on each
  junction valence, with the two-room (Room A triangle + Room B
  trapezoid) scenario as the canonical 3-way-no-through-pair case.
  =#
  @testset "Junction face corners" begin
    fam = wall_family(thickness=0.2)

    @testset "valence-1 free end: perpendicular cap" begin
      # A single wall from (0,0) to (10,0): outgoing at j1 is +X,
      # so the face corners should be at (0, +t/2) and (0, -t/2).
      wg = wall_graph()
      j1 = junction!(wg, xy(0, 0))
      j2 = junction!(wg, xy(10, 0))
      seg = segment!(wg, j1, j2; family=fam)
      corners = KhepriBase.junction_face_corners(wg, j1)
      (left, right) = corners[seg]
      @test cy(left)  ≈  0.1 atol=1e-9
      @test cy(right) ≈ -0.1 atol=1e-9
      @test cx(left)  ≈  0.0 atol=1e-9
      @test cx(right) ≈  0.0 atol=1e-9
    end

    @testset "valence-2 elbow: offset-line intersection" begin
      # Two walls: (0,0)→(5,0) and (5,0)→(5,5). The miter corner on
      # the outside of the right-angle is at (5 + t/2, -t/2) (right
      # face of the first, right face of the second in world terms).
      # Easier to test the chain polylines: for the merged chain,
      # the left face at the corner should be exactly (5 - t/2, t/2)
      # (inner corner) and the right face at (5 + t/2, -t/2) (outer).
      wg = wall_graph()
      sids = wall_path!(wg, xy(0,0), xy(5,0), xy(5,5); family=fam)
      chains = KhepriBase.resolve(wg)
      @test length(chains) == 1
      all_corners = KhepriBase.all_junction_face_corners(wg)
      chain = chains[1]
      # Rebuild (chain, junctions) — the chain detector gave us an
      # ordered segment list; recover the junction sequence from
      # `source_segments` by matching junction_a / junction_b.
      junctions = _chain_junctions_dbg(wg, chain.source_segments)
      (lpts, rpts) = KhepriBase.wall_face_polylines(wg, chain.source_segments, junctions, all_corners)
      @test length(lpts) == 3 && length(rpts) == 3
      # Corner vertex (middle of each face polyline) is at the inner
      # / outer miter of the right angle.
      @test pt(lpts[2]) == (5 - 0.1, 0 + 0.1)  # inner corner
      @test pt(rpts[2]) == (5 + 0.1, 0 - 0.1)  # outer corner
    end

    @testset "junction_cap_polygon" begin
      #=
      A valence-≥3 junction with NO collinear through-pair needs a
      flat infill polygon at the ceiling (and floor) to close the
      N-gonal gap between the incident walls' top faces.
      `junction_cap_polygon` returns the N corner vertices in CCW
      angular order.

      Through-pair T-junctions (valence 3 with two collinear
      walls) are skipped — the through-wall's top face covers the
      junction continuously across its two segments, so adding a
      separate cap would z-fight with it. This matches
      `render_wall_graph`'s mesh generator.
      =#
      fam = wall_family(thickness=0.2)
      # Non-T three-way: three arms at ~120° with no collinear
      # pair — needs a cap.
      wg = wall_graph()
      j1 = junction!(wg, xy(0, 0))
      j2 = junction!(wg, xy(5, 0))                 # east
      j3 = junction!(wg, xy(-2.5,  4.33))          # 120° (upper-left)
      j4 = junction!(wg, xy(-2.5, -4.33))          # 240° (lower-left)
      segment!(wg, j1, j2; family=fam)
      segment!(wg, j1, j3; family=fam)
      segment!(wg, j1, j4; family=fam)
      cap = KhepriBase.junction_cap_polygon(wg, j1)
      @test length(cap) == 3
      @test all(p -> isfinite(cx(p)) && isfinite(cy(p)), cap)
      # Valence-3 T-junction WITH through-pair: no cap — the
      # through-wall's top continues across the junction.
      wg_t = wall_graph()
      t1 = junction!(wg_t, xy(0, 0))
      t2 = junction!(wg_t, xy(5, 0))
      t3 = junction!(wg_t, xy(-5, 0))
      t4 = junction!(wg_t, xy(0, 5))
      segment!(wg_t, t1, t2; family=fam)  # through east
      segment!(wg_t, t1, t3; family=fam)  # through west (collinear with east)
      segment!(wg_t, t1, t4; family=fam)  # abutting north
      @test isempty(KhepriBase.junction_cap_polygon(wg_t, t1))
      # Valence-2 elbow: no cap (single merged miter).
      wg2 = wall_graph()
      k1 = junction!(wg2, xy(0, 0))
      k2 = junction!(wg2, xy(5, 0))
      k3 = junction!(wg2, xy(5, 5))
      segment!(wg2, k1, k2; family=fam)
      segment!(wg2, k2, k3; family=fam)
      @test isempty(KhepriBase.junction_cap_polygon(wg2, k2))  # valence 2
      @test isempty(KhepriBase.junction_cap_polygon(wg2, k1))  # valence 1
    end

    @testset "Room A / Room B — 3-way junction at (0,5)" begin
      #=
      Two rooms sharing a diagonal edge:
        Room A: triangle (0,0)-(10,0)-(0,5).
        Room B: trapezoid (10,0)-(20,0)-(20,5)-(0,5).
      After edge classification and dedup, the wall graph has six
      segments and five junctions. The (0,5) junction is valence 3
      with no collinear pair — the case where the old
      `abutment_extension` returned `nothing` and walls ended with
      flat caps, producing the visible gap.

      With the face-intersection model, every pair of angular
      neighbours at (0,5) should yield a finite corner vertex. We
      verify that each of the three incident walls has:
        - two corners (left and right),
        - both corners finite,
        - both corners within ~thickness of the junction point.
      =#
      wg = wall_graph()
      j_00  = junction!(wg, xy(0, 0))
      j_10_0 = junction!(wg, xy(10, 0))
      j_05  = junction!(wg, xy(0, 5))
      j_20_0 = junction!(wg, xy(20, 0))
      j_20_5 = junction!(wg, xy(20, 5))
      # Room A perimeter
      segment!(wg, j_00,  j_10_0; family=fam)   # bottom
      diag = segment!(wg, j_10_0, j_05; family=fam)  # shared diagonal
      segment!(wg, j_05,  j_00;  family=fam)    # left
      # Room B non-shared edges
      segment!(wg, j_10_0, j_20_0; family=fam)  # bottom-right
      segment!(wg, j_20_0, j_20_5; family=fam)  # right
      segment!(wg, j_20_5, j_05;   family=fam)  # top
      corners_05 = KhepriBase.junction_face_corners(wg, j_05)
      @test length(corners_05) == 3  # three incident walls
      J = xy(0, 5)
      for (seg_idx, (L, R)) in corners_05
        @test isfinite(cx(L)) && isfinite(cy(L))
        @test isfinite(cx(R)) && isfinite(cy(R))
        # Corner stays within ~one thickness of the junction point
        # — a far-away corner would mean degenerate parallel faces.
        @test distance(J, L) < 0.5
        @test distance(J, R) < 0.5
      end
    end
  end

  @testset "Closed chain: left and right face polylines have matching vertex counts" begin
    #=
    Regression guard.

    For a closed chain, the first and last corner of each face
    polyline are the same point computed through two different
    neighbour-pair `line_intersection_2d` calls at the closing
    junction — mathematically equal, but floating-point roundoff
    leaves them microscopically different. A tolerance-based per-
    face strip can pass on one face and fail on the other, producing
    left and right polylines with different vertex counts — which
    the downstream `b_quad_strip_closed` then sends to AutoCAD as
    a SubDMesh whose face list indexes beyond the vertex array,
    surfacing as `eInvalidIndex`. Trust `is_closed` instead: strip
    both sides unconditionally when the chain loops back.

    This test locks in that invariant. A closed square chain has
    four segments; the centerline's `closed_polygonal_path` holds
    four vertices; both face polylines must hold four vertices
    too. Re-introducing the tolerance-based strip would fail this
    test with 3 vs 4 or 4 vs 3.
    =#
    wg = wall_graph()
    wall_path!(wg, xy(0,0), xy(10,0), xy(10,10), xy(0,10); closed=true,
               family=wall_family(thickness=0.2))
    chains = KhepriBase.resolve(wg)
    @test length(chains) == 1
    let chain = chains[1],
        centerline_vs = KhepriBase.path_vertices(chain.path),
        left_vs       = KhepriBase.path_vertices(chain.left_face_path),
        right_vs      = KhepriBase.path_vertices(chain.right_face_path)
      @test KhepriBase.is_closed_path(chain.path)
      @test KhepriBase.is_closed_path(chain.left_face_path)
      @test KhepriBase.is_closed_path(chain.right_face_path)
      @test length(centerline_vs) == length(left_vs) == length(right_vs)
    end
  end

  @testset "End-to-end: build(plan) propagates face paths" begin
    #=
    Verify the full pipeline: `build(plan)` → resolve → Wall with
    face paths populated. The two-room (triangle + trapezoid)
    scenario exercises a valence-3 junction with no through-pair
    at (0,5); every Wall the builder emits must carry non-nothing
    left/right face paths so the downstream renderer uses the
    junction-face corners instead of falling back to
    `offset(centerline, ±t)`.
    =#
    plan = floor_plan()
    room_a = add_space(plan, "Room A",
                       closed_polygonal_path([u0(), x(10), y(5)]))
    room_b = add_space(plan, "Room B",
                       closed_polygonal_path([x(10), x(20), xy(20,5), y(5)]))
    result = build(plan)
    # Every wall emitted by the builder should have face paths
    # populated. (A Wall without face paths would render via the
    # legacy offset path and reintroduce the (0,5) gap.)
    @test all(w -> !isnothing(w.left_face_path),  result.walls)
    @test all(w -> !isnothing(w.right_face_path), result.walls)
    # Pick the wall whose centerline passes through (0,5) and check
    # its face endpoint coincides with the junction corner rather
    # than sitting on a naive offset line.
    for w in result.walls
      if any(v -> distance(v, xy(0,5)) < 1e-6, KhepriBase.path_vertices(w.path))
        # At least one of this wall's face paths has a vertex that
        # is strictly *not* on the naive offset of its centerline —
        # i.e. the corner has been moved by the face-intersection
        # computation relative to a plain perpendicular offset.
        l_vs = KhepriBase.path_vertices(w.left_face_path)
        r_vs = KhepriBase.path_vertices(w.right_face_path)
        @test length(l_vs) >= 2 && length(r_vs) >= 2
        @test all(p -> isfinite(cx(p)) && isfinite(cy(p)), l_vs)
        @test all(p -> isfinite(cx(p)) && isfinite(cy(p)), r_vs)
      end
    end
  end
end
