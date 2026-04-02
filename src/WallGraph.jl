#####################################################################
# WallGraph — Junction-aware wall network representation
#
# A WallGraph represents a network of wall segments connected at
# junction points. It resolves junction geometry (miters, T-joins)
# by merging chains of segments into multi-vertex paths, leveraging
# the existing offset_vertices/v_in_v miter math in Geometry.jl.
#
# Two construction modes:
#   1. Explicit: junction!() + segment!() for full control
#   2. Path-based: wall_path!() with auto-junction merging
#
# The resolve() function identifies chains of same-family segments
# connected at valence-2 junctions and merges them into single
# multi-vertex wall paths. T-junctions and crosses are handled by
# extending abutting walls to meet through-wall faces.

export WallJunction, WallSegmentOpening, WallSegment, WallGraph,
       ResolvedChain, WallMesh,
       wall_graph, junction!, segment!,
       add_wall_door!, add_wall_window!,
       wall_path!, find_or_create_junction!,
       segment_length, chain_junctions,
       resolve, build_walls,
       resolve_geometry, render_wall_graph,
       HasWallJoins, has_wall_joins

#=== Data Structures ===#

mutable struct WallJunction
  position::Loc
  segments::Vector{Int}  # indices into WallGraph.segments
end

struct WallSegmentOpening
  kind::Symbol            # :door or :window
  distance::Real          # meters from junction_a along centerline
  sill::Real              # height above floor (0 for doors)
  family::Union{DoorFamily, WindowFamily}
end

mutable struct WallSegment
  junction_a::Int         # index into WallGraph.junctions
  junction_b::Int
  family::WallFamily
  offset::Real
  openings::Vector{WallSegmentOpening}
end

mutable struct WallGraph
  junctions::Vector{WallJunction}
  segments::Vector{WallSegment}
  bottom_level::Level
  top_level::Level
end

# Result of resolve(): a chain of merged segments
struct ResolvedChain
  path::Path
  family::WallFamily
  offset::Real
  openings::Vector{WallSegmentOpening}
  source_segments::Vector{Int}  # original segment indices
end

#=== Construction API ===#

wall_graph(; level=default_level(),
             height=default_level_to_level_height()) =
  WallGraph(WallJunction[], WallSegment[],
            level, upper_level(level, height))

function junction!(wg::WallGraph, position)
  push!(wg.junctions, WallJunction(position, Int[]))
  length(wg.junctions)
end

function segment!(wg::WallGraph, j_a::Int, j_b::Int;
                   family=default_wall_family(),
                   offset=0)
  push!(wg.segments, WallSegment(j_a, j_b, family, offset, WallSegmentOpening[]))
  let idx = length(wg.segments)
    push!(wg.junctions[j_a].segments, idx)
    push!(wg.junctions[j_b].segments, idx)
    idx
  end
end

# Add door to a segment by index. at = distance from junction_a; nothing = centered.
function add_wall_door!(wg::WallGraph, seg_idx::Int;
                        at=nothing,
                        family=default_door_family())
  let seg = wg.segments[seg_idx],
      seg_len = segment_length(wg, seg_idx),
      dist = isnothing(at) ? (seg_len - family.width) / 2 : at,
      op = WallSegmentOpening(:door, dist, 0.0, family)
    push!(seg.openings, op)
    op
  end
end

# Add window to a segment by index. at = distance from junction_a; nothing = centered.
function add_wall_window!(wg::WallGraph, seg_idx::Int;
                          at=nothing,
                          sill=0.9,
                          family=default_window_family())
  let seg = wg.segments[seg_idx],
      seg_len = segment_length(wg, seg_idx),
      dist = isnothing(at) ? (seg_len - family.width) / 2 : at,
      op = WallSegmentOpening(:window, dist, sill, family)
    push!(seg.openings, op)
    op
  end
end

# Add door/window by junction pair (finds the segment between them)
add_wall_door!(wg::WallGraph, j_a::Int, j_b::Int; kw...) =
  add_wall_door!(wg, find_segment(wg, j_a, j_b); kw...)

add_wall_window!(wg::WallGraph, j_a::Int, j_b::Int; kw...) =
  add_wall_window!(wg, find_segment(wg, j_a, j_b); kw...)

#=== Path-Based Construction ===#

function wall_path!(wg::WallGraph, points...;
                    closed=false,
                    family=default_wall_family(),
                    offset=0)
  let tol = collinearity_tolerance(),
      jids = [find_or_create_junction!(wg, p, tol) for p in points],
      pairs = closed ? [zip(jids, [jids[2:end]..., jids[1]])...] :
                       [zip(jids[1:end-1], jids[2:end])...],
      sids = [segment!(wg, a, b, family=family, offset=offset) for (a, b) in pairs]
    sids
  end
end

function find_or_create_junction!(wg::WallGraph, pos, tol)
  for (i, j) in enumerate(wg.junctions)
    distance(j.position, pos) < tol && return i
  end
  junction!(wg, pos)
end

#=== Helpers ===#

segment_length(wg::WallGraph, seg_idx::Int) =
  let seg = wg.segments[seg_idx]
    distance(wg.junctions[seg.junction_a].position,
             wg.junctions[seg.junction_b].position)
  end

function find_segment(wg::WallGraph, j_a::Int, j_b::Int)
  for idx in wg.junctions[j_a].segments
    let seg = wg.segments[idx]
      ((seg.junction_a == j_a && seg.junction_b == j_b) ||
       (seg.junction_a == j_b && seg.junction_b == j_a)) && return idx
    end
  end
  error("No segment between junctions $j_a and $j_b")
end

junction_valence(wg::WallGraph, j_idx::Int) =
  length(wg.junctions[j_idx].segments)

# Direction vector from a junction along a segment (pointing away from junction)
function segment_direction(wg::WallGraph, seg_idx::Int, from_junction::Int)
  let seg = wg.segments[seg_idx],
      pa = wg.junctions[seg.junction_a].position,
      pb = wg.junctions[seg.junction_b].position
    from_junction == seg.junction_a ? unitized(pb - pa) : unitized(pa - pb)
  end
end

# The "other" junction of a segment
other_junction(seg::WallSegment, j_idx::Int) =
  seg.junction_a == j_idx ? seg.junction_b : seg.junction_a

#=== Chain Detection ===#

# A chain is a maximal sequence of segments connected at valence-2
# junctions with the same family and offset. Chains are the units
# that get merged into single multi-vertex wall paths.

function find_chains(wg::WallGraph)
  visited = falses(length(wg.segments))
  chains = Vector{Int}[]
  for (seg_idx, _) in enumerate(wg.segments)
    visited[seg_idx] && continue
    chain = grow_chain(wg, seg_idx, visited)
    push!(chains, chain)
  end
  chains
end

# Grow a chain in both directions from a seed segment
function grow_chain(wg::WallGraph, seed::Int, visited)
  seg = wg.segments[seed]
  visited[seed] = true
  # Grow backward from junction_a
  backward = grow_chain_direction(wg, seed, seg.junction_a, seg.family, seg.offset, visited)
  # Grow forward from junction_b
  forward = grow_chain_direction(wg, seed, seg.junction_b, seg.family, seg.offset, visited)
  [reverse(backward)..., seed, forward...]
end

# Grow chain in one direction from a junction.
# At T-junctions (valence 3), continue along the collinear "through" segment.
# At valence != 2 and non-through, stop.
function grow_chain_direction(wg::WallGraph, from_seg::Int, at_junction::Int,
                              family::WallFamily, offset::Real, visited)
  result = Int[]
  current_seg = from_seg
  current_junction = at_junction
  while true
    j = wg.junctions[current_junction]
    valence = length(j.segments)
    next_seg = if valence == 2
      # Elbow: continue to the other segment if same family
      let other = first(s for s in j.segments if s != current_seg)
        wg.segments[other].family == family &&
        wg.segments[other].offset == offset &&
        !visited[other] ? other : nothing
      end
    elseif valence == 3
      # T-junction: continue along the collinear through-pair partner
      through_partner(wg, current_seg, current_junction, family, offset, visited)
    else
      nothing  # valence 1 (free end) or 4+ (cross): stop
    end
    isnothing(next_seg) && break
    visited[next_seg] = true
    push!(result, next_seg)
    current_seg = next_seg
    current_junction = other_junction(wg.segments[next_seg], current_junction)
  end
  result
end

# At a T-junction, find the segment that is roughly collinear with from_seg
# (angle difference closest to pi) and has the same family.
function through_partner(wg::WallGraph, from_seg::Int, at_junction::Int,
                         family::WallFamily, offset::Real, visited)
  let from_dir = segment_direction(wg, from_seg, at_junction),
      from_angle = atan(cy(from_dir), cx(from_dir)),
      candidates = [(s, segment_direction(wg, s, at_junction))
                     for s in wg.junctions[at_junction].segments
                     if s != from_seg && !visited[s] &&
                        wg.segments[s].family == family &&
                        wg.segments[s].offset == offset]
    isempty(candidates) && return nothing
    let scored = [(s, abs(angle_diff(atan(cy(d), cx(d)), from_angle) - pi))
                  for (s, d) in candidates],
        (min_score, best_idx) = findmin(last, scored)
      # Only accept if roughly collinear (within ~30 degrees of straight)
      min_score < pi/6 ? scored[best_idx][1] : nothing
    end
  end
end

# Signed angle difference in [0, 2pi)
angle_diff(a, b) =
  let d = mod(a - b, 2pi)
    d < 0 ? d + 2pi : d
  end

#=== Chain Resolution ===#

function resolve(wg::WallGraph)
  chains = find_chains(wg)
  map(chains) do chain
    resolve_chain(wg, chain)
  end
end

function resolve_chain(wg::WallGraph, chain::Vector{Int})
  # Orient the chain: determine the ordered sequence of junctions
  junctions = chain_junctions(wg, chain)
  positions = [wg.junctions[j].position for j in junctions]
  seg0 = wg.segments[chain[1]]
  family = seg0.family
  offset = seg0.offset

  # Check if the chain forms a closed loop
  is_closed = length(junctions) > 2 && first(junctions) == last(junctions)

  # Adjust endpoints for T-junctions and crosses
  positions = adjust_endpoints(wg, chain, junctions, positions)

  path = is_closed ?
    closed_polygonal_path(positions[1:end-1]) :
    open_polygonal_path(positions)

  # Collect and reposition openings
  openings = collect_chain_openings(wg, chain, junctions)

  ResolvedChain(path, family, offset, openings, chain)
end

# Extract the ordered junction sequence for a chain of segments.
# For chain [s1, s2, s3], returns [j_start, j_mid1, j_mid2, j_end].
function chain_junctions(wg::WallGraph, chain::Vector{Int})
  length(chain) == 1 &&
    let seg = wg.segments[chain[1]]
      return [seg.junction_a, seg.junction_b]
    end
  # Determine orientation of first segment relative to second
  let s1 = wg.segments[chain[1]],
      s2 = wg.segments[chain[2]],
      # The shared junction between s1 and s2
      shared = if s1.junction_b == s2.junction_a || s1.junction_b == s2.junction_b
        s1.junction_b
      else
        s1.junction_a
      end,
      # First junction is the non-shared end of s1
      first_j = other_junction(s1, shared)
    result = [first_j]
    current_j = first_j
    for seg_idx in chain
      let seg = wg.segments[seg_idx],
          next_j = other_junction(seg, current_j)
        push!(result, next_j)
        current_j = next_j
      end
    end
    result
  end
end

# Adjust chain endpoint positions for T-junction and cross-junction abutments.
# When a chain ends at a T-junction or cross where it is the "abutting" wall,
# extend the endpoint to meet the through-wall's face.
function adjust_endpoints(wg::WallGraph, chain, junctions, positions)
  positions = Loc[p for p in positions]
  # Adjust start
  let j_start = first(junctions),
      valence = junction_valence(wg, j_start)
    if valence >= 3
      ext = abutment_extension(wg, chain[1], j_start)
      if !isnothing(ext)
        positions[1] = positions[1] + ext
      end
    end
  end
  # Adjust end
  let j_end = last(junctions),
      valence = junction_valence(wg, j_end)
    if valence >= 3
      ext = abutment_extension(wg, chain[end], j_end)
      if !isnothing(ext)
        positions[end] = positions[end] + ext
      end
    end
  end
  positions
end

# Compute the extension vector for a segment that abuts a through-wall
# at a T-junction or cross. Returns nothing if this segment is part of
# the through-pair (i.e., it continues through the junction).
function abutment_extension(wg::WallGraph, seg_idx::Int, at_junction::Int)
  let j = wg.junctions[at_junction],
      my_dir = segment_direction(wg, seg_idx, at_junction),
      my_angle = atan(cy(my_dir), cx(my_dir)),
      # Find the through-pair: the two most collinear segments (excluding this one)
      others = [s for s in j.segments if s != seg_idx],
      through = find_through_pair_at(wg, others, at_junction)
    isnothing(through) && return nothing
    # Check if we are part of the through pair
    (seg_idx == through[1] || seg_idx == through[2]) && return nothing
    # We are the abutting wall. Extend toward the through-wall.
    # The extension is along our direction by the through-wall's half-thickness.
    let through_seg = wg.segments[through[1]],
        through_thickness = through_seg.family.thickness,
        through_offset = through_seg.offset,
        # Which side of the through-wall are we on?
        through_dir = segment_direction(wg, through[1], at_junction),
        through_normal = vxy(-cy(through_dir), cx(through_dir)),
        side = cx(my_dir) * cx(through_normal) + cy(my_dir) * cy(through_normal),
        half_th = side > 0 ?
          l_thickness(through_offset, through_thickness) :
          r_thickness(through_offset, through_thickness)
      # Extend along our own direction (toward the junction) by half_th
      # But my_dir points AWAY from the junction, so we extend in -my_dir
      vxy(cx(my_dir) * (-half_th), cy(my_dir) * (-half_th))
    end
  end
end

# Find the most collinear pair among segments at a junction
function find_through_pair_at(wg::WallGraph, seg_indices, at_junction)
  length(seg_indices) < 2 && return nothing
  let best_pair = nothing,
      best_score = pi/6  # threshold: must be within 30 degrees of straight
    for i in 1:length(seg_indices)
      for j in i+1:length(seg_indices)
        let di = segment_direction(wg, seg_indices[i], at_junction),
            dj = segment_direction(wg, seg_indices[j], at_junction),
            ai = atan(cy(di), cx(di)),
            aj = atan(cy(dj), cx(dj)),
            diff = abs(angle_diff(ai, aj) - pi)
          if diff < best_score
            best_score = diff
            best_pair = (seg_indices[i], seg_indices[j])
          end
        end
      end
    end
    best_pair
  end
end

# Collect openings from all segments in a chain, adjusting distances
# to account for segment ordering and orientation within the chain.
function collect_chain_openings(wg::WallGraph, chain, junctions)
  openings = WallSegmentOpening[]
  cumulative_length = 0.0
  for (i, seg_idx) in enumerate(chain)
    let seg = wg.segments[seg_idx],
        seg_len = segment_length(wg, seg_idx),
        # Is this segment oriented same as chain direction?
        forward = seg.junction_a == junctions[i]
      for op in seg.openings
        let adjusted_dist = forward ?
              cumulative_length + op.distance :
              cumulative_length + seg_len - op.distance - op.family.width
          push!(openings, WallSegmentOpening(op.kind, adjusted_dist, op.sill, op.family))
        end
      end
      cumulative_length += seg_len
    end
  end
  openings
end

#=== Backend Dispatch ===#

struct HasWallJoins{T} end
has_wall_joins(::Type{<:Backend}) = HasWallJoins{false}()

#=== Build ===#

# Build wall/door/window objects from a WallGraph.
# Returns (walls=..., doors=..., windows=...) named tuple.
build_walls(wg::WallGraph) =
  has_current_backend() ?
    build_walls(has_wall_joins(typeof(top_backend())), wg) :
    build_walls(HasWallJoins{false}(), wg)

# Non-BIM backends: resolve chains, create merged walls
function build_walls(::HasWallJoins{false}, wg::WallGraph)
  chains = resolve(wg)
  walls = []
  doors = []
  windows = []
  for chain in chains
    let w = wall(chain.path,
                 bottom_level=wg.bottom_level,
                 top_level=wg.top_level,
                 family=chain.family,
                 offset=chain.offset)
      push!(walls, w)
      for op in chain.openings
        if op.kind == :door
          add_door(w, xy(op.distance, op.sill), op.family)
          push!(doors, w.doors[end])
        else
          add_window(w, xy(op.distance, op.sill), op.family)
          push!(windows, w.windows[end])
        end
      end
    end
  end
  (walls=walls, doors=doors, windows=windows)
end

# BIM backends: one wall per segment, let backend handle joins
function build_walls(::HasWallJoins{true}, wg::WallGraph)
  walls = []
  doors = []
  windows = []
  for seg in wg.segments
    let pa = wg.junctions[seg.junction_a].position,
        pb = wg.junctions[seg.junction_b].position,
        w = wall(open_polygonal_path([pa, pb]),
                 bottom_level=wg.bottom_level,
                 top_level=wg.top_level,
                 family=seg.family,
                 offset=seg.offset)
      push!(walls, w)
      for op in seg.openings
        if op.kind == :door
          add_door(w, xy(op.distance, op.sill), op.family)
          push!(doors, w.doors[end])
        else
          add_window(w, xy(op.distance, op.sill), op.family)
          push!(windows, w.windows[end])
        end
      end
    end
  end
  (walls=walls, doors=doors, windows=windows)
end

#=== Junction-Aware Mesh Generation ===#

# Output mesh for one wall segment: vertices + quad faces + per-face materials.
struct WallMesh
  vertices::Vector{Loc}
  quads::Vector{NTuple{4, Int}}
  quad_materials::Vector{Material}
end

# 2D line intersection: p1 + t*d1 = p2 + s*d2. Returns an XY or nothing.
function line_intersection_2d(p1, d1, p2, d2)
  let denom = cx(d1) * cy(d2) - cy(d1) * cx(d2)
    abs(denom) < 1e-10 && return nothing
    let dpx = cx(p2) - cx(p1),
        dpy = cy(p2) - cy(p1),
        t = (dpx * cy(d2) - dpy * cx(d2)) / denom
      xy(cx(p1) + cx(d1) * t, cy(p1) + cy(d1) * t)
    end
  end
end

const MITER_LIMIT = 4.0

# Compute (right_corner, left_corner) for every segment at every junction.
# "right" and "left" are relative to the segment's outgoing direction from that junction.
function compute_junction_corners(wg::WallGraph)
  result = Dict{Int, Dict{Int, Tuple{Loc, Loc}}}()
  for (j_idx, j) in enumerate(wg.junctions)
    pos = j.position
    segs = j.segments
    n = length(segs)
    if n == 0
      result[j_idx] = Dict{Int, Tuple{Loc, Loc}}()
      continue
    end
    if n == 1
      # Valence 1: flat perpendicular cap
      let seg = wg.segments[segs[1]],
          dir = segment_direction(wg, segs[1], j_idx),
          l_th = l_thickness(seg.offset, seg.family.thickness),
          r_th = r_thickness(seg.offset, seg.family.thickness)
        result[j_idx] = Dict(segs[1] => (
          xy(cx(pos) + cy(dir) * r_th, cy(pos) - cx(dir) * r_th),
          xy(cx(pos) - cy(dir) * l_th, cy(pos) + cx(dir) * l_th)))
      end
      continue
    end
    # Valence 2+: sort by outgoing angle, intersect adjacent offset lines
    sorted = sort([(s, segment_direction(wg, s, j_idx)) for s in segs],
                  by=((s, d),) -> atan(cy(d), cx(d)))
    corners = Loc[]
    for k in 1:n
      let dir_curr = sorted[k][2],
          dir_next = sorted[mod1(k + 1, n)][2],
          seg_curr = wg.segments[sorted[k][1]],
          seg_next = wg.segments[sorted[mod1(k + 1, n)][1]],
          l_th = l_thickness(seg_curr.offset, seg_curr.family.thickness),
          r_th = r_thickness(seg_next.offset, seg_next.family.thickness),
          p_left = xy(cx(pos) - cy(dir_curr) * l_th,
                      cy(pos) + cx(dir_curr) * l_th),
          p_right = xy(cx(pos) + cy(dir_next) * r_th,
                       cy(pos) - cx(dir_next) * r_th),
          corner = line_intersection_2d(p_left, dir_curr, p_right, dir_next)
        if isnothing(corner) || distance(pos, corner) > MITER_LIMIT * max(l_th, r_th)
          push!(corners, xy((cx(p_left) + cx(p_right)) / 2,
                            (cy(p_left) + cy(p_right)) / 2))
        else
          push!(corners, corner)
        end
      end
    end
    # corners[k] is between sorted[k] and sorted[k+1]:
    #   sorted[k]'s left corner, sorted[k+1]'s right corner
    j_corners = Dict{Int, Tuple{Loc, Loc}}()
    for k in 1:n
      j_corners[sorted[k][1]] = (corners[mod1(k - 1, n)], corners[k])
    end
    result[j_idx] = j_corners
  end
  result
end

# 2D wall quad for a segment in A->B frame: (right_a, right_b, left_b, left_a).
function segment_quad_2d(wg, seg_idx, jc)
  let seg = wg.segments[seg_idx],
      (right_a, left_a) = jc[seg.junction_a][seg_idx],
      (right_from_b, left_from_b) = jc[seg.junction_b][seg_idx]
    (right_a, left_from_b, right_from_b, left_a)
  end
end

# Linearly interpolate between two 2D points.
lerp2d(a, b, t) = xy(cx(a) + (cx(b) - cx(a)) * t,
                     cy(a) + (cy(b) - cy(a)) * t)

# Generate the 3D mesh for one wall segment with junction-aware corners.
function segment_mesh(wg, seg_idx, jc)
  seg = wg.segments[seg_idx]
  zbot = wg.bottom_level.height
  ztop = wg.top_level.height
  seg_len = segment_length(wg, seg_idx)
  right_a, right_b, left_b, left_a = segment_quad_2d(wg, seg_idx, jc)
  fam = seg.family
  lmat = fam.left_material
  rmat = fam.right_material
  smat = fam.side_material
  vertices = Loc[]
  quads = NTuple{4, Int}[]
  mats = Material[]
  vi = Ref(0)
  vtx(p2d, z) = (push!(vertices, xyz(cx(p2d), cy(p2d), z)); vi[] += 1; vi[])

  # Horizontal break points as fractions of seg_len
  breaks = Float64[0.0]
  op_ranges = Tuple{Float64, Float64, WallSegmentOpening}[]
  for op in sort(seg.openings, by=o -> o.distance)
    let t0 = op.distance / seg_len,
        t1 = (op.distance + op.family.width) / seg_len
      push!(breaks, t0, t1)
      push!(op_ranges, (t0, t1, op))
    end
  end
  push!(breaks, 1.0)
  sort!(unique!(breaks))

  # Generate left and right face quads for each horizontal strip
  for i in 1:length(breaks) - 1
    t0, t1 = breaks[i], breaks[i + 1]
    (t1 - t0) < 1e-10 && continue
    la0, la1 = lerp2d(left_a, left_b, t0), lerp2d(left_a, left_b, t1)
    ra0, ra1 = lerp2d(right_a, right_b, t0), lerp2d(right_a, right_b, t1)
    # Is this strip inside an opening?
    op_match = nothing
    for (ot0, ot1, op) in op_ranges
      if t0 >= ot0 - 1e-10 && t1 <= ot1 + 1e-10
        op_match = op; break
      end
    end
    if isnothing(op_match)
      # Solid strip
      push!(quads, (vtx(la0, zbot), vtx(la1, zbot), vtx(la1, ztop), vtx(la0, ztop))); push!(mats, lmat)
      push!(quads, (vtx(ra1, zbot), vtx(ra0, zbot), vtx(ra0, ztop), vtx(ra1, ztop))); push!(mats, rmat)
    else
      # Opening strip: subdivide vertically, add reveals
      let z0 = zbot + op_match.sill,
          z1 = zbot + op_match.sill + op_match.family.height
        # Below opening
        if z0 > zbot + 1e-10
          push!(quads, (vtx(la0, zbot), vtx(la1, zbot), vtx(la1, z0), vtx(la0, z0))); push!(mats, lmat)
          push!(quads, (vtx(ra1, zbot), vtx(ra0, zbot), vtx(ra0, z0), vtx(ra1, z0))); push!(mats, rmat)
        end
        # Above opening
        if z1 < ztop - 1e-10
          push!(quads, (vtx(la0, z1), vtx(la1, z1), vtx(la1, ztop), vtx(la0, ztop))); push!(mats, lmat)
          push!(quads, (vtx(ra1, z1), vtx(ra0, z1), vtx(ra0, ztop), vtx(ra1, ztop))); push!(mats, rmat)
        end
        # Reveals: sill, lintel, left jamb, right jamb
        push!(quads, (vtx(la0, z0), vtx(la1, z0), vtx(ra1, z0), vtx(ra0, z0))); push!(mats, smat)
        push!(quads, (vtx(la1, z1), vtx(la0, z1), vtx(ra0, z1), vtx(ra1, z1))); push!(mats, smat)
        push!(quads, (vtx(la0, z0), vtx(ra0, z0), vtx(ra0, z1), vtx(la0, z1))); push!(mats, smat)
        push!(quads, (vtx(ra1, z0), vtx(la1, z0), vtx(la1, z1), vtx(ra1, z1))); push!(mats, smat)
      end
    end
  end
  # Top and bottom faces
  push!(quads, (vtx(left_a, ztop), vtx(left_b, ztop), vtx(right_b, ztop), vtx(right_a, ztop))); push!(mats, smat)
  push!(quads, (vtx(left_b, zbot), vtx(left_a, zbot), vtx(right_a, zbot), vtx(right_b, zbot))); push!(mats, smat)
  # End caps at free ends only
  if junction_valence(wg, seg.junction_a) == 1
    push!(quads, (vtx(right_a, zbot), vtx(left_a, zbot), vtx(left_a, ztop), vtx(right_a, ztop))); push!(mats, smat)
  end
  if junction_valence(wg, seg.junction_b) == 1
    push!(quads, (vtx(left_b, zbot), vtx(right_b, zbot), vtx(right_b, ztop), vtx(left_b, ztop))); push!(mats, smat)
  end
  WallMesh(vertices, quads, mats)
end

# Generate meshes for all wall segments with junction-aware geometry.
resolve_geometry(wg::WallGraph) =
  let jc = compute_junction_corners(wg)
    [segment_mesh(wg, i, jc) for i in 1:length(wg.segments)]
  end

# Render a wall graph directly using the current backend's b_quad.
function render_wall_graph(wg::WallGraph)
  let meshes = resolve_geometry(wg),
      b = top_backend(),
      refs = []
    for mesh in meshes
      for (k, (i1, i2, i3, i4)) in enumerate(mesh.quads)
        append!(refs, b_quad(b,
          mesh.vertices[i1], mesh.vertices[i2],
          mesh.vertices[i3], mesh.vertices[i4],
          ref_value(b, mesh.quad_materials[k])))
      end
    end
    refs
  end
end

