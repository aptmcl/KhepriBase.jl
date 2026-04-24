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
       wall_graph, junction!, segment!, arc_segment!,
       add_wall_door!, add_wall_window!,
       wall_path!, find_or_create_junction!,
       segment_length, chain_junctions,
       resolve, build_walls,
       resolve_geometry, render_wall_graph,
       line_arc_intersection_2d, arc_arc_intersection_2d,
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

#=
`arc` carries the curvature of a wall segment. `nothing` means a
straight segment — the default, and the common case for rectangular
buildings. An `ArcPath` describes a circular-arc segment whose
endpoints coincide with the two junction positions and whose
`start_angle` / `amplitude` settle which way around the centre the
wall sweeps (CW vs CCW, major vs minor arc).

Arc walls are supported end-to-end by the BIM backend via
`wall(arc_path, …)`; this field lets the wall graph preserve the
original curvature instead of discretising it into a polyline before
passing through.

See `arc_segment!` for the friendly constructor that derives an
`ArcPath` from (centre, junctions, amplitude).
=#
mutable struct WallSegment
  junction_a::Int         # index into WallGraph.junctions
  junction_b::Int
  family::WallFamily
  offset::Real
  openings::Vector{WallSegmentOpening}
  arc::Union{Nothing, ArcPath}
end

mutable struct WallGraph
  junctions::Vector{WallJunction}
  segments::Vector{WallSegment}
  bottom_level::Level
  top_level::Level
end

#=
Result of `resolve()`: one merged chain of wall segments.

`path` is the centerline — an `open_polygonal_path` /
`closed_polygonal_path` for a straight chain, an `ArcPath` for a
co-circular arc chain. Kept because openings (doors, windows) are
positioned along the chain by arc-length on this centerline, which
must stay stable regardless of how the junctions resolve.

`left_face_path` / `right_face_path` are the two face polylines
whose vertices come from the face-intersection model at each
junction the chain touches. They replace `offset(path, ±thickness)`
in the backend: a straight `offset` can't produce the right corner
at a 3-way-no-through-pair junction, but the face polylines can.

For arc-carrying chains the face paths currently mirror the
centerline's offset (linear line-tangent approximation at
junctions); the existing arc offset math remains correct for
pure-arc chains, and line-arc / arc-arc exact corners are a
follow-up.
=#
struct ResolvedChain
  path::Path
  left_face_path::Path
  right_face_path::Path
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
                   offset=0,
                   arc::Union{Nothing, ArcPath}=nothing)
  j_a == j_b && error("Cannot create a self-loop segment (junction $j_a)")
  push!(wg.segments, WallSegment(j_a, j_b, family, offset, WallSegmentOpening[], arc))
  let idx = length(wg.segments)
    push!(wg.junctions[j_a].segments, idx)
    push!(wg.junctions[j_b].segments, idx)
    idx
  end
end

#=
Create an arc-shaped wall segment between two existing junctions.
`center` is the arc's centre; `amplitude` is the signed sweep angle
(positive = counterclockwise). `radius` is derived from the centre
and junction positions; `start_angle` is the angle of junction_a
relative to the centre. The caller is responsible for ensuring both
junctions lie on the circle — their positions must be at distance
`radius` from the centre.
=#
"Create a circular-arc wall segment sweeping from `j_a` to `j_b` around `center` through `amplitude` radians."
function arc_segment!(wg::WallGraph, j_a::Int, j_b::Int;
                      center::Loc, amplitude::Real,
                      family=default_wall_family(),
                      offset=0)
  j_a == j_b && error("Cannot create a self-loop arc segment (junction $j_a)")
  let pa = wg.junctions[j_a].position,
      radius = distance(center, pa),
      start_angle = atan(cy(pa) - cy(center), cx(pa) - cx(center)),
      arc = arc_path(center, radius, start_angle, amplitude)
    segment!(wg, j_a, j_b; family=family, offset=offset, arc=arc)
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

#=
Length of a wall segment along its actual path.

For a straight segment the answer is the chord length between the
two junctions; for an arc it is `radius · |amplitude|` (circular
arc length), not the chord. Openings positioned by arc-length along
the centerline — via `WallSegmentOpening.distance` — therefore sit
correctly on curved walls.
=#
segment_length(wg::WallGraph, seg_idx::Int) =
  let seg = wg.segments[seg_idx]
    isnothing(seg.arc) ?
      distance(wg.junctions[seg.junction_a].position,
               wg.junctions[seg.junction_b].position) :
      path_length(seg.arc)
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

#=
Unit direction vector "away from" a junction along a segment.

Straight segment: the chord direction, as before.

Arc segment: the tangent to the arc at the junction. The tangent is
perpendicular to the radius, with its sign chosen to point along the
arc's sweep direction away from the junction. This is what the
junction miter math wants — the direction a caller's eye follows
leaving the junction along that wall.
=#
function segment_direction(wg::WallGraph, seg_idx::Int, from_junction::Int)
  let seg = wg.segments[seg_idx]
    if isnothing(seg.arc)
      let pa = wg.junctions[seg.junction_a].position,
          pb = wg.junctions[seg.junction_b].position
        from_junction == seg.junction_a ? unitized(pb - pa) : unitized(pa - pb)
      end
    else
      _arc_tangent(seg.arc, from_junction == seg.junction_a)
    end
  end
end

# Tangent at one of an arc's endpoints, pointing outward along the
# arc's sweep direction. `at_start` picks which endpoint.
function _arc_tangent(arc::ArcPath, at_start::Bool)
  let s = sign(arc.amplitude),
      θ = at_start ? arc.start_angle : arc.start_angle + arc.amplitude,
      rad = vpol(1.0, θ, arc.center.cs)
    unitized(at_start ?
      vxy(-cy(rad) * s,  cx(rad) * s, arc.center.cs) :
      vxy( cy(rad) * s, -cx(rad) * s, arc.center.cs))
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
#=
Grow a chain from `from_seg` through `at_junction` and outward.

Chains merge *only* at valence-2 elbows. A valence-3+ junction is
always a chain terminator, even when two of the incident walls
form a collinear through-pair. Why: the abutting wall at a
through-pair T-junction intrudes onto the chain's face on one
side, so the chain's left and right face polylines accumulate
different numbers of corners across the junction (one for the
non-abutment side, two for the abutment side). Merging across
the T-junction therefore produces asymmetric face polylines,
which `b_quad_strip` can't render. Stopping the chain at every
valence-3+ junction keeps each chain's two face polylines
symmetric — each non-abutment T-junction endpoint gets one
clean corner from `junction_face_corners`, and the backend
renders every corner as its face-intersection miter rather than
a flat perpendicular cap.

The cost is more walls — a chain that previously merged through
a T-junction now emits one wall per side. For BIM backends that
do their own wall-join logic (`HasWallJoins{true}`) this is the
natural representation anyway. For non-BIM backends the
additional walls render cleanly because each pair shares the
junction corner via `junction_face_corners` — no gap, no
overshoot.

See also: `junction_face_corners`, `wall_face_polylines`.
=#
function grow_chain_direction(wg::WallGraph, from_seg::Int, at_junction::Int,
                              family::WallFamily, offset::Real, visited)
  result = Int[]
  current_seg = from_seg
  current_junction = at_junction
  while true
    j = wg.junctions[current_junction]
    valence = length(j.segments)
    next_seg = if valence == 2
      # Elbow: continue to the other segment if same family and
      # "chainable" with the current one. Two straight segments
      # always chain (a polyline is a valid multi-segment path); two
      # arc segments chain only if they are co-circular (same centre
      # and radius) and continue the sweep direction — a bend from
      # one radius to another, or a straight-arc transition, must
      # stay a hard junction so the backend can emit separate walls.
      let other = first(s for s in j.segments if s != current_seg),
          cur = wg.segments[current_seg],
          nxt = wg.segments[other]
        nxt.family == family &&
        nxt.offset == offset &&
        !visited[other] &&
        _chainable_segments(cur, nxt) ? other : nothing
      end
    else
      nothing  # valence 1 (free end), 3 (T-junction), 4+ (cross) — stop
    end
    isnothing(next_seg) && break
    visited[next_seg] = true
    push!(result, next_seg)
    current_seg = next_seg
    current_junction = other_junction(wg.segments[next_seg], current_junction)
  end
  result
end

# Can two segments be merged into one chain? Straight-straight is
# always OK; arc-arc is OK iff the two arcs share a centre and radius
# (they are sub-arcs of the same circle). Mixed straight/arc and
# distinct-circle arc-arc pairs break the chain.
#
# See also: `resolve_chain`, which consumes chains and emits either
# a polyline (straight-only) or a single `ArcPath` (single-arc), and
# in Tier 2 a merged `ArcPath` for co-circular arc chains.
function _chainable_segments(s1::WallSegment, s2::WallSegment)
  if isnothing(s1.arc) && isnothing(s2.arc)
    return true
  elseif !isnothing(s1.arc) && !isnothing(s2.arc)
    let c1 = s1.arc.center, c2 = s2.arc.center,
        tol = coincidence_tolerance()
      distance(c1, c2) < tol && abs(s1.arc.radius - s2.arc.radius) < tol
    end
  else
    false  # arc + straight is a hard junction
  end
end

# At a T-junction, find the segment that is roughly collinear with from_seg
# (angle difference closest to pi), has the same family, and is
# chainable with it (straight+straight or co-circular arc+arc).
function through_partner(wg::WallGraph, from_seg::Int, at_junction::Int,
                         family::WallFamily, offset::Real, visited)
  let from_dir = segment_direction(wg, from_seg, at_junction),
      from_angle = atan(cy(from_dir), cx(from_dir)),
      cur = wg.segments[from_seg],
      candidates = [(s, segment_direction(wg, s, at_junction))
                     for s in wg.junctions[at_junction].segments
                     if s != from_seg && !visited[s] &&
                        wg.segments[s].family == family &&
                        wg.segments[s].offset == offset &&
                        _chainable_segments(cur, wg.segments[s])]
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

#=
Resolve the whole graph into chains. The face-corner table is
computed once for the graph and shared across every `resolve_chain`
call — each chain needs the corners at its own endpoints plus any
interior valence-2 elbows it traverses, so pre-computing all of
them avoids O(chains·junctions) recomputation on large graphs.
=#
function resolve(wg::WallGraph)
  chains = find_chains(wg)
  all_corners = all_junction_face_corners(wg)
  map(chains) do chain
    resolve_chain(wg, chain; all_corners=all_corners)
  end
end

#=
Emit a `ResolvedChain` whose `path` is the right shape for the
segments it collects:

  - all straight → an `open_polygonal_path` (or closed variant for a
    cycle) as before, with T-junction abutment corrections applied.
  - single arc → the `ArcPath` itself; the backend draws the wall as
    a curve and `subpath` places openings by arc-length.
  - co-circular arcs → merged into one `ArcPath` whose amplitude is
    the sum of the parts (same sign; `_chainable_segments` guarantees
    same centre and radius).
  - mixed arc / straight in one chain → not constructible given
    `_chainable_segments`, so this case is unreachable.
=#
function resolve_chain(wg::WallGraph, chain::Vector{Int};
                       all_corners::Union{Nothing, Dict{Int, Dict{Int, Tuple{Loc, Loc}}}}=nothing)
  junctions = chain_junctions(wg, chain)
  seg0 = wg.segments[chain[1]]
  family = seg0.family
  offset = seg0.offset
  is_closed = length(junctions) > 2 && first(junctions) == last(junctions)
  corners = isnothing(all_corners) ? all_junction_face_corners(wg) : all_corners

  path = if any(!isnothing(wg.segments[s].arc) for s in chain)
    _resolve_arc_chain(wg, chain, junctions)
  else
    positions = [wg.junctions[j].position for j in junctions]
    positions = adjust_endpoints(wg, chain, junctions, positions)
    is_closed ?
      closed_polygonal_path(positions[1:end-1]) :
      open_polygonal_path(positions)
  end

  # Face polylines — the clean junction-aware inner/outer faces.
  #
  # For closed chains, the first and last corners of each face
  # polyline are the "same" chain-left (or chain-right) corner at
  # the closing junction, computed from the two neighbour-pair
  # intersections. They agree mathematically, but floating-point
  # roundoff in two independent `line_intersection_2d` calls can
  # leave them microscopically different. A tolerance-based strip
  # that decides per face independently can then produce left and
  # right polylines with *different* vertex counts — which is
  # catastrophic downstream: `b_quad_strip_closed` indexes into a
  # combined vertex array whose face list assumes matched lengths,
  # and AutoCAD rejects the SubDMesh with `eInvalidIndex`.
  #
  # The `is_closed` flag is the authoritative answer to "does this
  # chain loop back?" — it comes from the junction-index sequence,
  # not from coordinate comparison. Trust it: when set, always
  # strip the trailing corner on both faces, yielding two
  # `closed_polygonal_path`s with identical vertex counts equal to
  # the centerline's.
  (l_pts, r_pts) = wall_face_polylines(wg, chain, junctions, corners)
  (left_face_path, right_face_path) = is_closed ?
    (closed_polygonal_path(l_pts[1:end-1]),
     closed_polygonal_path(r_pts[1:end-1])) :
    (open_polygonal_path(l_pts),
     open_polygonal_path(r_pts))

  openings = collect_chain_openings(wg, chain, junctions)
  ResolvedChain(path, left_face_path, right_face_path, family, offset, openings, chain)
end

# Produce an `ArcPath` for a chain whose segments are all arcs (and
# co-circular, by `_chainable_segments`). Walks the chain in junction
# order, adding each arc's amplitude to the running total. Orientation
# of each arc vs the chain traversal flips its amplitude's sign.
function _resolve_arc_chain(wg::WallGraph, chain::Vector{Int}, junctions::Vector{Int})
  seg1 = wg.segments[chain[1]]
  forward = seg1.junction_a == junctions[1]
  first_arc = seg1.arc
  # Amplitude summed with orientation: +amplitude when the chain
  # walks the arc in its natural sweep direction, −amplitude otherwise.
  amp = forward ? first_arc.amplitude : -first_arc.amplitude
  # Starting angle is at junction[1], which in "forward" orientation
  # is the arc's natural start, otherwise its natural end.
  start_angle = forward ?
    first_arc.start_angle :
    first_arc.start_angle + first_arc.amplitude
  for k in 2:length(chain)
    let seg = wg.segments[chain[k]],
        fwd = seg.junction_a == junctions[k]
      amp += fwd ? seg.arc.amplitude : -seg.arc.amplitude
    end
  end
  arc_path(first_arc.center, first_arc.radius, start_angle, amp)
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
#
# The algebra below is straight-wall geometry: it divides by a dot
# product between the abutting direction and the through-pair normal.
# For arc through-walls the normal rotates along the arc, making that
# projection ill-defined at the junction. To avoid producing `Inf`/
# `NaN` endpoint extensions in arc-abutment cases, we skip the
# extension when either through-pair member is an arc; the BIM
# backend is left to render the abutting wall ending exactly at the
# junction position. Clean arc-abutment corners are on the Tier 3
# roadmap (line / arc and arc / arc offset-line intersections).
function abutment_extension(wg::WallGraph, seg_idx::Int, at_junction::Int)
  let j = wg.junctions[at_junction],
      my_dir = segment_direction(wg, seg_idx, at_junction),
      others = [s for s in j.segments if s != seg_idx],
      through = find_through_pair_at(wg, others, at_junction)
    isnothing(through) && return nothing
    (seg_idx == through[1] || seg_idx == through[2]) && return nothing
    # Arc through-walls: skip extension (see comment above).
    (!isnothing(wg.segments[through[1]].arc) ||
     !isnothing(wg.segments[through[2]].arc)) && return nothing
    # Straight through-wall: extend along our direction to the through-wall's near face.
    let through_seg = wg.segments[through[1]],
        through_thickness = through_seg.family.thickness,
        through_offset = through_seg.offset,
        through_dir = segment_direction(wg, through[1], at_junction),
        through_normal = vxy(-cy(through_dir), cx(through_dir)),
        side = cx(my_dir) * cx(through_normal) + cy(my_dir) * cy(through_normal),
        half_th = side > 0 ?
          l_thickness(through_offset, through_thickness) :
          r_thickness(through_offset, through_thickness)
      abs(side) < parallelism_tolerance() && return nothing
      let t = half_th / abs(side)
        vxy(cx(my_dir) * t, cy(my_dir) * t)
      end
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

#=== General n-way junction resolution ===#

#=
Clean wall joins at every valence, including 3-way corners without a
collinear through-pair.

The axis-endpoint model — "where does the wall's centerline stop?" —
cannot be made gap-free for arbitrary junctions. Ending the axis at
the neighbour's *inner surface* leaves a gap of at most `thickness/2`
along the centerline; ending it at the neighbour's *axis* makes a
thick abutting wall protrude through the neighbour's outer face.
Both errors scale with `thickness / sin(join_angle)`, so any non-
right-angled corner exposes one of them.

The face-intersection model side-steps the axis entirely. A junction
is a region, not a point; its boundary is the set of pairwise
intersections of its incident walls' offset curves in angular order.
Each wall receives *two* corner vertices at each junction — one on
its left face, one on its right face — which come from intersecting
its own face with the adjacent walls' faces:

    right_corner(w) = (w's right face) ∩ (CW-neighbour's left face)
    left_corner(w)  = (w's left face)  ∩ (CCW-neighbour's right face)

Where "CCW-neighbour" is the wall immediately counter-clockwise from
`w` when the incident walls are sorted by outgoing direction around
the junction position, and "CW-neighbour" the symmetric opposite.
For valence 1 (free end), the wall has no neighbours and the
endpoints fall back to its own face offsets. For valence 2, the two
walls are each other's neighbours and the corners are the classical
miter. For valence 3 with a collinear through-pair, this matches the
result `abutment_extension` computes. For valence 3 without a
through-pair — the (0,5) corner case in the two-room scenario — the
formula still produces clean corners whereas the old logic returned
`nothing` and left flat caps.

Convention: "left" and "right" are always relative to a wall's
*outgoing* direction at the specific junction — that is, looking
along the wall away from the junction, left is the CCW perpendicular.
A wall therefore sees different left/right sides at its two
junctions (they are its two canonical faces, just seen from opposite
ends); the chain-assembly code in `wall_face_polylines` flips the
sides appropriately so each face polyline threads the correct
corners as the chain is traversed.

Degeneracies:
  - CCW- and CW-neighbour faces parallel to `w`'s face: no
    intersection. Fall back to the nominal offset point at the
    junction (`J + thickness · perp(d)`) so the chain stays planar.
  - Valence 1 free end: same fallback, by construction (no neighbour
    means no intersection).

See also: `line_intersection_2d`, `line_arc_intersection_2d`,
`arc_arc_intersection_2d` (the underlying primitives per segment type).
=#

# Offset-face line at a junction. `side ∈ (:left, :right)` names the
# face relative to the wall's *outgoing* direction at that junction.
# Returns `(point_on_line, direction_vector)` — a parametric line in
# world coordinates. Arc segments currently fall back to their tangent
# line; exact arc-aware intersections are delegated to the caller.
function wall_face_line_at(wg::WallGraph, seg_idx::Int, at_junction::Int, side::Symbol)
  let seg = wg.segments[seg_idx],
      d   = segment_direction(wg, seg_idx, at_junction),
      J   = wg.junctions[at_junction].position,
      th  = seg.family.thickness,
      off = seg.offset,
      l_t = l_thickness(off, th + seg.family.left_coating_thickness),
      r_t = r_thickness(off, th + seg.family.right_coating_thickness),
      n_ccw = vxy(-cy(d),  cx(d)),   # +90° (left of outgoing)
      n_cw  = vxy( cy(d), -cx(d))    # -90° (right of outgoing)
    side == :left  ? (J + n_ccw * l_t, d) :
    side == :right ? (J + n_cw  * r_t, d) :
    error("wall_face_line_at: side must be :left or :right, got $(side)")
  end
end

# Segments at a junction sorted by their outgoing-direction angle
# (atan2), counter-clockwise from +X. Empty at orphan junctions.
function segments_sorted_ccw(wg::WallGraph, j_idx::Int)
  let j = wg.junctions[j_idx]
    sort(j.segments; by = s -> let d = segment_direction(wg, s, j_idx)
      atan(cy(d), cx(d))
    end)
  end
end

#=
Per-junction corners, unified across rendering pipelines.

Both `render_wall_graph` (the direct mesh generator for non-BIM
backends) and `build(plan)` (which produces `Wall` proxies that
flow through `b_wall_no_openings_faces`) need the same corner
geometry at each junction — otherwise the same wall graph renders
differently depending on which entry point was used.

For historical reasons the two pipelines settled on opposite tuple
orders: the mesh generator stores `(right_corner, left_corner)`
(used by `segment_quad_2d` / `junction_cap_mesh`), while the
faces-aware wall renderer stores `(left_corner, right_corner)`
(used by `wall_face_polylines` / this file's `junction_cap_polygon`).

Rather than pick a winner, delegate: compute once via
`compute_junction_corners`'s per-junction helper (which routes
through `t_junction_corners` for through-pair T-junctions and
`miter_corners` for the general n-way case), and swap the tuple
order for this accessor's callers.
=#
junction_face_corners(wg::WallGraph, j_idx::Int) =
  let raw = compute_junction_corners(wg)[j_idx]
    Dict{Int, Tuple{Loc, Loc}}(
      seg => (t[2], t[1]) for (seg, t) in raw)
  end

#=
Face polylines for a chain.

Given a chain `(segments, junctions)` — where `junctions[k]` is at
the start of `segments[k]` and `junctions[end]` at the end of
`segments[end]` — walk the chain assembling two parallel polylines,
one along the chain's left face and one along its right face.

Convention-B to chain-face mapping:
  - At `junctions[k]` (segment `k`'s START in chain order), the
    outgoing direction at the junction *is* the chain's forward
    direction, so `(left_B, right_B) == (chain_left, chain_right)`.
  - At `junctions[k+1]` (segment `k`'s END in chain order), the
    outgoing direction is the *reverse* of the chain's forward
    direction, so the perpendiculars flip and
    `(left_B, right_B) == (chain_right, chain_left)`.

Interior-junction dedup rule:

  - Valence 2 (elbow): the two adjacent segments' corners at this
    junction coincide on both sides (same mutual-miter point),
    so we emit the corner once per face.

  - Valence 3+ (T-junction or cross inside a chain): on the side
    the abutting wall enters, seg_k's chain-left (or -right)
    corner lies at the intersection with the abutting wall's
    left face, while seg_{k+1}'s lies at the intersection with
    the abutting wall's right face — they differ by the
    abutting wall's full thickness. On the opposite side, with
    no wall to intersect, both segments fall back to the same
    nominal offset point.

    If we deduped normally, the abutment side would keep both
    corners (two vertices) and the non-abutment side would
    collapse to one — mismatched face polyline lengths, which
    `b_quad_strip` interleaves 1:1 and sends to the backend as
    a SubDMesh with out-of-range face indices (`eInvalidIndex`).

    So at valence ≥ 3 we emit *both* corners on *both* sides.
    On the abutment side this correctly represents the wall's
    surface stepping around the intruding wall. On the non-
    abutment side it introduces a zero-length edge at the
    junction — geometrically inert (zero area, no visible
    artifact) and critically preserves the left/right count
    invariant.

Returns `(left_points::Vector{Loc}, right_points::Vector{Loc})` —
the two face polylines in chain-traversal order, with matched
vertex counts by construction.
=#
function wall_face_polylines(wg::WallGraph, chain::Vector{Int}, junctions::Vector{Int},
                             all_corners::Dict{Int, Dict{Int, Tuple{Loc, Loc}}})
  left_pts = Loc[]
  right_pts = Loc[]
  n = length(chain)
  for (k, seg_idx) in enumerate(chain)
    j_start = junctions[k]
    j_end   = junctions[k + 1]
    c_start = all_corners[j_start][seg_idx]  # (left_B, right_B)
    c_end   = all_corners[j_end][seg_idx]
    # Emit the start-corner *except* at a valence-2 elbow that the
    # previous iteration has already covered.
    if k == 1 || junction_valence(wg, j_start) != 2
      push!(left_pts,  c_start[1])
      push!(right_pts, c_start[2])
    end
    # Always emit the end-corner — it's either this segment's own
    # endpoint (for the last segment) or will be complemented by
    # the next segment's start-corner (at valence ≥ 3), or dropped
    # by its absence (at valence 2).
    push!(left_pts,  c_end[2])
    push!(right_pts, c_end[1])
  end
  (left_pts, right_pts)
end

# Compute face corners for every junction in the graph with the
# `(left, right)` tuple order my callers expect. Delegates to the
# unified `compute_junction_corners` (which yields `(right, left)`
# tuples for the mesh generator) and swaps the order once.
all_junction_face_corners(wg::WallGraph) =
  let raw = compute_junction_corners(wg)
    Dict{Int, Dict{Int, Tuple{Loc, Loc}}}(
      j_idx => Dict{Int, Tuple{Loc, Loc}}(
        seg => (t[2], t[1]) for (seg, t) in raw[j_idx])
      for j_idx in 1:length(wg.junctions))
  end

#=
Junction cap polygon — the floor/ceiling plate that closes the
N-gonal gap between the incident walls' top faces at a valence-≥3
junction.

Each wall's top face is a quad strip ending at its cap line
(`left_corner → right_corner`) at the junction. For `N` walls
meeting at the junction, those `N` cap lines form the edges of an
`N`-gon whose interior sits between the walls — uncovered by any
single wall's top face, and therefore visible as a dark polygon
when you look at the junction from above.

The fix is a flat `N`-gon that fills that interior. Its vertices
are the `N` *shared* face-corners at the junction, collected by
walking the incident walls in CCW angular order and taking each
wall's `left_corner` (which is also the CCW-neighbour's
`right_corner`).

When a cap is *not* needed:

  * Valence < 3: there is no N-gonal gap — a valence-2 elbow is
    a single miter covered by the two walls' merged chain, and a
    valence-1 free end has its own perpendicular cap.

  * Valence 3 with a collinear through-pair: the through-wall
    runs continuously across the junction (the two through
    segments share the same top face), so there is nothing to
    fill. The abutting wall ends on the through-wall's face —
    the through-wall's own top face already covers the area
    between the three cap lines. Adding a separate cap here
    would produce a coplanar overlap and z-fight. Pipeline 1's
    mesh generator in `render_wall_graph` skips it; we match.

See also: `junction_face_corners`, `wall_face_polylines`,
`find_through_pair_at`, `b_surface_polygon`.
=#
"Junction cap polygon vertices at a valence-≥3 junction, in CCW angular order."
function junction_cap_polygon(wg::WallGraph, j_idx::Int,
                              corners::Union{Nothing, Dict{Int, Tuple{Loc, Loc}}}=nothing)
  sorted = segments_sorted_ccw(wg, j_idx)
  n = length(sorted)
  n < 3 && return Loc[]
  # Through-pair T-junctions need no cap (see prose block above).
  n == 3 && !isnothing(find_through_pair_at(wg, sorted, j_idx)) && return Loc[]
  cs = isnothing(corners) ? junction_face_corners(wg, j_idx) : corners
  [cs[seg][1] for seg in sorted]
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

# Non-BIM backends: use junction-aware mesh generation
build_walls(::HasWallJoins{false}, wg::WallGraph) =
  render_wall_graph(wg)

# BIM backends: one wall per segment, let backend handle joins.
# Arc segments emit a curved `wall(arc_path, …)` so backends like
# AutoCAD and Revit draw a native circular wall instead of a
# polyline approximation.
function build_walls(::HasWallJoins{true}, wg::WallGraph)
  walls = []
  doors = []
  windows = []
  for seg in wg.segments
    let pa = wg.junctions[seg.junction_a].position,
        pb = wg.junctions[seg.junction_b].position,
        path = isnothing(seg.arc) ?
                 open_polygonal_path([pa, pb]) :
                 seg.arc,
        w = wall(path,
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
    # denom is the 2D cross product of the two direction vectors;
    # it goes to zero exactly when the lines are parallel.
    abs(denom) < parallelism_tolerance() && return nothing
    let dpx = cx(p2) - cx(p1),
        dpy = cy(p2) - cy(p1),
        t = (dpx * cy(d2) - dpy * cx(d2)) / denom
      xy(cx(p1) + cx(d1) * t, cy(p1) + cy(d1) * t)
    end
  end
end

#=
Intersections of a 2D parametric line `p + t·d` with a circle
`center, radius`. Returns a `Vector{Loc}` of 0, 1, or 2 points.
Solves `|p + t·d − c|² = r²` as a quadratic in `t`.

Used by Tier 2 miter math when an arc wall meets a straight wall at
a junction — the offset of the arc is another circle, and the line
wall's offset face is a straight line; their intersection is one of
these points (typically the one closest to the junction position).
=#
"Intersections of a 2D parametric line with a circle. Returns 0, 1, or 2 points."
function line_arc_intersection_2d(p_line::Loc, d_line, center::Loc, radius::Real)
  let dx = cx(d_line), dy = cy(d_line),
      fx = cx(p_line) - cx(center), fy = cy(p_line) - cy(center),
      a = dx*dx + dy*dy,
      b = 2 * (fx*dx + fy*dy),
      c = fx*fx + fy*fy - radius*radius,
      disc = b*b - 4*a*c
    if disc < -parallelism_tolerance()
      Loc[]
    elseif disc < parallelism_tolerance()
      let t = -b / (2*a)
        [xy(cx(p_line) + dx*t, cy(p_line) + dy*t)]
      end
    else
      let sq = sqrt(disc),
          t1 = (-b + sq) / (2*a),
          t2 = (-b - sq) / (2*a)
        [xy(cx(p_line) + dx*t1, cy(p_line) + dy*t1),
         xy(cx(p_line) + dx*t2, cy(p_line) + dy*t2)]
      end
    end
  end
end

#=
Intersections of two 2D circles. Returns a `Vector{Loc}` of 0, 1,
or 2 points. Uses the standard formula: the two intersection points
lie on the radical axis of the circles, a distance `h` above/below
the line joining the centres.

Tier 2 miter math uses this for arc/arc junctions — two arc walls
meeting at a radial; the offset of each is another circle (same
centre, shifted radius by half-thickness), and the miter corner is
one of their intersection points.
=#
"Intersections of two 2D circles. Returns 0, 1, or 2 points."
function arc_arc_intersection_2d(c1::Loc, r1::Real, c2::Loc, r2::Real)
  let dx = cx(c2) - cx(c1), dy = cy(c2) - cy(c1),
      d = sqrt(dx*dx + dy*dy)
    # No intersection: too far apart or one strictly inside the other
    (d > r1 + r2 + parallelism_tolerance() ||
     d < abs(r1 - r2) - parallelism_tolerance() ||
     d < parallelism_tolerance()) && return Loc[]
    let a = (r1*r1 - r2*r2 + d*d) / (2*d),
        h_sq = r1*r1 - a*a,
        h = h_sq < 0 ? 0.0 : sqrt(h_sq),
        mx = cx(c1) + a*dx/d,
        my = cy(c1) + a*dy/d
      h < parallelism_tolerance() ?
        [xy(mx, my)] :
        [xy(mx + h*dy/d, my - h*dx/d),
         xy(mx - h*dy/d, my + h*dx/d)]
    end
  end
end

#=
Pick the intersection closest to `ref` from a list. Used when the
line/circle or circle/circle solver returns two candidates — the
correct wall corner is almost always the one nearest the junction
position.
=#
_nearest(candidates, ref) =
  isempty(candidates) ? nothing :
  candidates[argmin([distance(c, ref) for c in candidates])]

#=
Corner between two wall-segment offset curves meeting at a
junction. Dispatches on the segments' geometry:

  - line ↔ line: parallel-line intersection (existing
    `line_intersection_2d`), as in rectangular buildings.
  - line ↔ arc: line / circle intersection, picking the solution
    closest to the junction position.
  - arc ↔ arc: circle / circle intersection, picking the solution
    closest to the junction position. The two arcs need not share a
    centre — this handles, e.g., two concentric rings on a polar
    plan meeting at a radial partition.

`seg_left` / `seg_right` are the `WallSegment`s flanking the
corner; `dir_left` / `dir_right` are their tangent directions
leaving the junction (as from `segment_direction`); `l_th` /
`r_th` are the relevant half-thicknesses.
=#
function wall_corner(pos::Loc,
                     seg_left::WallSegment, dir_left,
                     seg_right::WallSegment, dir_right,
                     l_th::Real, r_th::Real)
  # Offset-face anchor points: perpendicular offsets from `pos`
  # along each segment's outward-facing normal.
  p_left  = xy(cx(pos) - cy(dir_left)  * l_th, cy(pos) + cx(dir_left)  * l_th)
  p_right = xy(cx(pos) + cy(dir_right) * r_th, cy(pos) - cx(dir_right) * r_th)
  arc_l = seg_left.arc
  arc_r = seg_right.arc
  if isnothing(arc_l) && isnothing(arc_r)
    # Line / line — original behaviour.
    line_intersection_2d(p_left, dir_left, p_right, dir_right)
  elseif isnothing(arc_l) && !isnothing(arc_r)
    # Line / arc: the right-side offset is a circle at arc_r.center
    # with radius arc_r.radius shifted outward by r_th.
    let r = arc_r.radius + r_th
      _nearest(line_arc_intersection_2d(p_left, dir_left, arc_r.center, r), pos)
    end
  elseif !isnothing(arc_l) && isnothing(arc_r)
    let r = arc_l.radius + l_th
      _nearest(line_arc_intersection_2d(p_right, dir_right, arc_l.center, r), pos)
    end
  else
    # Arc / arc: two offset circles.
    let rl = arc_l.radius + l_th,
        rr = arc_r.radius + r_th
      _nearest(arc_arc_intersection_2d(arc_l.center, rl, arc_r.center, rr), pos)
    end
  end
end

const MITER_LIMIT = 10.0

# Compute a miter corner with clamping. When the miter distance exceeds the
# limit, scale the corner toward the limit distance (preserving direction from
# junction) instead of collapsing to the midpoint of the offset lines.
function clamped_miter(pos, p_left, dir_left, p_right, dir_right, l_th, r_th)
  _clamp_corner(pos,
                line_intersection_2d(p_left, dir_left, p_right, dir_right),
                p_left, p_right, l_th, r_th)
end

# Arc-aware miter: dispatches through `wall_corner` so line/arc and
# arc/arc junctions use the right intersection geometry.
function clamped_miter(pos,
                       seg_left::WallSegment, dir_left,
                       seg_right::WallSegment, dir_right,
                       l_th, r_th)
  let corner = wall_corner(pos, seg_left, dir_left, seg_right, dir_right, l_th, r_th),
      p_left = xy(cx(pos) - cy(dir_left)  * l_th, cy(pos) + cx(dir_left)  * l_th),
      p_right = xy(cx(pos) + cy(dir_right) * r_th, cy(pos) - cx(dir_right) * r_th)
    _clamp_corner(pos, corner, p_left, p_right, l_th, r_th)
  end
end

# Clamp a computed corner toward `MITER_LIMIT · max(l_th, r_th)` away
# from `pos`; if the intersection failed (parallel, no solution),
# fall back to the midpoint of the two offset anchor points.
function _clamp_corner(pos, corner, p_left, p_right, l_th, r_th)
  let max_dist = MITER_LIMIT * max(l_th, r_th)
    if isnothing(corner)
      xy((cx(p_left) + cx(p_right)) / 2, (cy(p_left) + cy(p_right)) / 2)
    else
      let d = distance(pos, corner)
        d <= max_dist ? corner :
          let v = corner - pos,
              s = max_dist / d
            xy(cx(pos) + cx(v) * s, cy(pos) + cy(v) * s)
          end
      end
    end
  end
end

# Standard miter corners for valence 2+ junctions without a
# through-pair. Dispatches to the arc-aware `clamped_miter` overload
# so line/arc and arc/arc junctions use the right intersection math.
function miter_corners(wg, j_idx, pos, segs)
  let n = length(segs),
      sorted = sort([(s, segment_direction(wg, s, j_idx)) for s in segs],
                    by=((s, d),) -> atan(cy(d), cx(d))),
      corners = Loc[]
    for k in 1:n
      let dir_curr = sorted[k][2],
          dir_next = sorted[mod1(k + 1, n)][2],
          seg_curr = wg.segments[sorted[k][1]],
          seg_next = wg.segments[sorted[mod1(k + 1, n)][1]],
          l_th = l_thickness(seg_curr.offset, seg_curr.family.thickness),
          r_th = r_thickness(seg_next.offset, seg_next.family.thickness),
          corner = clamped_miter(pos, seg_curr, dir_curr, seg_next, dir_next, l_th, r_th)
        push!(corners, corner)
      end
    end
    # corners[k] is between sorted[k] and sorted[k+1]:
    #   sorted[k]'s left corner, sorted[k+1]'s right corner
    let j_corners = Dict{Int, Tuple{Loc, Loc}}()
      for k in 1:n
        j_corners[sorted[k][1]] = (corners[mod1(k - 1, n)], corners[k])
      end
      j_corners
    end
  end
end

# T-junction corners: through-pair gets valence-2 miter (preserving through-wall
# continuity), abutting segments get face-line projection (no overlap).
function t_junction_corners(wg, j_idx, pos, segs, through)
  j_corners = Dict{Int, Tuple{Loc, Loc}}()
  # 1. Through-pair: valence-2 miter between the two through segments
  let d1 = segment_direction(wg, through[1], j_idx),
      d2 = segment_direction(wg, through[2], j_idx),
      seg1 = wg.segments[through[1]],
      seg2 = wg.segments[through[2]],
      t_sorted = sort([(through[1], d1, seg1), (through[2], d2, seg2)],
                      by=((s, d, seg),) -> atan(cy(d), cx(d))),
      (s_lo, d_lo, seg_lo) = t_sorted[1],
      (s_hi, d_hi, seg_hi) = t_sorted[2],
      # Corner between lo and hi (lo's left side, hi's right side)
      l_th_lo = l_thickness(seg_lo.offset, seg_lo.family.thickness),
      r_th_hi = r_thickness(seg_hi.offset, seg_hi.family.thickness),
      c_lo_hi = clamped_miter(pos, seg_lo, d_lo, seg_hi, d_hi, l_th_lo, r_th_hi),
      # Corner between hi and lo (hi's left side, lo's right side)
      l_th_hi = l_thickness(seg_hi.offset, seg_hi.family.thickness),
      r_th_lo = r_thickness(seg_lo.offset, seg_lo.family.thickness),
      c_hi_lo = clamped_miter(pos, seg_hi, d_hi, seg_lo, d_lo, l_th_hi, r_th_lo)
    # lo gets (right=c_hi_lo, left=c_lo_hi), hi gets (right=c_lo_hi, left=c_hi_lo)
    j_corners[s_lo] = (c_hi_lo, c_lo_hi)
    j_corners[s_hi] = (c_lo_hi, c_hi_lo)
  end
  # 2. Abutting segments: intersect offset lines with through-wall's near face
  let through_dir = segment_direction(wg, through[1], j_idx),
      through_seg = wg.segments[through[1]],
      through_normal = vxy(-cy(through_dir), cx(through_dir)),
      t_l_th = l_thickness(through_seg.offset, through_seg.family.thickness),
      t_r_th = r_thickness(through_seg.offset, through_seg.family.thickness)
    for s in segs
      (s == through[1] || s == through[2]) && continue
      let abt_dir = segment_direction(wg, s, j_idx),
          abt_seg = wg.segments[s],
          abt_l_th = l_thickness(abt_seg.offset, abt_seg.family.thickness),
          abt_r_th = r_thickness(abt_seg.offset, abt_seg.family.thickness),
          side = cx(abt_dir) * cx(through_normal) + cy(abt_dir) * cy(through_normal),
          face_dist = side > 0 ? t_l_th : t_r_th,
          face_sign = side > 0 ? 1 : -1,
          face_point = xy(cx(pos) + cx(through_normal) * face_dist * face_sign,
                         cy(pos) + cy(through_normal) * face_dist * face_sign),
          # Abutting wall's left and right offset points
          left_pt = xy(cx(pos) - cy(abt_dir) * abt_l_th,
                       cy(pos) + cx(abt_dir) * abt_l_th),
          right_pt = xy(cx(pos) + cy(abt_dir) * abt_r_th,
                        cy(pos) - cx(abt_dir) * abt_r_th),
          rc = line_intersection_2d(right_pt, abt_dir, face_point, through_dir),
          lc = line_intersection_2d(left_pt, abt_dir, face_point, through_dir)
        j_corners[s] = (isnothing(rc) ? right_pt : rc,
                        isnothing(lc) ? left_pt : lc)
      end
    end
  end
  j_corners
end

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
    # Check for T-junction (exactly 3 segments with a through-pair)
    through = n == 3 ? find_through_pair_at(wg, segs, j_idx) : nothing
    j_corners = if !isnothing(through)
      t_junction_corners(wg, j_idx, pos, segs, through)
    else
      miter_corners(wg, j_idx, pos, segs)
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

  # Centerline-based positions for opening reveals.
  # At t=0/1 use mitered corners; at intermediate t use centerline + perpendicular.
  pos_a = wg.junctions[seg.junction_a].position
  pos_b = wg.junctions[seg.junction_b].position
  dir = unitized(pos_b - pos_a)
  l_th = l_thickness(seg.offset, fam.thickness)
  r_th = r_thickness(seg.offset, fam.thickness)
  function lr_at(t)
    if t <= 0
      (left_a, right_a)
    elseif t >= 1
      (left_b, right_b)
    else
      let cx_c = cx(pos_a) + t * (cx(pos_b) - cx(pos_a)),
          cy_c = cy(pos_a) + t * (cy(pos_b) - cy(pos_a))
        (xy(cx_c - cy(dir) * l_th, cy_c + cx(dir) * l_th),
         xy(cx_c + cy(dir) * r_th, cy_c - cx(dir) * r_th))
      end
    end
  end

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
    (t1 - t0) < coincidence_tolerance() && continue
    (la0, ra0) = lr_at(t0)
    (la1, ra1) = lr_at(t1)
    # Is this strip inside an opening?
    op_match = nothing
    for (ot0, ot1, op) in op_ranges
      if t0 >= ot0 - coincidence_tolerance() && t1 <= ot1 + coincidence_tolerance()
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
        if z0 > zbot + coincidence_tolerance()
          push!(quads, (vtx(la0, zbot), vtx(la1, zbot), vtx(la1, z0), vtx(la0, z0))); push!(mats, lmat)
          push!(quads, (vtx(ra1, zbot), vtx(ra0, zbot), vtx(ra0, z0), vtx(ra1, z0))); push!(mats, rmat)
        end
        # Above opening
        if z1 < ztop - coincidence_tolerance()
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

# Junction cap polygon: the central area at a miter junction not covered by
# any segment's top/bottom face. Returns the polygon vertices in CCW order,
# or nothing if no cap is needed (valence < 3, or T-junction with through-pair).
function junction_cap_polygon(wg, j_idx, jc)
  j = wg.junctions[j_idx]
  segs = j.segments
  n = length(segs)
  n < 3 && return nothing
  # T-junctions (valence 3 with through-pair): through-wall is continuous, no cap
  if n == 3
    through = find_through_pair_at(wg, segs, j_idx)
    !isnothing(through) && return nothing
  end
  # Collect left corner of each segment in sorted angle order → cap polygon
  sorted = sort([(s, segment_direction(wg, s, j_idx)) for s in segs],
                by=((s, d),) -> atan(cy(d), cx(d)))
  Loc[jc[j_idx][s][2] for (s, _) in sorted]
end

# Generate a mesh for a junction cap polygon (top + bottom faces).
function junction_cap_mesh(wg, j_idx, poly)
  zbot = wg.bottom_level.height
  ztop = wg.top_level.height
  smat = wg.segments[wg.junctions[j_idx].segments[1]].family.side_material
  vertices = Loc[]
  quads = NTuple{4, Int}[]
  mats = Material[]
  vi = Ref(0)
  vtx(p2d, z) = (push!(vertices, xyz(cx(p2d), cy(p2d), z)); vi[] += 1; vi[])
  n = length(poly)
  if n == 4
    # Single quad
    let a = vtx(poly[1], ztop), b = vtx(poly[2], ztop),
        c = vtx(poly[3], ztop), d = vtx(poly[4], ztop)
      push!(quads, (a, b, c, d)); push!(mats, smat)
    end
    let a = vtx(poly[4], zbot), b = vtx(poly[3], zbot),
        c = vtx(poly[2], zbot), d = vtx(poly[1], zbot)
      push!(quads, (a, b, c, d)); push!(mats, smat)
    end
  else
    # Fan triangulation from first vertex (degenerate quads for triangles)
    for k in 2:n-1
      let a = vtx(poly[1], ztop), b = vtx(poly[k], ztop), c = vtx(poly[k+1], ztop)
        push!(quads, (a, b, c, c)); push!(mats, smat)
      end
      let a = vtx(poly[k+1], zbot), b = vtx(poly[k], zbot), c = vtx(poly[1], zbot)
        push!(quads, (a, b, c, c)); push!(mats, smat)
      end
    end
  end
  WallMesh(vertices, quads, mats)
end

# Generate meshes for all wall segments with junction-aware geometry.
resolve_geometry(wg::WallGraph) =
  let jc = compute_junction_corners(wg),
      seg_meshes = [segment_mesh(wg, i, jc) for i in 1:length(wg.segments)],
      cap_meshes = WallMesh[]
    for j in 1:length(wg.junctions)
      let poly = junction_cap_polygon(wg, j, jc)
        !isnothing(poly) && push!(cap_meshes, junction_cap_mesh(wg, j, poly))
      end
    end
    [seg_meshes; cap_meshes]
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
    # Render door/window panels and frames
    for seg in wg.segments
      for op in seg.openings
        append!(refs, render_opening_frame(wg, seg, op, b))
      end
    end
    refs
  end
end

# Render a door/window panel and frame for an opening in a wall segment.
function render_opening_frame(wg, seg, op, b)
  let pos_a = wg.junctions[seg.junction_a].position,
      pos_b = wg.junctions[seg.junction_b].position,
      zbot = wg.bottom_level.height,
      dir = unitized(pos_b - pos_a),
      base_height = zbot + op.sill + 0.0001,
      height = op.family.height - 0.0001,
      start_pos = pos_a + dir * op.distance,
      end_pos = pos_a + dir * (op.distance + op.family.width),
      subpath = translate(open_polygonal_path([start_pos, end_pos]), vz(base_height)),
      l_th = l_thickness(seg.offset, seg.family.thickness),
      r_th = r_thickness(seg.offset, seg.family.thickness),
      thickness = op.family.thickness,
      frame = op.family.frame,
      # Panel: thin wall at the opening (door/window surface)
      panel_refs = b_wall_no_openings(b, subpath, height,
        (l_th - r_th + thickness) / 2,
        (r_th - l_th + thickness) / 2,
        op.family),
      # Frame path: U-shape for doors, closed rectangle for windows
      fp = if op.kind == :door
        foldr(join_paths,
          [open_polygonal_path([path_end(subpath), path_end(subpath) + vz(height)]),
           translate(reverse(subpath), vz(height)),
           open_polygonal_path([path_start(subpath) + vz(height), path_start(subpath)])])
      else
        foldr(join_paths,
          [subpath,
           open_polygonal_path([path_end(subpath), path_end(subpath) + vz(height)]),
           translate(reverse(subpath), vz(height)),
           open_polygonal_path([path_start(subpath) + vz(height), path_start(subpath)])])
      end,
      frame_refs = b_sweep(b, fp, frame.profile, 0, 1,
        material_ref(b, frame.material))
    vcat(collect(panel_refs), collect(frame_refs))
  end
end

