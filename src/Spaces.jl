#####################################################################
# Spaces — Space-first architectural layout system
#
# Architects think in terms of spaces (rooms) and connections (doors,
# windows, arches), not individual walls. This module lets users define
# spaces as closed paths and connections between them, then automatically
# generates walls, doors, windows, and slabs via build().
#
# Inspired by IFC's IfcSpace / IfcRelSpaceBoundary model: build()
# produces a descriptive BuildResult that persists space-to-element
# boundary relationships, enabling introspection and rule validation.

export Space, SpaceConnection, SpaceBoundary, SpaceRule, BuildResult, FloorPlan,
       floor_plan, add_space, add_arch, add_rule,
       build, validate,
       space_area, space_perimeter,
       shared_boundary, exterior_edges, neighbors,
       space_boundaries, space_walls, space_doors, space_windows,
       bounding_spaces, adjacent_spaces,
       min_area_rule, max_area_rule, has_door_rule, has_connection_rule

#=== Data Structures ===#

# A named bounded area on a floor plan (cf. IFC IfcSpace).
# kind classifies the space's function, similar to IfcSpaceTypeEnum.
struct Space
  name::String
  kind::Symbol      # :space, :room, :wc, :kitchen, :corridor, :office, :bedroom, :parking, ...
  boundary::ClosedPath
end

# A connection between two spaces, or between a space and exterior
struct SpaceConnection
  kind::Symbol                    # :door, :window, :arch
  space_a::Space
  space_b::Union{Space, Symbol}   # Space or :exterior
  family::Union{Family, Nothing}
  loc::Union{Loc, Nothing}        # World-space point on the boundary edge
end

# A relationship between a space and a bounding element (cf. IFC IfcRelSpaceBoundary).
# Persists after build() to enable introspection: which elements bound a space,
# which spaces does an element separate, etc.
struct SpaceBoundary
  space::Space
  element               # Wall, Door, Window, or nothing (for arches)
  kind::Symbol          # :physical (wall), :virtual (opening/arch)
  side::Symbol          # :interior, :exterior
  related_space::Union{Space, Nothing}  # space on the other side (cf. 2nd level boundary)
  p1::Loc               # boundary segment start
  p2::Loc               # boundary segment end
end

# A validation rule that can be checked against spaces after build().
struct SpaceRule
  name::String
  check               # (Space, BuildResult) → Union{Nothing, String}
end

# A floor plan holds spaces, connections, and validation rules at a given level.
mutable struct FloorPlan
  spaces::Vector{Space}
  connections::Vector{SpaceConnection}
  level::Level
  height::Real
  wall_family::WallFamily
  slab_family::SlabFamily
  generate_slabs::Bool
  rules::Vector{SpaceRule}
end

# The result of build(): BIM elements plus the descriptive boundary model.
# Supports tuple destructuring: walls, doors, windows, slabs = build(plan)
struct BuildResult
  plan::FloorPlan
  walls::Vector
  doors::Vector
  windows::Vector
  slabs::Vector
  boundaries::Vector{SpaceBoundary}
end

Base.iterate(r::BuildResult, state=1) =
  state == 1 ? (r.walls, 2) :
  state == 2 ? (r.doors, 3) :
  state == 3 ? (r.windows, 4) :
  state == 4 ? (r.slabs, 5) :
  nothing
Base.length(::BuildResult) = 4

Base.show(io::IO, r::BuildResult) =
  print(io, "BuildResult($(length(r.plan.spaces)) spaces, $(length(r.walls)) walls, ",
            "$(length(r.doors)) doors, $(length(r.windows)) windows, ",
            "$(length(r.slabs)) slabs, $(length(r.boundaries)) boundaries)")

#=== Constructor Functions ===#

floor_plan(; level=default_level(),
             height=default_level_to_level_height(),
             wall_family=default_wall_family(),
             slab_family=default_slab_family(),
             generate_slabs=true,
             rules=SpaceRule[]) =
  FloorPlan(Space[], SpaceConnection[], level, height,
            wall_family, slab_family, generate_slabs, rules)

add_space(plan::FloorPlan, name, boundary; kind=:space) =
  let s = Space(name, kind, boundary)
    push!(plan.spaces, s)
    s
  end

add_door(plan::FloorPlan, space_a::Space, space_b::Union{Space, Symbol};
         family=default_door_family(), loc=nothing) =
  let c = SpaceConnection(:door, space_a, space_b, family, loc)
    push!(plan.connections, c)
    c
  end

add_window(plan::FloorPlan, space_a::Space, space_b::Union{Space, Symbol};
           family=default_window_family(), loc=nothing) =
  let c = SpaceConnection(:window, space_a, space_b, family, loc)
    push!(plan.connections, c)
    c
  end

add_arch(plan::FloorPlan, space_a::Space, space_b::Space) =
  let c = SpaceConnection(:arch, space_a, space_b, nothing, nothing)
    push!(plan.connections, c)
    c
  end

add_rule(plan::FloorPlan, rule::SpaceRule) =
  let _ = push!(plan.rules, rule)
    rule
  end

#=== Geometry Helpers ===#

# Extract directed edge segments from vertices of a closed polygon.
polygon_edges(vertices) =
  let n = length(vertices)
    [(vertices[i], vertices[mod1(i + 1, n)]) for i in 1:n]
  end

# Polygon area via the shoelace formula (2D, using world xy coordinates).
polygon_area(vertices) =
  let n = length(vertices),
      ws = [in_world(v) for v in vertices]
    abs(sum(cx(ws[i]) * cy(ws[mod1(i + 1, n)]) -
            cx(ws[mod1(i + 1, n)]) * cy(ws[i])
            for i in 1:n)) / 2
  end

# Find collinear overlap between directed segments (a1→a2) and (b1→b2).
# Returns (t_start, t_end) parametric on (a1→a2), or nothing.
collinear_overlap(a1, a2, b1, b2, tol=collinearity_tolerance()) =
  let edge_len = distance(a1, a2)
    edge_len < tol ? nothing :
    !(collinear_points(a1, b1, a2, tol) &&
      collinear_points(a1, b2, a2, tol)) ? nothing :
    let d = unitized(a2 - a1),
        tb1 = dot(b1 - a1, d),
        tb2 = dot(b2 - a1, d),
        ov_start = max(0.0, min(tb1, tb2)),
        ov_end = min(edge_len, max(tb1, tb2))
      (ov_end - ov_start) > tol ? (ov_start, ov_end) : nothing
    end
  end

#=== Computed Properties ===#

space_area(space::Space) = polygon_area(path_vertices(space.boundary))

space_perimeter(space::Space) =
  let vs = path_vertices(space.boundary),
      n = length(vs)
    sum(distance(vs[i], vs[mod1(i + 1, n)]) for i in 1:n)
  end

#=== Edge Classification ===#

# Classify a single directed edge against edges from other spaces.
# Returns list of (p1, p2, kind, space, other_space_or_nothing) tuples
# where kind is :interior (shared) or :exterior.
function classify_edge(a1, a2, space, other_edges, tol)
  edge_len = distance(a1, a2)
  edge_len < tol && return []
  d = unitized(a2 - a1)
  overlaps = []
  for (b1, b2, other_space) in other_edges
    ov = collinear_overlap(a1, a2, b1, b2, tol)
    if !isnothing(ov)
      push!(overlaps, (ov[1], ov[2], other_space))
    end
  end
  sort!(overlaps, by=first)
  segments = []
  cursor = 0.0
  for (t_start, t_end, other_space) in overlaps
    if t_start > cursor + tol
      push!(segments, (a1 + d * cursor, a1 + d * t_start, :exterior, space, nothing))
    end
    push!(segments, (a1 + d * t_start, a1 + d * t_end, :interior, space, other_space))
    cursor = max(cursor, t_end)
  end
  if cursor < edge_len - tol
    push!(segments, (a1 + d * cursor, a2, :exterior, space, nothing))
  end
  isempty(segments) ? [(a1, a2, :exterior, space, nothing)] : segments
end

# Classify all edges in the plan.
# Interior (shared) walls are deduplicated — only one wall per shared boundary.
function classify_all_edges(plan, tol)
  space_edges = Dict(s => polygon_edges(path_vertices(s.boundary)) for s in plan.spaces)
  segments = []
  for space in plan.spaces
    other_edges = [(b1, b2, s)
                   for s in plan.spaces if s !== space
                   for (b1, b2) in space_edges[s]]
    for (a1, a2) in space_edges[space]
      for seg in classify_edge(a1, a2, space, other_edges, tol)
        _, _, kind, sp_a, sp_b = seg
        if kind == :interior
          objectid(sp_a) < objectid(sp_b) && push!(segments, seg)
        else
          push!(segments, seg)
        end
      end
    end
  end
  segments
end

#=== Query Helpers ===#

# Find shared boundary segments between two spaces.
shared_boundary(space_a::Space, space_b::Space, tol=collinearity_tolerance()) =
  let edges_a = polygon_edges(path_vertices(space_a.boundary)),
      edges_b = polygon_edges(path_vertices(space_b.boundary)),
      shared = Tuple{Loc, Loc}[]
    for (a1, a2) in edges_a, (b1, b2) in edges_b
      let ov = collinear_overlap(a1, a2, b1, b2, tol)
        if !isnothing(ov)
          let d = unitized(a2 - a1)
            push!(shared, (a1 + d * ov[1], a1 + d * ov[2]))
          end
        end
      end
    end
    shared
  end

# Find all exterior edges of a space within a plan.
exterior_edges(plan::FloorPlan, space::Space, tol=collinearity_tolerance()) =
  let edges = polygon_edges(path_vertices(space.boundary)),
      other_edges = [(b1, b2, s)
                     for s in plan.spaces if s !== space
                     for (b1, b2) in polygon_edges(path_vertices(s.boundary))]
    [(p1, p2)
     for (a1, a2) in edges
     for (p1, p2, kind, _, _) in classify_edge(a1, a2, space, other_edges, tol)
     if kind == :exterior]
  end

# Find all spaces that share a boundary with the given space.
neighbors(plan::FloorPlan, space::Space) =
  [s for s in plan.spaces
   if s !== space && !isempty(shared_boundary(space, s))]

#=== Builder ===#

# Generate all BIM elements from a floor plan and return a descriptive BuildResult.
# The result persists space-to-element boundary relationships for introspection.
function build(plan::FloorPlan)
  tol = collinearity_tolerance()
  top = upper_level(plan.level, plan.height)
  all_segments = classify_all_edges(plan, tol)
  # Arch connections suppress wall generation on their shared boundary
  arch_pairs = Set(
    minmax(objectid(c.space_a), objectid(c.space_b))
    for c in plan.connections
    if c.kind == :arch && c.space_b isa Space)
  segments = filter(all_segments) do seg
    _, _, kind, sp_a, sp_b = seg
    !(kind == :interior && !isnothing(sp_b) &&
      minmax(objectid(sp_a), objectid(sp_b)) in arch_pairs)
  end
  # Build WallGraph from classified edge segments
  wg = wall_graph(level=plan.level, height=plan.height)
  edge_to_seg = Int[]  # maps edge index → graph segment index
  for (p1, p2, _, _, _) in segments
    let j1 = find_or_create_junction!(wg, p1, tol),
        j2 = find_or_create_junction!(wg, p2, tol)
      push!(edge_to_seg, segment!(wg, j1, j2, family=plan.wall_family))
    end
  end
  # Resolve chains and create walls (without openings)
  chains = resolve(wg)
  walls = []
  seg_to_wall = Dict{Int, Int}()    # graph_seg → wall index
  seg_offset = Dict{Int, Real}()    # graph_seg → cumulative distance in merged wall
  seg_forward = Dict{Int, Bool}()   # graph_seg → same orientation as chain?
  for chain in chains
    let w = wall(chain.path,
                 bottom_level=wg.bottom_level, top_level=wg.top_level,
                 family=chain.family, offset=chain.offset),
        wall_idx = length(walls) + 1,
        junctions = chain_junctions(wg, chain.source_segments),
        cumulative = 0.0
      push!(walls, w)
      for (k, s) in enumerate(chain.source_segments)
        seg_to_wall[s] = wall_idx
        seg_obj = wg.segments[s]
        # Check if segment is forward (junction_a matches chain direction)
        fwd = seg_obj.junction_a == junctions[k]
        seg_forward[s] = fwd
        seg_offset[s] = cumulative
        cumulative += segment_length(wg, s)
      end
    end
  end
  # Create wall boundaries
  boundaries = SpaceBoundary[]
  for (i, (p1, p2, kind, sp_a, sp_b)) in enumerate(segments)
    let w = walls[seg_to_wall[edge_to_seg[i]]]
      push!(boundaries, SpaceBoundary(sp_a, w, :physical,
            kind == :interior ? :interior : :exterior, sp_b, p1, p2))
      if kind == :interior && !isnothing(sp_b)
        push!(boundaries, SpaceBoundary(sp_b, w, :physical, :interior, sp_a, p1, p2))
      end
    end
  end
  # Record arch boundaries (virtual, no element)
  for conn in plan.connections
    if conn.kind == :arch && conn.space_b isa Space
      for (p1, p2) in shared_boundary(conn.space_a, conn.space_b)
        push!(boundaries, SpaceBoundary(conn.space_a, nothing, :virtual, :interior, conn.space_b, p1, p2))
        push!(boundaries, SpaceBoundary(conn.space_b, nothing, :virtual, :interior, conn.space_a, p1, p2))
      end
    end
  end
  # Place doors and windows on merged walls, recording their boundaries
  doors = []
  windows = []
  for conn in plan.connections
    conn.kind == :arch && continue
    if conn.space_b isa Space
      place_interior_connection!(conn, segments, edge_to_seg, walls,
                                 seg_to_wall, seg_offset, seg_forward, wg,
                                 doors, windows, boundaries)
    elseif !isnothing(conn.loc)
      place_exterior_connection!(conn, segments, edge_to_seg, walls,
                                 seg_to_wall, seg_offset, seg_forward, wg,
                                 doors, windows, boundaries)
    else
      error("Exterior connections require a loc parameter")
    end
  end
  # Generate floor slabs
  slabs = plan.generate_slabs ?
    [slab(s.boundary, level=plan.level, family=plan.slab_family)
     for s in plan.spaces] : []
  BuildResult(plan, walls, doors, windows, slabs, boundaries)
end

# Compute the position along a merged wall path for an opening on a given edge.
# local_x is the distance from edge start (p1) to the opening's left edge.
# opening_width is needed when the segment is reversed in the chain.
function edge_to_wall_x(local_x, opening_width, edge_idx, edge_to_seg, seg_offset, seg_forward, wg)
  let gs = edge_to_seg[edge_idx],
      offset = seg_offset[gs],
      fwd = seg_forward[gs]
    fwd ? offset + local_x :
      let seg_len = segment_length(wg, gs)
        offset + seg_len - local_x - opening_width
      end
  end
end

function place_interior_connection!(conn, segments, edge_to_seg, walls,
                                    seg_to_wall, seg_offset, seg_forward, wg,
                                    doors, windows, boundaries)
  target = minmax(objectid(conn.space_a), objectid(conn.space_b))
  for (i, seg) in enumerate(segments)
    _, _, kind, sp_a, sp_b = seg
    kind == :interior || continue
    isnothing(sp_b) && continue
    minmax(objectid(sp_a), objectid(sp_b)) == target || continue
    p1, p2 = seg[1], seg[2]
    w = walls[seg_to_wall[edge_to_seg[i]]]
    wall_len = distance(p1, p2)
    opening_width = conn.family.width
    local_x = if isnothing(conn.loc)
      (wall_len - opening_width) / 2
    else
      let d = unitized(p2 - p1)
        dot(conn.loc - p1, d) - opening_width / 2
      end
    end
    wall_x = edge_to_wall_x(local_x, opening_width, i, edge_to_seg, seg_offset, seg_forward, wg)
    d = unitized(p2 - p1)
    op_start = p1 + d * local_x
    op_end = p1 + d * (local_x + opening_width)
    wall_loc = xy(wall_x, 0)
    if conn.kind == :door
      add_door(w, wall_loc, conn.family)
      push!(doors, w.doors[end])
      let el = w.doors[end]
        push!(boundaries, SpaceBoundary(conn.space_a, el, :virtual, :interior, conn.space_b, op_start, op_end))
        push!(boundaries, SpaceBoundary(conn.space_b, el, :virtual, :interior, conn.space_a, op_start, op_end))
      end
    else
      add_window(w, wall_loc, conn.family)
      push!(windows, w.windows[end])
      let el = w.windows[end]
        push!(boundaries, SpaceBoundary(conn.space_a, el, :virtual, :interior, conn.space_b, op_start, op_end))
        push!(boundaries, SpaceBoundary(conn.space_b, el, :virtual, :interior, conn.space_a, op_start, op_end))
      end
    end
    return
  end
  error("No shared wall found between '$(conn.space_a.name)' and '$(conn.space_b.name)'")
end

function place_exterior_connection!(conn, segments, edge_to_seg, walls,
                                    seg_to_wall, seg_offset, seg_forward, wg,
                                    doors, windows, boundaries)
  best_idx = 0
  best_t = 0.0
  best_dist = Inf
  for (i, seg) in enumerate(segments)
    _, _, kind, sp_a, _ = seg
    kind == :exterior || continue
    sp_a !== conn.space_a && continue
    p1, p2 = seg[1], seg[2]
    d = unitized(p2 - p1)
    wall_len = distance(p1, p2)
    t = clamp(dot(conn.loc - p1, d), 0.0, wall_len)
    dist = distance(conn.loc, p1 + d * t)
    if dist < best_dist
      best_dist = dist
      best_idx = i
      best_t = t
    end
  end
  best_idx == 0 &&
    error("No exterior wall found for '$(conn.space_a.name)' near $(conn.loc)")
  p1, p2 = segments[best_idx][1], segments[best_idx][2]
  d = unitized(p2 - p1)
  w = walls[seg_to_wall[edge_to_seg[best_idx]]]
  opening_width = conn.family.width
  local_x = best_t - opening_width / 2
  wall_x = edge_to_wall_x(local_x, opening_width, best_idx, edge_to_seg, seg_offset, seg_forward, wg)
  op_start = p1 + d * local_x
  op_end = p1 + d * (local_x + opening_width)
  wall_loc = xy(wall_x, 0)
  if conn.kind == :door
    add_door(w, wall_loc, conn.family)
    push!(doors, w.doors[end])
    let el = w.doors[end]
      push!(boundaries, SpaceBoundary(conn.space_a, el, :virtual, :exterior, nothing, op_start, op_end))
    end
  else
    add_window(w, wall_loc, conn.family)
    push!(windows, w.windows[end])
    let el = w.windows[end]
      push!(boundaries, SpaceBoundary(conn.space_a, el, :virtual, :exterior, nothing, op_start, op_end))
    end
  end
end

#=== BuildResult Queries ===#

# All boundaries for a given space (cf. IfcSpace.BoundedBy)
space_boundaries(result::BuildResult, space::Space) =
  [b for b in result.boundaries if b.space === space]

# Walls bounding a space
space_walls(result::BuildResult, space::Space) =
  unique([b.element for b in result.boundaries
          if b.space === space && b.kind == :physical])

# Doors accessible from a space
space_doors(result::BuildResult, space::Space) =
  unique([b.element for b in result.boundaries
          if b.space === space && !isnothing(b.element) && b.element isa Door])

# Windows on a space
space_windows(result::BuildResult, space::Space) =
  unique([b.element for b in result.boundaries
          if b.space === space && !isnothing(b.element) && b.element isa Window])

# Spaces bounded by a given element
bounding_spaces(result::BuildResult, element) =
  unique([b.space for b in result.boundaries if b.element === element])

# Spaces adjacent to a given space (connected through any boundary)
adjacent_spaces(result::BuildResult, space::Space) =
  unique(filter(!isnothing,
    [b.related_space for b in result.boundaries if b.space === space]))

#=== Predefined Rules ===#

# Minimum area for all spaces
min_area_rule(area::Real) = SpaceRule(
  "Minimum area: $(area)m\u00b2",
  (space, result) ->
    space_area(space) < area ?
      "$(space.name): area $(round(space_area(space), digits=2))m\u00b2 < minimum $(area)m\u00b2" :
      nothing)

# Minimum area for spaces of a given kind
min_area_rule(kind::Symbol, area::Real) = SpaceRule(
  "Minimum $(kind) area: $(area)m\u00b2",
  (space, result) ->
    space.kind == kind && space_area(space) < area ?
      "$(space.name) ($(kind)): area $(round(space_area(space), digits=2))m\u00b2 < minimum $(area)m\u00b2" :
      nothing)

# Maximum area for spaces of a given kind
max_area_rule(kind::Symbol, area::Real) = SpaceRule(
  "Maximum $(kind) area: $(area)m\u00b2",
  (space, result) ->
    space.kind == kind && space_area(space) > area ?
      "$(space.name) ($(kind)): area $(round(space_area(space), digits=2))m\u00b2 > maximum $(area)m\u00b2" :
      nothing)

# Every space of a given kind must have at least one door
has_door_rule(kind::Symbol) = SpaceRule(
  "$(kind) must have a door",
  (space, result) ->
    space.kind == kind && isempty(space_doors(result, space)) ?
      "$(space.name) ($(kind)): has no door" :
      nothing)

# Every space must have at least one door
has_door_rule() = SpaceRule(
  "Every space must have a door",
  (space, result) ->
    isempty(space_doors(result, space)) ?
      "$(space.name): has no door" :
      nothing)

# Every space must have at least one connection (door, window, or arch)
has_connection_rule() = SpaceRule(
  "Every space must have a connection",
  (space, result) ->
    let conns = [c for c in result.plan.connections
                 if c.space_a === space || c.space_b === space]
      isempty(conns) ?
        "$(space.name): has no connections" :
        nothing
    end)

#=== Validation ===#

# Validate all rules on the plan. Returns a list of violation messages (empty = all pass).
validate(result::BuildResult) = validate(result, result.plan.rules)

validate(result::BuildResult, rules) =
  let violations = String[]
    for rule in rules, space in result.plan.spaces
      let msg = rule.check(space, result)
        !isnothing(msg) && push!(violations, msg)
      end
    end
    violations
  end
