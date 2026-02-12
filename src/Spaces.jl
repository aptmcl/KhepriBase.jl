#####################################################################
# Spaces — Space-first architectural layout system
#
# Architects think in terms of spaces (rooms) and connections (doors,
# windows, arches), not individual walls. This module lets users define
# spaces as closed paths and connections between them, then automatically
# generates walls, doors, windows, and slabs via build().

export Space, SpaceConnection, FloorPlan,
       floor_plan, add_space, add_arch,
       build,
       shared_boundary, exterior_edges, neighbors

#=== Data Structures ===#

# A named bounded area on a floor plan
struct Space
  name::String
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

# A floor plan holds spaces and connections at a given level
mutable struct FloorPlan
  spaces::Vector{Space}
  connections::Vector{SpaceConnection}
  level::Level
  height::Real
  wall_family::WallFamily
  slab_family::SlabFamily
  generate_slabs::Bool
end

#=== Constructor Functions ===#

floor_plan(; level=default_level(),
             height=default_level_to_level_height(),
             wall_family=default_wall_family(),
             slab_family=default_slab_family(),
             generate_slabs=true) =
  FloorPlan(Space[], SpaceConnection[], level, height,
            wall_family, slab_family, generate_slabs)

add_space(plan::FloorPlan, name, boundary) =
  let s = Space(name, boundary)
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

#=== Geometry Helpers ===#

# Extract directed edge segments from vertices of a closed polygon.
# Returns [(v1, v2), (v2, v3), ..., (vn, v1)].
polygon_edges(vertices) =
  let n = length(vertices)
    [(vertices[i], vertices[mod1(i + 1, n)]) for i in 1:n]
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

#=== Edge Classification ===#

# Classify a single directed edge against edges from other spaces.
# Returns list of (p1, p2, kind, space, other_space_or_nothing) tuples
# where kind is :interior (shared) or :exterior.
function classify_edge(a1, a2, space, other_edges, tol)
  edge_len = distance(a1, a2)
  edge_len < tol && return []
  d = unitized(a2 - a1)
  # Collect overlap intervals along this edge
  overlaps = []
  for (b1, b2, other_space) in other_edges
    ov = collinear_overlap(a1, a2, b1, b2, tol)
    if !isnothing(ov)
      push!(overlaps, (ov[1], ov[2], other_space))
    end
  end
  sort!(overlaps, by=first)
  # Split the edge into classified segments
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
          # Emit shared wall only once: from the space with smaller objectid
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
# Returns list of (p_start, p_end) tuples in world coordinates.
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
# Returns list of (p_start, p_end) tuples.
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

# Generate all BIM elements (walls, doors, windows, slabs) from a floor plan.
# Returns (walls, doors, windows, slabs).
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
  # Create walls (one per classified edge segment)
  walls = [wall(open_polygonal_path([p1, p2]),
                bottom_level=plan.level, top_level=top,
                family=plan.wall_family)
           for (p1, p2, _, _, _) in segments]
  # Place doors and windows on their corresponding walls
  doors = []
  windows = []
  for conn in plan.connections
    conn.kind == :arch && continue
    if conn.space_b isa Space
      place_interior_connection!(conn, segments, walls, doors, windows)
    elseif !isnothing(conn.loc)
      place_exterior_connection!(conn, segments, walls, doors, windows)
    else
      error("Exterior connections require a loc parameter")
    end
  end
  # Generate floor slabs
  slabs = plan.generate_slabs ?
    [slab(s.boundary, level=plan.level, family=plan.slab_family)
     for s in plan.spaces] : []
  (walls, doors, windows, slabs)
end

# Place a door or window on the shared interior wall between two spaces.
function place_interior_connection!(conn, segments, walls, doors, windows)
  target = minmax(objectid(conn.space_a), objectid(conn.space_b))
  for (i, seg) in enumerate(segments)
    _, _, kind, sp_a, sp_b = seg
    kind == :interior || continue
    isnothing(sp_b) && continue
    minmax(objectid(sp_a), objectid(sp_b)) == target || continue
    p1, p2 = seg[1], seg[2]
    w = walls[i]
    wall_len = distance(p1, p2)
    opening_width = conn.family.width
    wall_x = if isnothing(conn.loc)
      (wall_len - opening_width) / 2
    else
      let d = unitized(p2 - p1)
        dot(conn.loc - p1, d) - opening_width / 2
      end
    end
    wall_loc = xy(wall_x, 0)
    if conn.kind == :door
      add_door(w, wall_loc, conn.family)
      push!(doors, w.doors[end])
    else
      add_window(w, wall_loc, conn.family)
      push!(windows, w.windows[end])
    end
    return
  end
  error("No shared wall found between '$(conn.space_a.name)' and '$(conn.space_b.name)'")
end

# Place a door or window on the exterior wall closest to conn.loc.
function place_exterior_connection!(conn, segments, walls, doors, windows)
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
  w = walls[best_idx]
  opening_width = conn.family.width
  wall_loc = xy(best_t - opening_width / 2, 0)
  if conn.kind == :door
    add_door(w, wall_loc, conn.family)
    push!(doors, w.doors[end])
  else
    add_window(w, wall_loc, conn.family)
    push!(windows, w.windows[end])
  end
end
