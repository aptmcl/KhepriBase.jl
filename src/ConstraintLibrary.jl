#####################################################################
# Constraint Library — context-polymorphic constructors
#
# A `Constraint`'s check closure takes any context. In practice the
# two contexts it runs against are:
#
#   - `Layout`       — output of `layout(desc::SpaceDesc)` or of an
#                      imperative `floor_plan`/`add_storey!`/add_space
#                      sequence. Multi-storey, boundaries-only.
#   - `BuildResult`  — output of `build(storey)` (single-storey) or an
#                      entry in `build(layout)` (Vector). Carries walls,
#                      doors, windows, slabs, and `SpaceBoundary`
#                      records.
#
# The helpers `_ctx_spaces`, `_ctx_adjacencies`, `_ctx_storeys` below
# bridge both so each constraint constructor is written once and
# dispatches at validation time. A handful of constraints (`has_door`,
# `has_connection`) inspect door/connection information that only a
# `BuildResult` carries — those stay BuildResult-only by virtue of the
# accessor used inside the check.

export min_area, max_area, min_dimension, max_aspect_ratio, floor_height_range,
       min_corridor_width, area_ratio,
       must_adjoin, must_not_adjoin, vertical_alignment,
       all_reachable, max_dead_end, min_egress_routes,
       min_exterior_exposure, preferred_orientation, facade_ratio,
       has_door, has_connection

#=== Context Accessors ===#

# Spaces of the context — iterable of `Space`, regardless of whether
# the context is a Layout (multi-storey) or a BuildResult (single).
_ctx_spaces(l::Layout) = spaces(l)
_ctx_spaces(r::BuildResult) = r.storey.spaces
_ctx_spaces(rs::AbstractVector{BuildResult}) =
  Iterators.flatten(r.storey.spaces for r in rs)

# Storeys of the context — a short iterable. BuildResult wraps its
# single storey into a one-tuple so callers can use the same shape.
_ctx_storeys(l::Layout) = l.storeys
_ctx_storeys(r::BuildResult) = (r.storey,)
_ctx_storeys(rs::AbstractVector{BuildResult}) = (r.storey for r in rs)

# AdjacencyRelations in the context. Both Layout and BuildResult go
# through `_storey_adjacencies` so the two paths share the same
# classify_all_edges code.
_ctx_adjacencies(l::Layout) = adjacencies(l)
_ctx_adjacencies(r::BuildResult) =
  _storey_adjacencies(r.storey, storey_z(r.storey))
_ctx_adjacencies(rs::AbstractVector{BuildResult}) =
  reduce(vcat, (_storey_adjacencies(r.storey, storey_z(r.storey)) for r in rs);
         init=AdjacencyRelation[])

#=== Dimensional ===#

"""
    min_area(kind, sqm; severity=HARD)

Every space with the given `kind` must have floor area ≥ `sqm` m².
"""
min_area(kind::Symbol, sqm::Real; severity=HARD) = Constraint(
  "min_area_$(kind)_$(sqm)", severity, AREA_PROPORTION,
  ctx -> [
    Violation("min_area_$(kind)", severity, AREA_PROPORTION,
      space_name(sp),
      "$(space_name(sp)): area $(round(a; digits=2))m² < $(sqm)m²",
      a, Float64(sqm))
    for sp in _ctx_spaces(ctx)
    if sp.kind == kind
    for a in (space_area(sp),)
    if a < sqm])

"""
    max_area(kind, sqm; severity=HARD)

Every space with the given `kind` must have floor area ≤ `sqm` m².
"""
max_area(kind::Symbol, sqm::Real; severity=HARD) = Constraint(
  "max_area_$(kind)_$(sqm)", severity, AREA_PROPORTION,
  ctx -> [
    Violation("max_area_$(kind)", severity, AREA_PROPORTION,
      space_name(sp),
      "$(space_name(sp)): area $(round(a; digits=2))m² > $(sqm)m²",
      a, Float64(sqm))
    for sp in _ctx_spaces(ctx)
    if sp.kind == kind
    for a in (space_area(sp),)
    if a > sqm])

"""
    min_dimension(kind, meters; severity=HARD)

Every space of the given `kind` must have a minimum side length of
`meters` (i.e. `min(width, depth) ≥ meters`).
"""
min_dimension(kind::Symbol, meters::Real; severity=HARD) = Constraint(
  "min_dim_$(kind)_$(meters)", severity, DIMENSIONAL,
  ctx -> [
    Violation("min_dim_$(kind)", severity, DIMENSIONAL,
      space_name(sp),
      "$(space_name(sp)): min dim $(d)m < $(meters)m",
      d, Float64(meters))
    for sp in _ctx_spaces(ctx)
    if sp.kind == kind
    for d in (min(space_width(sp), space_depth(sp)),)
    if d < meters])

"""
    max_aspect_ratio(kind, ratio; severity=SOFT)

Limit the bbox aspect ratio of spaces of the given `kind` to at most
`ratio`. Useful against pathological long-thin rooms that otherwise
pass `min_area` / `min_dimension` checks.
"""
max_aspect_ratio(kind::Symbol, ratio::Real; severity=SOFT) = Constraint(
  "max_aspect_$(kind)_$(ratio)", severity, DIMENSIONAL,
  ctx -> [
    Violation("max_aspect_$(kind)", severity, DIMENSIONAL,
      space_name(sp),
      "$(space_name(sp)): aspect ratio $(ar) > $(ratio)",
      ar, Float64(ratio))
    for sp in _ctx_spaces(ctx)
    if sp.kind == kind
    for ar in (let w = space_width(sp), d = space_depth(sp)
                 max(w / d, d / w)
               end,)
    if ar > ratio])

"""
    floor_height_range(min_h, max_h; severity=HARD)

Every storey's height must fall within `[min_h, max_h]` metres.
"""
floor_height_range(min_h::Real, max_h::Real; severity=HARD) = Constraint(
  "floor_height_$(min_h)_$(max_h)", severity, DIMENSIONAL,
  ctx -> [
    Violation("floor_height", severity, DIMENSIONAL,
      "Level $(i)",
      "height $(st.height)m outside [$(min_h), $(max_h)]",
      Float64(st.height),
      st.height < min_h ? Float64(min_h) : Float64(max_h))
    for (i, st) in enumerate(_ctx_storeys(ctx))
    if st.height < min_h || st.height > max_h])

"""
    min_corridor_width(meters; severity=HARD)

Shorthand for `min_dimension(:corridor, meters; severity)`.
"""
min_corridor_width(meters::Real; severity=HARD) =
  min_dimension(:corridor, meters; severity)

"""
    area_ratio(kind_a, kind_b, min_ratio; severity=SOFT)

Total floor area of spaces with `kind_a` must be at least
`min_ratio` times the total floor area of spaces with `kind_b`.
Classic use: "circulation area ≥ 15% of usable office area".
"""
area_ratio(kind_a::Symbol, kind_b::Symbol, min_ratio::Real; severity=SOFT) =
  Constraint(
    "area_ratio_$(kind_a)_$(kind_b)_$(min_ratio)", severity, AREA_PROPORTION,
    ctx -> let
      area_a = sum((space_area(sp) for sp in _ctx_spaces(ctx) if sp.kind == kind_a); init=0.0)
      area_b = sum((space_area(sp) for sp in _ctx_spaces(ctx) if sp.kind == kind_b); init=0.0)
      ratio = area_b > 0 ? area_a / area_b : 0.0
      ratio >= min_ratio ? Violation[] : [Violation(
        "area_ratio_$(kind_a)_$(kind_b)", severity, AREA_PROPORTION,
        "$(kind_a)/$(kind_b)",
        "$(kind_a) area / $(kind_b) area = $(round(ratio; digits=3)) < $(min_ratio)",
        ratio, Float64(min_ratio))]
    end)

#=== Adjacency ===#

"""
    must_adjoin(kind_a, kind_b; severity=HARD)

Every space of `kind_a` must share a boundary with at least one space
of `kind_b`.
"""
must_adjoin(kind_a::Symbol, kind_b::Symbol; severity=HARD) = Constraint(
  "must_adjoin_$(kind_a)_$(kind_b)", severity, ADJACENCY,
  ctx -> let
    out = Violation[]
    adjs = _ctx_adjacencies(ctx)
    spaces_a = [sp for sp in _ctx_spaces(ctx) if sp.kind == kind_a]
    b_ids = Set(sp.id for sp in _ctx_spaces(ctx) if sp.kind == kind_b)
    for sa in spaces_a
      has_adj = any(adjs) do adj
        (adj.space_a == sa.id && adj.space_b in b_ids) ||
        (!isnothing(adj.space_b) && adj.space_b == sa.id && adj.space_a in b_ids)
      end
      has_adj || push!(out, Violation(
        "must_adjoin", severity, ADJACENCY,
        space_name(sa),
        "$(space_name(sa)) ($(kind_a)) not adjacent to any $(kind_b)",
        0.0, 1.0))
    end
    out
  end)

"""
    must_not_adjoin(kind_a, kind_b; severity=HARD)

No space of `kind_a` may share a boundary with any space of `kind_b`.
"""
must_not_adjoin(kind_a::Symbol, kind_b::Symbol; severity=HARD) = Constraint(
  "must_not_adjoin_$(kind_a)_$(kind_b)", severity, ADJACENCY,
  ctx -> let
    out = Violation[]
    by_id = Dict(sp.id => sp for sp in _ctx_spaces(ctx))
    for adj in _ctx_adjacencies(ctx)
      isnothing(adj.space_b) && continue
      sa = get(by_id, adj.space_a, nothing)
      sb = get(by_id, adj.space_b, nothing)
      (isnothing(sa) || isnothing(sb)) && continue
      if (sa.kind == kind_a && sb.kind == kind_b) ||
         (sa.kind == kind_b && sb.kind == kind_a)
        push!(out, Violation(
          "must_not_adjoin", severity, ADJACENCY,
          "$(space_name(sa))-$(space_name(sb))",
          "$(kind_a) '$(space_name(sa))' adjacent to $(kind_b) '$(space_name(sb))'",
          1.0, 0.0))
      end
    end
    out
  end)

"""
    vertical_alignment(kind; severity=HARD)

Spaces of the given `kind` on consecutive storeys must overlap in
plan. Used to enforce alignment of shafts (bathrooms, stairs) across
floors for structural or mechanical continuity. Only meaningful when
the context has two or more storeys — `BuildResult` (single-storey)
never produces violations.
"""
vertical_alignment(kind::Symbol; severity=HARD) = Constraint(
  "vertical_align_$(kind)", severity, ADJACENCY,
  ctx -> let
    out = Violation[]
    storeys = collect(_ctx_storeys(ctx))
    length(storeys) < 2 && return out
    for i in 1:(length(storeys) - 1)
      lower = [sp for sp in storeys[i].spaces if sp.kind == kind]
      upper = [sp for sp in storeys[i + 1].spaces if sp.kind == kind]
      for sl in lower
        (slx, sly) = space_origin(sl)
        slw, sld = space_width(sl), space_depth(sl)
        aligned = any(upper) do su
          (sux, suy) = space_origin(su)
          suw, sud = space_width(su), space_depth(su)
          !(sux + suw < slx + 1e-6 ||
            slx + slw < sux + 1e-6 ||
            suy + sud < sly + 1e-6 ||
            sly + sld < suy + 1e-6)
        end
        aligned || push!(out, Violation(
          "vertical_align", severity, ADJACENCY,
          space_name(sl),
          "$(space_name(sl)) on level $(i) has no aligned $(kind) above",
          0.0, 1.0))
      end
    end
    out
  end)

#=== Circulation ===#

"""
    all_reachable(; severity=HARD)

Every space on each storey must be reachable from an entrance or
corridor via interior adjacencies (BFS connectivity check). Spaces of
kind `:entrance` or `:corridor` are seed points; if none exist on a
storey, the first space is used.
"""
all_reachable(; severity=HARD) = Constraint(
  "all_reachable", severity, CIRCULATION,
  ctx -> let
    out = Violation[]
    adjs = _ctx_adjacencies(ctx)
    for storey in _ctx_storeys(ctx)
      isempty(storey.spaces) && continue
      level_spaces = Set(sp.id for sp in storey.spaces)

      graph = Dict{Symbol, Set{Symbol}}()
      for sid in level_spaces
        graph[sid] = Set{Symbol}()
      end
      for adj in adjs
        isnothing(adj.space_b) && continue
        adj.space_a in level_spaces && adj.space_b in level_spaces || continue
        push!(graph[adj.space_a], adj.space_b)
        push!(graph[adj.space_b], adj.space_a)
      end

      starts = [sp.id for sp in storey.spaces
                if sp.kind in (:entrance, :corridor)]
      isempty(starts) && (starts = [first(storey.spaces).id])

      visited = Set{Symbol}()
      queue = Symbol[starts...]
      while !isempty(queue)
        s = popfirst!(queue)
        s in visited && continue
        push!(visited, s)
        for n in get(graph, s, Set{Symbol}())
          n in visited || push!(queue, n)
        end
      end

      for sid in level_spaces
        sid in visited && continue
        push!(out, Violation(
          "all_reachable", severity, CIRCULATION,
          String(sid), "$(sid) not reachable from entrance/corridor",
          0.0, 1.0))
      end
    end
    out
  end)

"""
    max_dead_end(kind, max_length; severity=HARD)

Flag any space of the given `kind` (typically `:corridor`) that
terminates in a dead end — fewer than two interior neighbours — and
whose primary dimension exceeds `max_length`. Models fire-code
dead-end corridor limits as a conservative single-corridor check.
"""
max_dead_end(kind::Symbol, max_length::Real; severity=HARD) = Constraint(
  "max_dead_end_$(kind)_$(max_length)", severity, CIRCULATION,
  ctx -> let
    out = Violation[]
    adjs = _ctx_adjacencies(ctx)
    for sp in _ctx_spaces(ctx)
      sp.kind == kind || continue
      nbrs = Set{Symbol}()
      for adj in adjs
        isnothing(adj.space_b) && continue
        adj.space_a == sp.id && push!(nbrs, adj.space_b)
        adj.space_b == sp.id && push!(nbrs, adj.space_a)
      end
      length(nbrs) >= 2 && continue
      len = max(space_width(sp), space_depth(sp))
      len <= max_length && continue
      push!(out, Violation(
        "max_dead_end_$(kind)", severity, CIRCULATION,
        space_name(sp),
        "$(space_name(sp)): dead-end length $(len)m > $(max_length)m",
        len, Float64(max_length)))
    end
    out
  end)

"""
    min_egress_routes(kind, n; severity=HARD)

Every space of the given `kind` must have at least `n` distinct
interior neighbour spaces. A conservative per-room degree check that
approximates fire-code egress — not a full k-disjoint-path analysis.
"""
min_egress_routes(kind::Symbol, n::Integer; severity=HARD) = Constraint(
  "min_egress_$(kind)_$(n)", severity, CIRCULATION,
  ctx -> let
    out = Violation[]
    adjs = _ctx_adjacencies(ctx)
    for sp in _ctx_spaces(ctx)
      sp.kind == kind || continue
      nbrs = Set{Symbol}()
      for adj in adjs
        isnothing(adj.space_b) && continue
        adj.space_a == sp.id && push!(nbrs, adj.space_b)
        adj.space_b == sp.id && push!(nbrs, adj.space_a)
      end
      length(nbrs) >= n && continue
      push!(out, Violation(
        "min_egress_$(kind)", severity, CIRCULATION,
        space_name(sp),
        "$(space_name(sp)): $(length(nbrs)) egress route(s), need $(n)",
        Float64(length(nbrs)), Float64(n)))
    end
    out
  end)

"""
    has_door(; severity=HARD)
    has_door(kind; severity=HARD)

Every space (or every space of the given `kind`) must carry at least
one door. Only meaningful on a `BuildResult` — `Layout` doesn't know
about physical doors until `build(layout)` runs.
"""
has_door(; severity=HARD) = Constraint(
  "has_door", severity, CIRCULATION,
  r::BuildResult -> [
    Violation("has_door", severity, CIRCULATION,
      space_name(sp), "$(space_name(sp)): has no door", 0.0, 1.0)
    for sp in r.storey.spaces
    if isempty(space_doors(r, sp))])

has_door(kind::Symbol; severity=HARD) = Constraint(
  "has_door_$(kind)", severity, CIRCULATION,
  r::BuildResult -> [
    Violation("has_door_$(kind)", severity, CIRCULATION,
      space_name(sp), "$(space_name(sp)) ($(kind)): has no door", 0.0, 1.0)
    for sp in r.storey.spaces
    if sp.kind == kind && isempty(space_doors(r, sp))])

"""
    has_connection(; severity=HARD)

Every space must be connected to at least one neighbour (via door,
window, or arch). BuildResult-only; inspects `storey.connections`.
"""
has_connection(; severity=HARD) = Constraint(
  "has_connection", severity, CIRCULATION,
  r::BuildResult -> [
    Violation("has_connection", severity, CIRCULATION,
      space_name(sp), "$(space_name(sp)): has no connections", 0.0, 1.0)
    for sp in r.storey.spaces
    if isempty([c for c in r.storey.connections
                if c.space_a === sp || c.space_b === sp])])

#=== Environmental ===#

# Classify the exterior-facing direction of a wall edge relative to its space.
function _exterior_face(adj, sp)
  (p1, p2) = adj.shared_edge
  dx = abs(p2[1] - p1[1])
  dy = abs(p2[2] - p1[2])
  mid_x = (p1[1] + p2[1]) / 2
  mid_y = (p1[2] + p2[2]) / 2
  (ox, oy) = space_origin(sp)
  cx_sp = ox + space_width(sp) / 2
  cy_sp = oy + space_depth(sp) / 2
  if dy > dx
    mid_x > cx_sp ? :east : :west
  else
    mid_y > cy_sp ? :north : :south
  end
end

# Length of an adjacency's shared edge (2D).
function _edge_length_xy(edge)
  dx = edge[2][1] - edge[1][1]
  dy = edge[2][2] - edge[1][2]
  sqrt(dx^2 + dy^2)
end

"""
    preferred_orientation(kind, orientation; severity=PREFERENCE)

Prefer that every space with the given `kind` has at least one
exterior wall facing `orientation` (one of `:north`, `:south`,
`:east`, `:west`). Emits a `PREFERENCE` violation per space with no
qualifying exterior edge.
"""
preferred_orientation(kind::Symbol, orientation::Symbol; severity=PREFERENCE) =
  Constraint(
    "orientation_$(kind)_$(orientation)", severity, ENVIRONMENTAL,
    ctx -> let
      out = Violation[]
      adjs = _ctx_adjacencies(ctx)
      for sp in _ctx_spaces(ctx)
        sp.kind == kind || continue
        ext = [adj for adj in adjs
               if adj.space_a == sp.id && isnothing(adj.space_b)]
        has_or = any(adj -> _exterior_face(adj, sp) == orientation, ext)
        has_or || push!(out, Violation(
          "orientation_$(kind)_$(orientation)", severity, ENVIRONMENTAL,
          space_name(sp),
          "$(space_name(sp)) has no exterior face toward $(orientation)",
          0.0, 1.0))
      end
      out
    end)

"""
    facade_ratio(kind, min_ratio; severity=SOFT)

Each space with the given `kind` must have at least `min_ratio` of
its perimeter exposed to the exterior. Encourages daylit rooms on
the building skin rather than landlocked interior zones.
"""
facade_ratio(kind::Symbol, min_ratio::Real; severity=SOFT) = Constraint(
  "facade_ratio_$(kind)_$(min_ratio)", severity, ENVIRONMENTAL,
  ctx -> let
    out = Violation[]
    adjs = _ctx_adjacencies(ctx)
    for sp in _ctx_spaces(ctx)
      sp.kind == kind || continue
      perimeter = 2 * (space_width(sp) + space_depth(sp))
      perimeter > 0 || continue
      ext_len = sum(
        (_edge_length_xy(adj.shared_edge)
         for adj in adjs
         if adj.space_a == sp.id && isnothing(adj.space_b));
        init=0.0)
      ratio = ext_len / perimeter
      ratio >= min_ratio && continue
      push!(out, Violation(
        "facade_ratio_$(kind)", severity, ENVIRONMENTAL,
        space_name(sp),
        "$(space_name(sp)): facade ratio $(round(ratio; digits=3)) < $(min_ratio)",
        ratio, Float64(min_ratio)))
    end
    out
  end)

"""
    min_exterior_exposure(kind, min_edge_length; severity=SOFT)

Every space of the given `kind` must have at least `min_edge_length`
metres of exterior-facing edge (for daylight / ventilation).
"""
min_exterior_exposure(kind::Symbol, min_edge_length::Real; severity=SOFT) =
  Constraint(
    "ext_exposure_$(kind)_$(min_edge_length)", severity, ENVIRONMENTAL,
    ctx -> let
      out = Violation[]
      adjs = _ctx_adjacencies(ctx)
      for sp in _ctx_spaces(ctx)
        sp.kind == kind || continue
        ext_len = sum(
          _edge_length_xy(adj.shared_edge)
          for adj in adjs
          if adj.space_a == sp.id && isnothing(adj.space_b);
          init=0.0)
        ext_len < min_edge_length || continue
        push!(out, Violation(
          "ext_exposure_$(kind)", severity, ENVIRONMENTAL,
          space_name(sp),
          "$(space_name(sp)): exterior edge $(ext_len)m < $(min_edge_length)m",
          ext_len, Float64(min_edge_length)))
      end
      out
    end)
