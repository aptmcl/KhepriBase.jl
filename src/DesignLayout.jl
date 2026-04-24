#####################################################################
# DesignLayout — Level 2 → Level 1 compilation
#
# Walks a `SpaceDesc` tree (Level 2) and assigns concrete (x, y, z)
# coordinates to every leaf, returning a unified `Layout` (Level 1).
#
# The engine operates on `Space`s directly: each leaf constructs a
# `Space` with a concrete `ClosedPath` boundary, which the storey
# pass then groups by z-elevation. Earlier revisions carried an
# engine-internal `PlacedSpace` view that duplicated the bbox fields
# and was converted at the end — it is gone.

export rectangular_boundary

# Rectangular 2D polygon helper used throughout the engine to
# materialise axis-aligned space boundaries.
"Axis-aligned rectangle as `Vector{NTuple{2, Float64}}` counterclockwise from `(x, y)`."
rectangular_boundary(x, y, w, d) =
  [(x, y), (x + w, y), (x + w, y + d), (x, y + d)]

"""
    layout(desc::SpaceDesc; origin_x=0.0, origin_y=0.0, origin_z=0.0)

Walk the `SpaceDesc` tree and assign concrete world coordinates to
every space. Returns a [`Layout`](@ref) whose `storeys` group placed
spaces by z-elevation. Adjacencies are computed on demand via
[`adjacencies`](@ref).
"""
function layout(desc::SpaceDesc; origin_x=0.0, origin_y=0.0, origin_z=0.0)
  spaces = Dict{Symbol, Space}()
  _layout!(spaces, desc, Float64(origin_x), Float64(origin_y), Float64(origin_z), Symbol[])
  _build_layout(spaces, collect_annotations(desc))
end

#=
Group placed `Space`s by z-elevation into one `Storey` per level,
giving each storey the tallest per-space height as its floor-to-
floor distance. The storey's `Level` is anchored at the group's z.
=#
function _build_layout(spaces::Dict{Symbol, Space}, annotations)
  groups = Dict{Float64, Vector{Space}}()
  for sp in values(spaces)
    z = round(sp.origin_z, digits=6)
    push!(get!(groups, z, Space[]), sp)
  end
  zs = sort(collect(keys(groups)))
  storeys = Storey[]
  for z in zs
    members = groups[z]
    h = isempty(members) ? 0.0 : maximum(sp.height for sp in members)
    push!(storeys,
          Storey(members, SpaceConnection[], level(z), h,
                 default_wall_family(), default_slab_family(), true))
  end
  Layout(storeys, Constraint[], collect(annotations))
end

# Construct a Space at origin (x, y, z) with rectangular footprint (w, d).
# Every `_layout!` leaf branch goes through this helper so the
# boundary-construction detail stays in one place.
_make_space(id, use, x, y, z, w, d, h, props) =
  let verts = [xy(p[1], p[2]) for p in rectangular_boundary(x, y, w, d)]
    Space(id, use, closed_polygonal_path(verts);
          height=h, props=props, origin_z=z)
  end

# ---- Recursive layout ----
# _layout! populates `spaces` dict in-place and returns (width, depth, height)
# of the laid-out subtree. `prefix` is the namespace path for scoped naming.

function _layout!(spaces, r::Room, x, y, z, prefix)
  id = _scoped_id(prefix, r.id)
  spaces[id] = _make_space(id, r.use, x, y, z, r.width, r.depth, r.height, r.props)
  (r.width, r.depth, r.height)
end

function _layout!(spaces, v::Void, x, y, z, prefix)
  # A Void reserves its declared dimensions as a spacer but produces no
  # Space, so it leaves a gap in adjacency detection.
  (v.width, v.depth, 0.0)
end

function _layout!(spaces, e::Envelope, x, y, z, prefix)
  id = _scoped_id(prefix, e.id)
  spaces[id] = _make_space(id, :envelope, x, y, z, e.width, e.depth, e.height, e.props)
  (e.width, e.depth, e.height)
end

function _layout!(spaces, b::BesideX, x, y, z, prefix)
  lw, ld, lh = _layout!(spaces, b.left, x, y, z, prefix)
  rw, rd, rh = _layout!(spaces, b.right, x + lw, y, z, prefix)
  (lw + rw, max(ld, rd), max(lh, rh))
end

function _layout!(spaces, b::BesideY, x, y, z, prefix)
  fw, fd, fh = _layout!(spaces, b.front, x, y, z, prefix)
  bw, bd, bh = _layout!(spaces, b.back, x, y + fd, z, prefix)
  (max(fw, bw), fd + bd, max(fh, bh))
end

function _layout!(spaces, a::Above, x, y, z, prefix)
  bw, bd, bh = _layout!(spaces, a.below, x, y, z, prefix)
  aw, ad, ah = _layout!(spaces, a.above, x, y, z + bh, prefix)
  (max(bw, aw), max(bd, ad), bh + ah)
end

function _layout!(spaces, r::Repeated, x, y, z, prefix)
  total_w = 0.0
  total_d = 0.0
  total_h = 0.0
  for i in 1:r.count
    unit_prefix = vcat(prefix, [Symbol("unit_", i)])
    # Mirroring only makes sense along a plan axis; suppress it for :z.
    unit_desc = if r.mirror_alternate && iseven(i) && r.axis != :z
      Mirrored(r.unit, r.axis == :x ? :x : :y)
    else
      r.unit
    end
    if r.axis == :x
      uw, ud, uh = _layout!(spaces, unit_desc, x + total_w, y, z, unit_prefix)
      total_w += uw
      total_d = max(total_d, ud)
      total_h = max(total_h, uh)
    elseif r.axis == :y
      uw, ud, uh = _layout!(spaces, unit_desc, x, y + total_d, z, unit_prefix)
      total_w = max(total_w, uw)
      total_d += ud
      total_h = max(total_h, uh)
    else  # :z
      # Stack along the z axis: each copy's placed spaces land on a
      # successively higher storey, and the scoped `unit_i/` prefix
      # keeps room ids unique across floors.
      uw, ud, uh = _layout!(spaces, unit_desc, x, y, z + total_h, unit_prefix)
      total_w = max(total_w, uw)
      total_d = max(total_d, ud)
      total_h += uh
    end
  end
  (total_w, total_d, total_h)
end

function _layout!(spaces, g::GridLayout, x, y, z, prefix)
  col_widths = KhepriBase._grid_col_widths(g)
  row_depths = KhepriBase._grid_row_depths(g)
  col_offsets = [0.0; cumsum(col_widths)]
  row_offsets = [0.0; cumsum(row_depths)]
  grid_h = 0.0
  for r in 1:g.rows, c in 1:g.cols
    cell = g.cell_fn(r, c)
    _, _, ch = _layout!(spaces, cell, x + col_offsets[c], y + row_offsets[r], z, prefix)
    grid_h = max(grid_h, ch)
  end
  (sum(col_widths), sum(row_depths), grid_h)
end

function _layout!(spaces, s::Scaled, x, y, z, prefix)
  # Lay the subtree out at (0, 0, 0), then re-stamp each child Space
  # with scaled coordinates — the stored boundary is rebuilt so bbox
  # queries report the scaled dimensions without a coordinate-free
  # transform on `ClosedPath`.
  sub = Dict{Symbol, Space}()
  sw, sd, sh = _layout!(sub, s.base, 0.0, 0.0, 0.0, prefix)
  for (id, sp) in sub
    (ox, oy) = space_origin(sp)
    w = space_width(sp)
    d = space_depth(sp)
    spaces[id] = _make_space(
      id, sp.kind,
      x + ox * s.sx, y + oy * s.sy, z + sp.origin_z,
      w * s.sx, d * s.sy, sp.height, sp.props)
  end
  (sw * s.sx, sd * s.sy, sh)
end

function _layout!(spaces, m::Mirrored, x, y, z, prefix)
  sub = Dict{Symbol, Space}()
  sw, sd, sh = _layout!(sub, m.base, 0.0, 0.0, 0.0, prefix)
  for (id, sp) in sub
    (ox, oy) = space_origin(sp)
    w = space_width(sp)
    d = space_depth(sp)
    new_x, new_y = if m.axis == :x
      (sw - (ox + w), oy)
    else  # :y
      (ox, sd - (oy + d))
    end
    spaces[id] = _make_space(
      id, sp.kind,
      x + new_x, y + new_y, z + sp.origin_z,
      w, d, sp.height, sp.props)
  end
  (sw, sd, sh)
end

function _layout!(spaces, h::HeightOverride, x, y, z, prefix)
  sw, sd, _ = _layout!(spaces, h.base, x, y, z, prefix)
  # Override height on every space laid out on this level.
  for (id, sp) in spaces
    if sp.origin_z == z
      (ox, oy) = space_origin(sp)
      spaces[id] = _make_space(
        id, sp.kind, ox, oy, sp.origin_z,
        space_width(sp), space_depth(sp), h.height, sp.props)
    end
  end
  (sw, sd, h.height)
end

function _layout!(spaces, a::Annotated, x, y, z, prefix)
  # Annotations are transparent to layout
  _layout!(spaces, a.base, x, y, z, prefix)
end

function _layout!(spaces, p::PropsOverlay, x, y, z, prefix)
  before = Set(keys(spaces))
  dims = _layout!(spaces, p.base, x, y, z, prefix)
  for (id, sp) in spaces
    id in before && continue
    (ox, oy) = space_origin(sp)
    spaces[id] = _make_space(
      id, sp.kind, ox, oy, sp.origin_z,
      space_width(sp), space_depth(sp), sp.height,
      merge(p.props, sp.props))
  end
  dims
end

function _layout!(spaces, s::Subdivided, x, y, z, prefix)
  base_w = desc_width(s.base)
  base_d = desc_depth(s.base)
  base_h = desc_height(s.base)
  if s.axis == :x
    cx_pos = x
    for (ratio, id) in zip(s.ratios, s.ids)
      zone_w = base_w * ratio
      sid = _scoped_id(prefix, id)
      spaces[sid] = _make_space(sid, :zone, cx_pos, y, z, zone_w, base_d, base_h, (;))
      cx_pos += zone_w
    end
  else  # :y
    cy_pos = y
    for (ratio, id) in zip(s.ratios, s.ids)
      zone_d = base_d * ratio
      sid = _scoped_id(prefix, id)
      spaces[sid] = _make_space(sid, :zone, x, cy_pos, z, base_w, zone_d, base_h, (;))
      cy_pos += zone_d
    end
  end
  (base_w, base_d, base_h)
end

function _layout!(spaces, p::Partitioned, x, y, z, prefix)
  base_w = desc_width(p.base)
  base_d = desc_depth(p.base)
  base_h = desc_height(p.base)
  if p.axis == :x
    cell_w = base_w / p.count
    for i in 1:p.count
      id = _scoped_id(prefix, Symbol(p.id_prefix, "_", i))
      cx_pos = x + (i - 1) * cell_w
      spaces[id] = _make_space(id, :zone, cx_pos, y, z, cell_w, base_d, base_h, (;))
    end
  else  # :y
    cell_d = base_d / p.count
    for i in 1:p.count
      id = _scoped_id(prefix, Symbol(p.id_prefix, "_", i))
      cy_pos = y + (i - 1) * cell_d
      spaces[id] = _make_space(id, :zone, x, cy_pos, z, base_w, cell_d, base_h, (;))
    end
  end
  (base_w, base_d, base_h)
end

function _layout!(spaces, c::Carved, x, y, z, prefix)
  bw, bd, bh = _layout!(spaces, c.base, x, y, z, prefix)
  id = _scoped_id(prefix, c.id)
  spaces[id] = _make_space(id, c.use,
    x + c.x, y + c.y, z, c.width, c.depth, bh, (;))
  (bw, bd, bh)
end

function _layout!(spaces, r::Refined, x, y, z, prefix)
  # Lay out the base first so the zone's placed Space is available.
  bw, bd, bh = _layout!(spaces, r.base, x, y, z, prefix)
  zone_id = _scoped_id(prefix, r.zone_id)
  haskey(spaces, zone_id) || error("refine: zone '$(r.zone_id)' not found in layout")
  zone = spaces[zone_id]
  # Replace the zone placeholder with the transformed subtree, anchored
  # at the zone's origin. Polar zones (flagged by a `:_polar` entry in
  # their props) receive a `polar_envelope`; rectangular zones receive
  # a standard `envelope`, so the transform can dispatch on type.
  if haskey(zone.props, :_polar)
    p = zone.props._polar
    env = polar_envelope(p.center, p.r_inner, p.r_outer,
                         p.theta_start, p.theta_end, zone.height;
                         id=r.zone_id, use=zone.kind,
                         props=Base.structdiff(zone.props, NamedTuple{(:_polar,)}),
                         n_arc=get(p, :n_arc, 16))
    delete!(spaces, zone_id)
    new_desc = r.transform(env)
    # Polar subtrees are absolute-positioned; pass zeroed (x, y).
    _layout!(spaces, new_desc, 0.0, 0.0, zone.origin_z, prefix)
  else
    (zx, zy) = space_origin(zone)
    zw, zd = space_width(zone), space_depth(zone)
    delete!(spaces, zone_id)
    new_desc = r.transform(envelope(zw, zd, zone.height; id=r.zone_id))
    _layout!(spaces, new_desc, zx, zy, zone.origin_z, prefix)
  end
  (bw, bd, bh)
end

function _layout!(spaces, a::Assigned, x, y, z, prefix)
  bw, bd, bh = _layout!(spaces, a.base, x, y, z, prefix)
  zone_id = _scoped_id(prefix, a.zone_id)
  haskey(spaces, zone_id) || error("assign: zone '$(a.zone_id)' not found in layout")
  zone = spaces[zone_id]
  (zx, zy) = space_origin(zone)
  spaces[zone_id] = _make_space(zone.id, a.use,
    zx, zy, zone.origin_z,
    space_width(zone), space_depth(zone), zone.height, a.props)
  (bw, bd, bh)
end

function _layout!(spaces, sr::SubdivideRemaining, x, y, z, prefix)
  before = Set(keys(spaces))
  bw, bd, bh = _layout!(spaces, sr.base, x, y, z, prefix)
  added = [id for id in keys(spaces) if !(id in before)]
  # Separate the envelope-like base (spans the full footprint) from carves.
  envelope_id = nothing
  carves = Symbol[]
  for id in added
    sp = spaces[id]
    sp.origin_z ≈ z || continue
    (ox, oy) = space_origin(sp)
    w = space_width(sp); d = space_depth(sp)
    if ox ≈ x && oy ≈ y && w ≈ bw && d ≈ bd
      envelope_id = id
    else
      push!(carves, id)
    end
  end
  length(carves) == 1 ||
    error("subdivide_remaining: expects exactly one carved zone in the base, got $(length(carves))")
  isnothing(envelope_id) || delete!(spaces, envelope_id)
  hole = spaces[carves[1]]
  (hx, hy) = space_origin(hole)
  hw, hd = space_width(hole), space_depth(hole)
  block_rect = Dict{Symbol, NTuple{4, Float64}}(
    :north => (x, hy + hd, bw, (y + bd) - (hy + hd)),
    :south => (x, y, bw, hy - y),
    :east  => (hx + hw, hy, (x + bw) - (hx + hw), hd),
    :west  => (x, hy, hx - x, hd),
  )
  for (bid, pos) in sr.blocks
    haskey(block_rect, pos) || error("subdivide_remaining: unknown position $pos")
    (bx, by, bwid, bdep) = block_rect[pos]
    (bwid > 0 && bdep > 0) ||
      error("subdivide_remaining: block $bid at $pos has non-positive extent")
    sid = _scoped_id(prefix, bid)
    spaces[sid] = _make_space(sid, :zone, bx, by, z, bwid, bdep, bh, (;))
  end
  (bw, bd, bh)
end

# ---- Polar branches ----
#
# Polar nodes carry their own absolute geometry (a `center` `Loc`
# plus radii / angles), so they ignore the Cartesian `(x, y)` cursor
# threaded through the rectangular tree — only `z` is meaningful,
# for vertical stacking via `Above`. Each placed Space stores its
# polar parameters in `props[:_polar]` so that [`refine`](@ref) can
# rebuild a `polar_envelope` for the transform at a deeper level.

# Build a `Space` from polar parameters, tagging its props with
# `:_polar` metadata and the boundary with the polygonal discretisation.
function _make_polar_space(id, use, center, r_inner, r_outer,
                           theta_start, theta_end, z, height, n_arc, props)
  boundary = polar_sector_path(center, r_inner, r_outer,
                               theta_start, theta_end; n_arc=n_arc)
  polar_meta = (_polar = (center=center, r_inner=Float64(r_inner),
                          r_outer=Float64(r_outer),
                          theta_start=Float64(theta_start),
                          theta_end=Float64(theta_end),
                          n_arc=Int(n_arc)),)
  Space(id, use, boundary;
        height=Float64(height),
        props=merge(props, polar_meta),
        origin_z=Float64(z))
end

function _layout!(spaces, pe::PolarEnvelope, x, y, z, prefix)
  id = _scoped_id(prefix, pe.id)
  spaces[id] = _make_polar_space(id, pe.use, pe.center,
    pe.r_inner, pe.r_outer, pe.theta_start, pe.theta_end,
    z, pe.height, pe.n_arc, pe.props)
  # Polar subtrees are absolute-positioned, so report zero contribution
  # to the Cartesian (x, y) budget. `above` still stacks correctly via
  # the height term.
  (0.0, 0.0, pe.height)
end

function _layout!(spaces, sp::SubdividedPolar, x, y, z, prefix)
  pe = sp.base isa PolarEnvelope ? sp.base :
    error("subdivide_$(sp.axis): base must be a polar_envelope")
  total = sum(sp.ratios)
  if sp.axis == :radial
    r_span = pe.r_outer - pe.r_inner
    cursor = pe.r_inner
    for (ratio, id) in zip(sp.ratios, sp.ids)
      r_band = r_span * ratio / total
      sid = _scoped_id(prefix, id)
      spaces[sid] = _make_polar_space(sid, :zone, pe.center,
        cursor, cursor + r_band,
        pe.theta_start, pe.theta_end,
        z, pe.height, pe.n_arc, (;))
      cursor += r_band
    end
  elseif sp.axis == :angular
    th_span = pe.theta_end - pe.theta_start
    cursor = pe.theta_start
    for (ratio, id) in zip(sp.ratios, sp.ids)
      th_band = th_span * ratio / total
      sid = _scoped_id(prefix, id)
      spaces[sid] = _make_polar_space(sid, :zone, pe.center,
        pe.r_inner, pe.r_outer,
        cursor, cursor + th_band,
        z, pe.height, pe.n_arc, (;))
      cursor += th_band
    end
  else
    error("subdivide polar: unknown axis $(sp.axis)")
  end
  (0.0, 0.0, pe.height)
end

function _layout!(spaces, pp::PartitionedPolar, x, y, z, prefix)
  pe = pp.base isa PolarEnvelope ? pp.base :
    error("partition_$(pp.axis): base must be a polar_envelope")
  if pp.axis == :angular
    th_step = (pe.theta_end - pe.theta_start) / pp.count
    for i in 1:pp.count
      sid = _scoped_id(prefix, Symbol(pp.id_prefix, "_", i))
      th0 = pe.theta_start + (i - 1) * th_step
      spaces[sid] = _make_polar_space(sid, :zone, pe.center,
        pe.r_inner, pe.r_outer, th0, th0 + th_step,
        z, pe.height, pe.n_arc, (;))
    end
  elseif pp.axis == :radial
    r_step = (pe.r_outer - pe.r_inner) / pp.count
    for i in 1:pp.count
      sid = _scoped_id(prefix, Symbol(pp.id_prefix, "_", i))
      r0 = pe.r_inner + (i - 1) * r_step
      spaces[sid] = _make_polar_space(sid, :zone, pe.center,
        r0, r0 + r_step, pe.theta_start, pe.theta_end,
        z, pe.height, pe.n_arc, (;))
    end
  else
    error("partition polar: unknown axis $(pp.axis)")
  end
  (0.0, 0.0, pe.height)
end

# ---- Scoped naming ----

function _scoped_id(prefix, id)
  isempty(prefix) && return id
  Symbol(join(string.(prefix), "/"), "/", id)
end
