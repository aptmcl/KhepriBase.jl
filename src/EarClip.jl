# Robust triangulation of an outer boundary MINUS holes.
#
# This is a faithful, self-contained port of the mapbox/earcut hole-elimination
# ear-clipping algorithm (the non-z-order variant), operating on a doubly linked
# list of vertices. Holes are spliced into the outer boundary with doubled
# "bridge" edges (earcut's findHoleBridge: leftmost hole vertex, leftward ray to
# the nearest outer edge, reflex-vertex-in-triangle tie-break); the linked-list
# ear clipper then triangulates the result. Unlike the plain-coordinate
# `triangulate_polygon`, this clipper keeps bridge endpoints topologically
# distinct, so the coincident doubled-bridge vertices do not defeat the ear test.
#
# `triangulate_polygon` (the simple-polygon path used by b_surface_polygon and
# every backend) is left completely untouched.
#
# Winding convention: callers pass holes in the OPPOSITE winding to the outer
# boundary (outer CCW => holes CW) - every call site wraps holes in
# `reverse(path_vertices(hole))`. We additionally re-orient by 2D signed area, so
# a hole supplied with the wrong winding is still handled.
#
# Returns (coords, trigs): `coords` is the combined vertex vector
# `vcat(outer, holes...)` of the ORIGINAL point objects (callers keep exact
# types / coordinate systems for b_trig); `trigs` are (i,j,k) index triples into
# `coords`.
public triangulate_polygon_with_holes
function triangulate_polygon_with_holes(outer, holes)
  hs = [h for h in holes if length(h) >= 3]
  isempty(hs) && return (collect(outer), triangulate_polygon([raw_point(p) for p in outer]))
  combined = collect(outer)
  for h in hs; append!(combined, h); end
  raw = [raw_point(p) for p in combined]
  n_outer = length(outer)
  # 2D projection: drop the axis most aligned with the outer's Newell normal.
  nx, ny, nz = 0.0, 0.0, 0.0
  for i in 1:n_outer
    j = mod(i, n_outer) + 1
    a = raw[i]; b = raw[j]
    nx += (a[2] - b[2]) * (a[3] + b[3])
    ny += (a[3] - b[3]) * (a[1] + b[1])
    nz += (a[1] - b[1]) * (a[2] + b[2])
  end
  ax, ay, az = abs(nx), abs(ny), abs(nz)
  ci = ax >= ay && ax >= az ? (2, 3) : ay >= ax && ay >= az ? (1, 3) : (1, 2)
  U = [Float64(r[ci[1]]) for r in raw]
  V = [Float64(r[ci[2]]) for r in raw]
  # Loop index sequences (into combined/raw/U/V).
  outer_idx = collect(1:n_outer)
  hole_loops = Vector{Vector{Int}}()
  off = n_outer
  for h in hs
    push!(hole_loops, collect(off+1:off+length(h)))
    off += length(h)
  end
  # Enforce earcut's expected orientation: outer CCW, holes CW (in the 2D frame).
  _loop_area2(seq) = begin
    s = 0.0; m = length(seq)
    for i in 1:m
      p = seq[i]; q = seq[mod(i, m) + 1]
      s += U[p] * V[q] - U[q] * V[p]
    end
    s
  end
  _loop_area2(outer_idx) < 0 && reverse!(outer_idx)
  for hl in hole_loops
    _loop_area2(hl) > 0 && reverse!(hl)
  end
  # Build the linked list and triangulate.
  ec = _EC()
  outerNode = _linked_list(ec, outer_idx, U, V)
  (outerNode == 0 || ec.next[outerNode] == ec.prev[outerNode]) &&
    return (combined, Tuple{Int,Int,Int}[])
  outerNode = _eliminate_holes(ec, hole_loops, outerNode, U, V)
  trigs = Tuple{Int,Int,Int}[]
  _earcut_linked(ec, outerNode, trigs, 0)
  (combined, trigs)
end

# ---- Linked-list node store (parallel arrays; node id = index) --------------
mutable struct _EC
  vi::Vector{Int}       # combined-vertex index this node refers to
  x::Vector{Float64}
  y::Vector{Float64}
  prev::Vector{Int}
  next::Vector{Int}
  steiner::Vector{Bool}
end
_EC() = _EC(Int[], Float64[], Float64[], Int[], Int[], Bool[])

_new_node(ec, vi, x, y) =
  (push!(ec.vi, vi); push!(ec.x, x); push!(ec.y, y);
   push!(ec.prev, 0); push!(ec.next, 0); push!(ec.steiner, false);
   length(ec.vi))

function _insert_node(ec, vi, x, y, last)
  p = _new_node(ec, vi, x, y)
  if last == 0
    ec.prev[p] = p; ec.next[p] = p
  else
    ec.next[p] = ec.next[last]
    ec.prev[p] = last
    ec.prev[ec.next[last]] = p
    ec.next[last] = p
  end
  p
end

_remove_node(ec, p) =
  (ec.next[ec.prev[p]] = ec.next[p]; ec.prev[ec.next[p]] = ec.prev[p])

function _linked_list(ec, idxseq, U, V)
  last = 0
  for vi in idxseq
    last = _insert_node(ec, vi, U[vi], V[vi], last)
  end
  if last != 0 && _equals(ec, last, ec.next[last])
    _remove_node(ec, last)
    last = ec.next[last]
  end
  last
end

# ---- Geometric primitives (earcut) ------------------------------------------
_area(ec, p, q, r) =
  (ec.y[q] - ec.y[p]) * (ec.x[r] - ec.x[q]) -
  (ec.x[q] - ec.x[p]) * (ec.y[r] - ec.y[q])

_equals(ec, p, q) = ec.x[p] == ec.x[q] && ec.y[p] == ec.y[q]

_sign(v) = v > 0 ? 1 : v < 0 ? -1 : 0

_point_in_triangle(ax, ay, bx, by, cx, cy, px, py) =
  (cx - px) * (ay - py) - (ax - px) * (cy - py) >= 0 &&
  (ax - px) * (by - py) - (bx - px) * (ay - py) >= 0 &&
  (bx - px) * (cy - py) - (cx - px) * (by - py) >= 0

_on_segment(px, py, qx, qy, rx, ry) =
  qx <= max(px, rx) && qx >= min(px, rx) &&
  qy <= max(py, ry) && qy >= min(py, ry)

# Do segments p1-q1 and p2-q2 intersect?
function _intersects(ec, p1, q1, p2, q2)
  o1 = _sign(_area(ec, p1, q1, p2))
  o2 = _sign(_area(ec, p1, q1, q2))
  o3 = _sign(_area(ec, p2, q2, p1))
  o4 = _sign(_area(ec, p2, q2, q1))
  (o1 != o2 && o3 != o4) && return true
  o1 == 0 && _on_segment(ec.x[p1], ec.y[p1], ec.x[p2], ec.y[p2], ec.x[q1], ec.y[q1]) && return true
  o2 == 0 && _on_segment(ec.x[p1], ec.y[p1], ec.x[q2], ec.y[q2], ec.x[q1], ec.y[q1]) && return true
  o3 == 0 && _on_segment(ec.x[p2], ec.y[p2], ec.x[p1], ec.y[p1], ec.x[q2], ec.y[q2]) && return true
  o4 == 0 && _on_segment(ec.x[p2], ec.y[p2], ec.x[q1], ec.y[q1], ec.x[q2], ec.y[q2]) && return true
  false
end

# Does segment a-b cross any polygon edge?
function _intersects_polygon(ec, a, b)
  p = a
  while true
    pn = ec.next[p]
    if p != a && pn != a && p != b && pn != b && _intersects(ec, p, pn, a, b)
      return true
    end
    p = pn
    p == a && break
  end
  false
end

# Is b inside the sector at a?
_locally_inside(ec, a, b) =
  _area(ec, ec.prev[a], a, ec.next[a]) < 0 ?
    (_area(ec, a, b, ec.next[a]) >= 0 && _area(ec, a, ec.prev[a], b) >= 0) :
    (_area(ec, a, b, ec.prev[a]) < 0 || _area(ec, a, ec.next[a], b) < 0)

# Is the midpoint of a-b inside the polygon?
function _middle_inside(ec, a, b)
  p = a
  inside = false
  px = (ec.x[a] + ec.x[b]) / 2
  py = (ec.y[a] + ec.y[b]) / 2
  while true
    pn = ec.next[p]
    if ((ec.y[p] > py) != (ec.y[pn] > py)) && ec.y[pn] != ec.y[p] &&
       (px < (ec.x[pn] - ec.x[p]) * (py - ec.y[p]) / (ec.y[pn] - ec.y[p]) + ec.x[p])
      inside = !inside
    end
    p = pn
    p == a && break
  end
  inside
end

_sector_contains_sector(ec, m, p) =
  _area(ec, ec.prev[m], m, ec.prev[p]) < 0 && _area(ec, ec.next[p], ec.next[m], m) < 0

# ---- Point / hole removal ---------------------------------------------------
# Remove collinear or duplicate vertices between start and en (inclusive loop).
function _filter_points(ec, start, en=0)
  start == 0 && return start
  en == 0 && (en = start)
  p = start
  while true
    again = false
    if !ec.steiner[p] && (_equals(ec, p, ec.next[p]) || _area(ec, ec.prev[p], p, ec.next[p]) == 0)
      _remove_node(ec, p)
      p = en = ec.prev[p]
      p == ec.next[p] && break
      again = true
    else
      p = ec.next[p]
    end
    (again || p != en) || break
  end
  en
end

function _get_leftmost(ec, start)
  p = start
  leftmost = start
  while true
    if ec.x[p] < ec.x[leftmost] || (ec.x[p] == ec.x[leftmost] && ec.y[p] < ec.y[leftmost])
      leftmost = p
    end
    p = ec.next[p]
    p == start && break
  end
  leftmost
end

# Split the polygon by a diagonal a-b, returning the new node paired with a2.
function _split_polygon(ec, a, b)
  a2 = _new_node(ec, ec.vi[a], ec.x[a], ec.y[a])
  b2 = _new_node(ec, ec.vi[b], ec.x[b], ec.y[b])
  an = ec.next[a]
  bp = ec.prev[b]
  ec.next[a] = b
  ec.prev[b] = a
  ec.next[a2] = an
  ec.prev[an] = a2
  ec.next[b2] = a2
  ec.prev[a2] = b2
  ec.next[bp] = b2
  ec.prev[b2] = bp
  b2
end

# earcut findHoleBridge: leftmost hole vertex, leftward ray, reflex tie-break.
function _find_hole_bridge(ec, hole, outerNode)
  p = outerNode
  hx = ec.x[hole]; hy = ec.y[hole]
  qx = -Inf
  m = 0
  while true
    pn = ec.next[p]
    if _equals(ec, hole, p); return p; end
    if hy <= ec.y[p] && hy >= ec.y[pn] && ec.y[pn] != ec.y[p]
      x = ec.x[p] + (hy - ec.y[p]) * (ec.x[pn] - ec.x[p]) / (ec.y[pn] - ec.y[p])
      if x <= hx && x > qx
        qx = x
        m = ec.x[p] < ec.x[pn] ? p : pn
        x == hx && return m
      end
    end
    p = pn
    p == outerNode && break
  end
  m == 0 && return 0
  stop = m
  mx = ec.x[m]; my = ec.y[m]
  tanMin = Inf
  p = m
  while true
    if hx >= ec.x[p] && ec.x[p] >= mx && hx != ec.x[p] &&
       _point_in_triangle(hy < my ? hx : qx, hy, mx, my,
                          hy < my ? qx : hx, hy, ec.x[p], ec.y[p])
      tanv = abs(hy - ec.y[p]) / (hx - ec.x[p])
      if _locally_inside(ec, p, hole) &&
         (tanv < tanMin ||
          (tanv == tanMin && (ec.x[p] > ec.x[m] ||
           (ec.x[p] == ec.x[m] && _sector_contains_sector(ec, m, p)))))
        m = p
        tanMin = tanv
      end
    end
    p = ec.next[p]
    p == stop && break
  end
  m
end

function _eliminate_hole(ec, hole, outerNode)
  bridge = _find_hole_bridge(ec, hole, outerNode)
  bridge == 0 && return outerNode
  bridgeReverse = _split_polygon(ec, bridge, hole)
  _filter_points(ec, bridgeReverse, ec.next[bridgeReverse])
  _filter_points(ec, bridge, ec.next[bridge])
end

function _eliminate_holes(ec, hole_loops, outerNode, U, V)
  queue = Int[]
  for hl in hole_loops
    list = _linked_list(ec, hl, U, V)
    list == 0 && continue
    list == ec.next[list] && (ec.steiner[list] = true)
    push!(queue, _get_leftmost(ec, list))
  end
  sort!(queue, by = q -> (ec.x[q], ec.y[q]))
  for q in queue
    outerNode = _eliminate_hole(ec, q, outerNode)
  end
  outerNode
end

# ---- Ear clipping -----------------------------------------------------------
function _is_ear(ec, ear)
  a = ec.prev[ear]; b = ear; c = ec.next[ear]
  _area(ec, a, b, c) >= 0 && return false  # reflex
  ax = ec.x[a]; bx = ec.x[b]; cx = ec.x[c]
  ay = ec.y[a]; by = ec.y[b]; cy = ec.y[c]
  x0 = min(ax, bx, cx); y0 = min(ay, by, cy)
  x1 = max(ax, bx, cx); y1 = max(ay, by, cy)
  p = ec.next[c]
  while p != a
    if ec.x[p] >= x0 && ec.x[p] <= x1 && ec.y[p] >= y0 && ec.y[p] <= y1 &&
       _point_in_triangle(ax, ay, bx, by, cx, cy, ec.x[p], ec.y[p]) &&
       _area(ec, ec.prev[p], p, ec.next[p]) >= 0
      return false
    end
    p = ec.next[p]
  end
  true
end

function _cure_local_intersections(ec, start, trigs)
  p = start
  while true
    a = ec.prev[p]
    b = ec.next[ec.next[p]]
    if !_equals(ec, a, b) && _intersects(ec, a, p, ec.next[p], b) &&
       _locally_inside(ec, a, b) && _locally_inside(ec, b, a)
      push!(trigs, (ec.vi[a], ec.vi[p], ec.vi[b]))
      _remove_node(ec, p)
      _remove_node(ec, ec.next[p])
      p = start = b
    end
    p = ec.next[p]
    p == start && break
  end
  _filter_points(ec, p)
end

_is_valid_diagonal(ec, a, b) =
  ec.vi[ec.next[a]] != ec.vi[b] && ec.vi[ec.prev[a]] != ec.vi[b] &&
  !_intersects_polygon(ec, a, b) &&
  ((_locally_inside(ec, a, b) && _locally_inside(ec, b, a) && _middle_inside(ec, a, b) &&
    (_area(ec, ec.prev[a], a, ec.prev[b]) != 0 || _area(ec, a, ec.prev[b], b) != 0)) ||
   (_equals(ec, a, b) && _area(ec, ec.prev[a], a, ec.next[a]) > 0 &&
    _area(ec, ec.prev[b], b, ec.next[b]) > 0))

function _split_earcut(ec, start, trigs)
  a = start
  while true
    b = ec.next[ec.next[a]]
    while b != ec.prev[a]
      if ec.vi[a] != ec.vi[b] && _is_valid_diagonal(ec, a, b)
        c = _split_polygon(ec, a, b)
        a = _filter_points(ec, a, ec.next[a])
        c = _filter_points(ec, c, ec.next[c])
        _earcut_linked(ec, a, trigs, 0)
        _earcut_linked(ec, c, trigs, 0)
        return
      end
      b = ec.next[b]
    end
    a = ec.next[a]
    a == start && break
  end
end

function _earcut_linked(ec, ear, trigs, pass)
  ear == 0 && return
  stop = ear
  while ec.prev[ear] != ec.next[ear]
    prev = ec.prev[ear]
    next = ec.next[ear]
    if _is_ear(ec, ear)
      push!(trigs, (ec.vi[prev], ec.vi[ear], ec.vi[next]))
      _remove_node(ec, ear)
      ear = ec.next[next]
      stop = ec.next[next]
      continue
    end
    ear = next
    if ear == stop
      if pass == 0
        _earcut_linked(ec, _filter_points(ec, ear), trigs, 1)
      elseif pass == 1
        ear = _cure_local_intersections(ec, _filter_points(ec, ear), trigs)
        _earcut_linked(ec, ear, trigs, 2)
      elseif pass == 2
        _split_earcut(ec, ear, trigs)
      end
      break
    end
  end
end