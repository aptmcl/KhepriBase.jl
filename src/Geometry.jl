# Geometric Utilities
# Areas
export triangle_area, circle_area, annulus_area

triangle_area(a, b, c) =
  let s = (a + b + c)/2.0
    sqrt(max(0.0, s*(s - a)*(s - b)*(s - c)))
  end
circle_area(r) = π*r^2
annulus_area(rₒ, rᵢ) = π*(rₒ^2 - rᵢ^2)

project_to_world(surf) =
    transform(surf, inverse_transformation(frame_at(surf, 0, 0)))

#=

Given a poligonal line described by its vertices, we need to compute another
polygonal line that is parallel to the first one.

=#

v_in_v(v0, v1) =
  let v = v0 + v1
    v*dot(v0, v0)/dot(v, v0)
  end

rotated_v(v, alpha) =
  vpol(pol_rho(v), pol_phi(v) + alpha)

centered_rectangle(p0, w, p1) =
  let v0 = p1 - p0,
      v1 = rotated_v(v0, pi/2),
      c = loc_from_o_vx_vy(p0, v0, v1)
    rectangle(c-vy(w/2, c.cs), distance(p0, p1), w)
  end

offset_vertices(ps::Locs, d::Real, closed) =
  let qs = closed ? [ps[end], ps..., ps[1]] : ps,
      vs = map((p0, p1) -> rotated_v(unitized(p1 - p0)*d, pi/2), qs[1:end-1], qs[2:end]),
      ws = map(v_in_v, vs[1:end-1], vs[2:end])
    map(+, ps, closed ? ws : [vs[1], ws..., vs[end]])
  end

function offset(path, d::Real; join::Symbol=:miter, cap::Symbol=:butt, miter_limit::Real=10.0)
  join in (:miter, :round, :bevel) ||
    throw(ArgumentError("Unsupported offset join style $(join). Use :miter, :round, or :bevel."))
  cap in (:butt, :square, :round) ||
    throw(ArgumentError("Unsupported offset cap style $(cap). Use :butt, :square, or :round."))
  miter_limit > 0 ||
    throw(ArgumentError("miter_limit must be positive."))
  d == 0 ? path : nonzero_offset(path, d; join=join, cap=cap, miter_limit=miter_limit)
end

function nonzero_offset(path::RectangularPath, d::Real; join::Symbol=:miter, cap::Symbol=:butt, miter_limit::Real=10.0)
  join == :miter && cap == :butt ?
    rectangular_path(add_xy(path.corner, d, d), path.dx - 2d, path.dy - 2d) :
    nonzero_offset(convert(ClosedPolygonalPath, path), d; join=join, cap=cap, miter_limit=miter_limit)
end
function nonzero_offset(path::OpenPolygonalPath, d::Real; join::Symbol=:miter, cap::Symbol=:butt, miter_limit::Real=10.0)
  _offset_polygonal_path(path.vertices, d, false; join=join, cap=cap, miter_limit=miter_limit)
end
function nonzero_offset(path::ClosedPolygonalPath, d::Real; join::Symbol=:miter, cap::Symbol=:butt, miter_limit::Real=10.0)
  _offset_polygonal_path(path.vertices, d, true; join=join, cap=cap, miter_limit=miter_limit)
end
nonzero_offset(path::CircularPath, d::Real; join::Symbol=:miter, cap::Symbol=:butt, miter_limit::Real=10.0) =
  circular_path(path.center, path.radius - d)
nonzero_offset(path::ArcPath, d::Real; join::Symbol=:miter, cap::Symbol=:butt, miter_limit::Real=10.0) =
  arc_path(path.center, path.radius - d, path.start_angle, path.amplitude)
nonzero_offset(path::CompositePath{true}, d::Real; join::Symbol=:miter, cap::Symbol=:butt, miter_limit::Real=10.0) =
  CompositePath{true}(OpenPath[nonzero_offset(piece, d; join=join, cap=cap, miter_limit=miter_limit) for piece in path.pieces])

_offset_edges(ps::Locs, d::Real, closed::Bool) =
  let n = length(ps),
      pairs = closed ? [(ps[i], ps[mod1(i + 1, n)]) for i in 1:n] :
                       [(ps[i], ps[i + 1]) for i in 1:n-1]
    [(p0=p0,
      p1=p1,
      tangent=unitized(p1 - p0),
      normal=rotated_v(unitized(p1 - p0)*d, pi/2),
      a=p0 + rotated_v(unitized(p1 - p0)*d, pi/2),
      b=p1 + rotated_v(unitized(p1 - p0)*d, pi/2))
     for (p0, p1) in pairs]
  end

_signed_angle_between(v0, v1) =
  let a0 = pol_phi(v0), a1 = pol_phi(v1)
    mod(a1 - a0 + π, 2π) - π
  end

_offset_miter_point(prev, next, vertex, d, miter_limit) =
  let p = lines_intersection(prev.a, prev.b, next.a, next.b)
    if isnothing(p) || distance(vertex, p) > abs(d) * miter_limit
      nothing
    else
      p
    end
  end

_offset_join_points(prev, next, vertex, d, join, miter_limit) =
  if join == :miter
    let p = _offset_miter_point(prev, next, vertex, d, miter_limit)
      isnothing(p) ? Loc[prev.b, next.a] : Loc[p]
    end
  elseif join == :bevel || join == :round
    Loc[prev.b, next.a]
  else
    throw(ArgumentError("Unsupported offset join style $(join)."))
  end

_offset_start_point(edge, cap, d) =
  cap == :square ? edge.a - edge.tangent * abs(d) : edge.a

_offset_end_point(edge, cap, d) =
  cap == :square ? edge.b + edge.tangent * abs(d) : edge.b

function _offset_polygonal_vertices(ps::Locs, d::Real, closed::Bool; join::Symbol=:miter, cap::Symbol=:butt, miter_limit::Real=10.0)
  edges = _offset_edges(ps, d, closed)
  isempty(edges) && return ps
  if closed
    pts = Loc[]
    for i in eachindex(edges)
      prev = edges[mod1(i - 1, length(edges))]
      next = edges[i]
      append!(pts, _offset_join_points(prev, next, ps[i], d, join, miter_limit))
    end
    pts
  else
    pts = Loc[_offset_start_point(edges[1], cap, d)]
    for i in 1:max(length(edges) - 1, 0)
      append!(pts, _offset_join_points(edges[i], edges[i + 1], ps[i + 1], d, join, miter_limit))
    end
    push!(pts, _offset_end_point(edges[end], cap, d))
    pts
  end
end

function _offset_round_join_segment(vertex, p0, p1, d)
  amp = _signed_angle_between(p0 - vertex, p1 - vertex)
  arc_path(vertex, abs(d), pol_phi(p0 - vertex), amp)
end

function _offset_round_path(ps::Locs, d::Real, closed::Bool; cap::Symbol=:butt)
  edges = _offset_edges(ps, d, closed)
  isempty(edges) && return open_polygonal_path(ps)
  if closed
    start = edges[1].a
    pieces = OpenPath[]
    for i in eachindex(edges)
      edge = edges[i]
      next = edges[mod1(i + 1, length(edges))]
      vertex = ps[mod1(i + 1, length(ps))]
      !coincident_path_location(edge.a, edge.b) && push!(pieces, line_path(edge.a, edge.b))
      !coincident_path_location(edge.b, next.a) &&
        push!(pieces, _offset_round_join_segment(vertex, edge.b, next.a, d))
    end
    composite_path(pieces; closed=true)
  else
    start = _offset_start_point(edges[1], cap, d)
    pieces = OpenPath[]
    first_line_start = start
    for i in eachindex(edges)
      edge = edges[i]
      line_start = i == 1 ? first_line_start : edge.a
      line_end = i == length(edges) ? _offset_end_point(edge, cap, d) : edge.b
      !coincident_path_location(line_start, line_end) && push!(pieces, line_path(line_start, line_end))
      if i < length(edges) && !coincident_path_location(edge.b, edges[i + 1].a)
        push!(pieces, _offset_round_join_segment(ps[i + 1], edge.b, edges[i + 1].a, d))
      end
    end
    composite_path(pieces; closed=false)
  end
end

function _offset_polygonal_path(ps::Locs, d::Real, closed::Bool; join::Symbol=:miter, cap::Symbol=:butt, miter_limit::Real=10.0)
  join == :round && return _offset_round_path(ps, d, closed; cap=cap)
  pts = _offset_polygonal_vertices(ps, d, closed; join=join, cap=cap, miter_limit=miter_limit)
  closed ? closed_polygonal_path(pts) : open_polygonal_path(pts)
end


export offset

# Polygon combination

closest_vertices_indexes(pts1, pts2) =
  # This is a brute force method. There are better algorithms to do this.
  let min_dist = Inf,
      min_i = nothing,
      min_j = nothing
    for (i, pt1) in enumerate(pts1), (j, pt2) in enumerate(pts2)
      let dist = distance(pt1, pt2)
        if dist < min_dist
          min_dist = dist
          min_i = i
          min_j = j
        end
      end
    end
    min_i, min_j
  end

#=
`point_in_segment` decides whether `r` lies on the closed segment p--q
(endpoints included). The previous implementation compared the per-axis
ratios cx(pr)/cx(pq), cy(pr)/cy(pq), cz(pr)/cz(pq): for ANY axis-aligned
segment one component of pq is 0, turning a ratio into 0/0 (NaN) or k/0
(Inf), so a point genuinely ON a horizontal, vertical or planar segment
was rejected — silently corrupting `subtract_polygon_vertices` on the
axis-aligned edges that dominate building polygons. The ratio test also
ignored the segment bounds, accepting collinear points beyond an endpoint.

Instead we project r onto the segment's line by the parameter
  t = dot(pr, pq) / dot(pq, pq)  (t = 0 at p, t = 1 at q),
clamp t to [0, 1] so the foot of the projection stays on the *segment*
(this is what keeps the endpoints in and points past them out), and accept
r iff its distance to that foot is within coincidence_tolerance() — the
codebase's length-valued "same point" tolerance (1e-10 m), used here exactly
as in `coincident_path_location`. A zero-length segment (dot(pq, pq) == 0)
has no line to project onto, so r is in it iff it coincides with p.

See also: coincidence_tolerance, distance, collinear_segments.
=#
point_in_segment(r, p, q) =
  let pq = q-p,
      pr = r-p,
      d2 = dot(pq, pq)
    d2 == 0 ?
      distance(r, p) <= coincidence_tolerance() :
      let t = clamp(dot(pr, pq)/d2, 0, 1)
        distance(p + pq*t, r) <= coincidence_tolerance()
      end
  end

collinear_segments(p1, p2, q1, q2) =
  point_in_segment(q1, p1, p2) &&
  point_in_segment(q2, p1, p2)

collinear_vertices_indexes(pts1, pts2) =
  for (i1, p1) in enumerate(pts1)
    let i2 = i1%length(pts1)+1,
        p2 = pts1[i2]
      for (j1, q1) in enumerate(pts2)
        let j2 = j1%length(pts2)+1,
            q2 = pts2[j2]
          if collinear_segments(p1, p2, q1, q2)
            return (i1, j1)
          end
        end
      end
    end
  end

subtract_polygon_vertices(pts1, pts2) =
  let ij = collinear_vertices_indexes(pts1, pts2)
    isnothing(ij) ?
      inject_polygon_vertices_at_indexes(pts1, pts2, closest_vertices_indexes(pts1, pts2)) :
      splice_polygon_vertices_at_indexes(pts1, pts2, ij)
  end

inject_polygon_vertices_at_indexes(pts1, pts2, (i, j)) =
  [pts1[1:i]..., reverse([pts2[j:end]..., pts2[1:j]...])..., pts1[i:end]...]

export closest_vertices_indexes, inject_polygon_vertices_at_indexes, subtract_polygon_vertices

# Intersection

segments_intersection(p0, p1, p2, p3) =
  let pts = intersection_points(intersections(line_path(p0, p1), line_path(p2, p3); method=:local))
    isempty(pts) ? nothing : pts[1]
  end

lines_intersection(p0, p1, p2, p3) =
  let denom = (p3.y - p2.y)*(p1.x - p0.x) - (p3.x - p2.x)*(p1.y - p0.y)
    if denom == 0
      nothing
    else
      let u = ((p3.x - p2.x)*(p0.y - p2.y) - (p3.y - p2.y)*(p0.x - p2.x))/denom,
          v = ((p1.x - p0.x)*(p0.y - p2.y) - (p1.y - p0.y)*(p0.x - p2.x))/denom
        xy(p0.x + u*(p1.x - p0.x), p0.y + u*(p1.y - p0.y))
      end
    end
  end

#=
Parallelism tolerance.

Deciding whether two vectors (or two lines) are parallel cannot be a
bit-exact test, because numerical construction regularly produces
vectors that should be parallel but differ by a tiny rotation. The
canonical signal for non-parallelism is the angle θ between the
directions, measured through the cross product: |u × v| = |u||v| sin θ.

This tolerance is DIMENSIONLESS: it is a threshold on sin²θ, applied
in multiply form so no square roots or divisions are needed:

  |u × v|² < parallelism_tolerance() · |u|² · |v|²

or, equivalently, for the Gram determinant D = (u·u)(v·v) − (u·v)²
(which equals |u × v|²) used when solving the
closest-points-on-two-lines system:

  D < parallelism_tolerance() · (u·u) · (v·v)

The multiply form makes classification scale-invariant: a raw
cross-magnitude threshold (the previous semantics) silently classified
every pair of millimetre-scale segments as parallel (|u × v| ~ 1e-12 m²
even for perpendicular mm vectors), returning empty intersections.

The default 1e-8 as sin²θ means θ ≲ 1e-4 rad (≈ 0.0057°). Two reasons:
(a) the plane-plane and circle-plane kernels already compare
|n₁ × n₂|² of *unit* normals against this constant, i.e. they always
used it at this dimension, so 1e-8 needs no value change; (b)
conditioning — the transversal solution of two lines amplifies
positional noise by ~1/sin θ, which is 1e4 at the threshold, so
coincidence-tolerance-level coordinate noise (~1e-10 m) yields at most
~1e-6 m of solution error there. Tighten for precision work; loosen to
roughly (expected angular noise)² when ingesting data whose directions
are known to be sloppy.

A few legacy sites still compare a raw, dimensional cross/determinant
magnitude against this constant (the collinearity probe in
circle_from_three_points_2d, the clip denominator in
_clip_line_intersection, the top-view camera check in Backend.jl);
keeping the value at 1e-8 leaves them numerically intact — normalizing
them is a follow-up.

See also: coplanarity_tolerance (cosine deviation between unit
normals); zero_vector_tolerance (for checking that a single vector is
mathematically zero, as distinct from two vectors being parallel).
=#

"Dimensionless sin²θ threshold below which two vectors/lines are classified as parallel. `|cross(u, v)|² < parallelism_tolerance() * |u|² * |v|²`."
const parallelism_tolerance = Parameter(1e-8)
export parallelism_tolerance, nearest_point_from_lines, circle_from_three_points

#=
Maximum |1 − |cos θ|| between two surface normals for them to count as parallel
when deciding whether two circles/arcs are coplanar. This is a DIMENSIONLESS
cosine deviation (≈ θ²/2 for small θ) — NOT a length, and distinct from the
length-valued coincidence_tolerance used for the companion plane-distance test
in _circle_plane_relation.

The default 1e-7 (≈ 4.5e-4 rad ≈ 0.026°) is intentionally looser than the
length tolerance it replaced (1e-10): each normal is built by
`unitized(in_world(vz(1, cs)))`, and the repeated coordinate-frame
compositions accumulate ordinary Float64 rounding error (the transforms are
Float64 — Mat4f is an SMatrix{4,4,Float64}, the `f` is historical) that a
1e-10 length-valued tolerance would reject for genuinely coplanar geometry.
Tighten it for high-precision analytic work.

See also: parallelism_tolerance, coincidence_tolerance.
=#
"Max |1 - |cos θ|| between two normals for coplanarity classification (dimensionless cosine deviation)."
const coplanarity_tolerance = Parameter(1e-7)
export coplanarity_tolerance

nearest_point_from_lines(l0p0::Loc, l0p1::Loc, l1p0::Loc, l1p1::Loc) =
  let u = l0p1-l0p0,
      v = l1p1-l1p0,
      w = l0p0-l1p0,
      a = dot(u, u),
      b = dot(u, v),
      c = dot(v, v),
      d = dot(u, w),
      e = dot(v, w),
      D = a*c-b*b,
      #= `<=` (not `<`) so that exactly degenerate inputs (a·c == 0 forces
         D == 0 by Cauchy-Schwarz) take the guarded branch instead of
         dividing 0/0 in the transversal solution. The old fallback
         `(0.0, b > c ? d/b : e/c)` was dropped: it is NaN for a
         zero-length second line (b == c == 0 picks e/c = 0/0), and for
         genuinely parallel lines d/b equals e/c anyway, so projecting
         l0p0 onto line 1 (tc = e/c) covers it; degenerate lines project
         onto whichever line still has a direction. =#
      (sc, tc) = D <= parallelism_tolerance()*a*c ?
                #almost parallel or degenerate
                (c <= zero_vector_tolerance() ?
                   ((a <= zero_vector_tolerance() ? 0.0 : -d/a), 0.0) :
                   (0.0, e/c)) :
                ((b*e-c*d)/D, (a*e-b*d)/D),
      (p0, p1) = (l0p0+u*sc, l1p0+v*tc)
    intermediate_loc(p0, p1)
  end

circle_from_three_points_2d(v0::Loc, v1::Loc, v2::Loc) =
  let v1sv0 = v1-v0,
      v2sv0 = v2-v0,
      v2sv1 = v2-v1,
      v1pv0 = v1+(v0-u0()),
      v2pv0 = v2+(v0-u0()),
      a = v1sv0.x,
      b = v1sv0.y,
      c = v2sv0.x,
      d = v2sv0.y,
      e = a*v1pv0.x+b*v1pv0.y,
      f = c*v2pv0.x+d*v2pv0.y,
      g = 2.0*(a*v2sv1.y-b*v2sv1.x),
      iscolinear = abs(g) < parallelism_tolerance(),
      (cx, cy, dx, dy) = iscolinear ?
                         let minx = min(v0.x, v1.x, v2.x),
                             miny = min(v0.y, v1.y, v2.y),
                             maxx = max(v0.x, v1.x, v2.x),
                             maxy = max(v0.y, v1.y, v2.y),
                             x = (minx+maxx)/2,
                             y = (miny+maxy)/2
                           (x, y, x-minx, y-miny)
                         end :
                         let x = (d*e-b*f)/g,
                             y = (a*f-c*e)/g
                           (x, y, x-v0.x, y-v0.y)
                         end,
      radius_squared = dx*dx+dy*dy,
      radius = sqrt(radius_squared)
    (xy(cx, cy), radius)
  end

circle_from_three_points(p0::Loc, p1::Loc, p2::Loc) =
  let cs = cs_from_o_vx_vy(p0, p1-p0, p2-p0)
    with(current_cs, cs) do
      c, r = circle_from_three_points_2d(in_cs(p0, cs),
                                         in_cs(p1, cs),
                                         in_cs(p2, cs))
      (c, r)
    end
  end


#=
Collinearity tolerance.

Classifying three points as collinear cannot rely on an exact test:
points authored on a line may drift off it by fractions of a micrometre
through the usual transform-and-rounding churn, and subdivision
algorithms (path sampling, adaptive tessellation) depend on a stable
collinearity predicate to decide when to stop recursing.

We use the area of the triangle with the three points as vertices,
computed via Heron's formula from the three pairwise distances. This
has the convenient property of being scale-coherent: doubling all
three points' coordinates multiplies the area by four, matching user
intuition that "collinear" is a geometric, not numeric, judgement.

Compared against `triangle_area(|p0 pm|, |pm p1|, |p1 p0|)`, in metres²
(Khepri's canonical unit, squared). The default 1e-2 is deliberately
loose: 1 cm² is below the precision at which architectural designers
reason about alignment, and subdivision algorithms that halve the
triangle area at each step converge much faster with a tolerance at
this scale. Precision work may tighten it via
`with(collinearity_tolerance, 1e-6) do ... end`.

See also: parallelism_tolerance (for a stricter, vector-based
classification used when the three points come from line intersection
rather than path sampling).
=#

"Triangle area below which three points are classified as collinear. `triangle_area(a, b, c) < collinearity_tolerance()`. [metres²]"
const collinearity_tolerance = Parameter(1e-2)
export collinearity_tolerance

# Are the three points sufficiently collinear?
collinear_points(p0, pm, p1, tol=collinearity_tolerance()) =
  let a = distance(p0, pm),
      b = distance(pm, p1),
      c = distance(p1, p0)
    triangle_area(a, b, c) < tol
  end

#=
export sweep_path_with_path
sweep_path_with_path(path, profile) =
  let vertices = in_world.(path_frames(profile)),
      frames = path_frames(path)
      #show_cs.(frames, 0.1)
    surface_grid([[xyz(cx(p), cy(p), cz(p), frame.cs) for p in vertices]
                  for frame in frames],
                 is_closed_path(profile),
                 is_closed_path(path),
                 is_smooth_path(profile),
                 is_smooth_path(path))
  end
=#
export quad_grid, quad_grid_indexes
quad_grid(quad, points, closed_u, closed_v) =
  let pts = in_world.(points),
      si = size(pts, 1),
      sj = size(pts, 2)
    for i in 1:si-1
      for j in 1:sj-1
        quad(pts[i,j], pts[i+1,j], pts[i+1,j+1], pts[i,j+1])
      end
      if closed_v
        quad(pts[i,sj], pts[i+1,sj], pts[i+1,1], pts[i,1])
      end
    end
    if closed_u
      for j in 1:sj-1
        quad(pts[si,j], pts[1,j], pts[1,j+1], pts[si,j+1])
      end
      if closed_v
        quad(pts[si,sj], pts[1,sj], pts[1,1], pts[si,1])
      end
    end
  end

quad_grid_indexes(si, sj, closed_u, closed_v) =
  let idx(i,j) = (i-1)*sj+(j-1),
      idxs = Vector{Int}[],
      quad(a,b,c,d) = (push!(idxs, [a, b, d]); push!(idxs, [d, b, c]))
    for i in 1:si-1
      for j in 1:sj-1
        quad(idx(i,j), idx(i+1,j), idx(i+1,j+1), idx(i,j+1))
      end
      if closed_v
        quad(idx(i,sj), idx(i+1,sj), idx(i+1,1), idx(i,1))
      end
    end
    if closed_u
      for j in 1:sj-1
        quad(idx(si,j), idx(1,j), idx(1,j+1), idx(si,j+1))
      end
      if closed_v
        quad(idx(si,sj), idx(1,sj), idx(1,1), idx(si,1))
      end
    end
    idxs
  end

export illustrate_path
illustrate_path(path) =
  begin
    for (i, v) in enumerate(path_vertices(path))
      sphere(v, 0.01)
      text(string(i), v+vxy(0.05, 0.05), 0.1)
    end
    stroke(path)
  end

illustrate_expr(expr) =
  :(let p = $(esc(expr)); text($(string(expr)), p); p end)

export @illustrate
macro illustrate(exprs...)
  :(tuple($([illustrate_expr(expr) for expr in exprs]...)))
end
