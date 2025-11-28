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

offset(path, d::Real) = d == 0 ? path : nonzero_offset(path, d)
nonzero_offset(path::RectangularPath, d::Real) =
  rectangular_path(add_xy(path.corner, d, d), path.dx - 2d, path.dy - 2d)
nonzero_offset(path::OpenPolygonalPath, d::Real) =
  open_polygonal_path(offset_vertices(path.vertices, d, false))
nonzero_offset(path::ClosedPolygonalPath, d::Real) =
  closed_polygonal_path(offset_vertices(path.vertices, d, true))
nonzero_offset(path::CircularPath, d::Real) =
  circular_path(path.center, path.radius - d)
nonzero_offset(path::ArcPath, d::Real) =
  arc_path(path.center, path.radius - d, path.start_angle, path.amplitude)
nonzero_offset(path::ClosedPathSequence, d::Real) =
  ClosedPathSequence([nonzero_offset(path, d) for path in path.paths])


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

point_in_segment(r, p, q) =
  let pr = r-p,
      pq = q-p,
      rx = cx(pr)/cx(pq),
      ry = cy(pr)/cy(pq)
    isapprox(rx, ry) && isapprox(ry, cz(pr)/cz(pq))
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

polygon_centroid(pts) =
  let pts = ensure_no_repeated_locations(pts),
      p0 = pts[1],
      cs = p0.cs, # Assume planar polygon defined by first point's CS or World if simpler
      # For robustness, we should compute a best fit plane or use the normal if available.
      # But sticking to existing CS is standard in Khepri.
      # However, if the CS is world but the polygon is vertical, x/y projection fails.
      # Let's project to the local CS of the first point if it's not World,
      # or compute 3D centroid.
      # 3D Centroid of a planar polygon is the weighted sum of triangle centroids.
      centroid = vxyz(0, 0, 0, world_cs),
      area_sum = 0.0,
      n = length(pts)
    for i in 2:n-1
      let p1 = pts[i],
          p2 = pts[i+1],
          tri_center = (p0 + p1 + p2) / 3.0,
          tri_area = triangle_area(distance(p0, p1), distance(p1, p2), distance(p2, p0)) # Approximate
          # Better to use cross product area for signed area if normal is known,
          # but for convex or simple polygons, sum of absolute areas from a vertex might fail for concave.
          # The standard way is using cross products.
          v1 = p1 - p0,
          v2 = p2 - p0,
          cross_prod = cross(v1, v2),
          tri_area_vec = norm(cross_prod) / 2.0
        centroid += (p0 + p1 + p2) * tri_area_vec / 3.0
        area_sum += tri_area_vec
      end
    end
    area_sum > 1e-9 ? centroid / area_sum : p0
  end

is_coplanar(pts::Locs, tol=1e-9) =
  length(pts) <= 3 ||
  let p0 = pts[1],
      n = vz(0)
    for i in 2:length(pts)-1
      let v1 = pts[i] - p0,
          v2 = pts[i+1] - p0,
          cross_prod = cross(v1, v2)
        if norm(cross_prod) > tol
          n = unitized(cross_prod)
          break
        end
      end
    end
    norm(n) < tol ? true : all(p -> abs(dot(p - p0, n)) < tol, pts)
  end

# Intersection

segments_intersection(p0, p1, p2, p3) =
  let denom = (p3.y - p2.y)*(p1.x - p0.x) - (p3.x - p2.x)*(p1.y - p0.y)
    if denom == 0
      nothing
    else
      let u = ((p3.x - p2.x)*(p0.y - p2.y) - (p3.y - p2.y)*(p0.x - p2.x))/denom,
          v = ((p1.x - p0.x)*(p0.y - p2.y) - (p1.y - p0.y)*(p0.x - p2.x))/denom
        if 0 <= u <= 1 && 0 <= v <= 1
          xy(p0.x + u*(p1.x - p0.x), p0.y + u*(p1.y - p0.y))
        else
          nothing
        end
      end
    end
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

export epsilon, nearest_point_from_lines, circle_from_three_points, incircle, fit_line, fit_circle, is_coplanar, polygon_centroid
const epsilon = Parameter(1e-8)

incircle(p1, p2, p3) =
  let a = distance(p2, p3),
      b = distance(p1, p3),
      c = distance(p1, p2),
      s = (a + b + c) / 2,
      r = sqrt((s - a) * (s - b) * (s - c) / s),
      center = (p1 * a + p2 * b + p3 * c) / (a + b + c)
    (center, r)
  end

fit_line(pts::Locs) =
  let n = length(pts),
      mean_p = sum(pts) / n,
      dx = sum(map(p -> (p.x - mean_p.x)^2, pts)),
      dy = sum(map(p -> (p.y - mean_p.y)^2, pts))
    if dx < 1e-9 # Vertical line
      (Inf, mean_p.x)
    else
      let m = sum(map(p -> (p.x - mean_p.x) * (p.y - mean_p.y), pts)) / dx,
          c = mean_p.y - m * mean_p.x
        (m, c)
      end
    end
  end

fit_circle(pts::Locs) =
  # Project points to 2D using the CS of the first point or a best fit plane
  let cs = length(pts) >= 3 ?
             cs_from_o_vx_vy(pts[1], pts[2]-pts[1], pts[3]-pts[1]) :
             pts[1].cs,
      pts_2d = [in_cs(p, cs) for p in pts],
      n = length(pts),
      sum_x = sum(p -> p.x, pts_2d),
      sum_y = sum(p -> p.y, pts_2d),
      sum_x2 = sum(p -> p.x^2, pts_2d),
      sum_y2 = sum(p -> p.y^2, pts_2d),
      sum_xy = sum(p -> p.x * p.y, pts_2d),
      sum_x3 = sum(p -> p.x^3, pts_2d),
      sum_y3 = sum(p -> p.y^3, pts_2d),
      sum_xy2 = sum(p -> p.x * p.y^2, pts_2d),
      sum_x2y = sum(p -> p.x^2 * p.y, pts_2d),
      C = n * sum_x2 - sum_x^2,
      D = n * sum_xy - sum_x * sum_y,
      E = n * sum_x3 + n * sum_xy2 - (sum_x2 + sum_y2) * sum_x,
      G = n * sum_y2 - sum_y^2,
      H = n * sum_x2y + n * sum_y3 - (sum_x2 + sum_y2) * sum_y,
      a = (H * D - E * G) / (C * G - D^2),
      b = (H * C - E * D) / (D^2 - G * C),
      c = -(a * sum_x + b * sum_y + sum_x2 + sum_y2) / n,
      cx = -a / 2,
      cy = -b / 2,
      r = sqrt(max(0, cx^2 + cy^2 - c))
    (in_cs(xy(cx, cy), cs), r)
  end

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
      (sc, tc) = D < epsilon() ?
                #almost parallel
                (0.0, b > c ? d/b : e/c) :
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
      iscolinear = abs(g) < 1e-8,
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


const collinearity_tolerance = Parameter(1e-2)
# are the three points sufficiently collinear?
collinear_points(p0, pm, p1, epsilon=collinearity_tolerance()) =
  let a = distance(p0, pm),
      b = distance(pm, p1),
      c = distance(p1, p0)
    triangle_area(a, b, c) < epsilon
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
        quad(pts[si,j], pts[1,j], pts[si,j+1], pts[si,j+1])
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
