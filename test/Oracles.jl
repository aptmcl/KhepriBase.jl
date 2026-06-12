# Oracles.jl — backend-independent geometric ground truth for Tier 0 tests.
#
# These helpers compute geometric quantities (lengths, axis-aligned bounding
# boxes, signed areas, mesh-topology predicates) directly from KhepriBase
# geometry in pure Julia, with NO backend involved. They are the single source
# of expected values that Tier 0 — and, later, higher tiers via a swapped
# measurement function — asserts against. See Docs/TestingStrategy.md.
#
# Include sites guard with `isdefined(Main, :Oracles) || include(...)` so this
# file can be referenced by several test files without redefining the module.
# (A `module` must be at top level — it cannot be wrapped in an `if` guard the
# way TestMockBackend.jl wraps its struct/function definitions.)

module Oracles

using KhepriBase

export world_xyz, sampled_world_aabb, aabb_shift, aabb_isapprox,
       signed_area, scale_points, reflect_x,
       face_area, face_normal, face_indices_valid, has_repeated_face_indices,
       has_degenerate_faces, planar_normals_consistent,
       MEASURE_ATOL, AABB_SAMPLES

#= Residual budget for backend-free geometry.

   Tier 0 quantities are computed in Float64 with only a handful of arithmetic
   operations on coordinates in the unit-to-thousands range, so residuals sit a
   few ulps above machine epsilon scaled by magnitude. 1e-9 is far below any
   geometric error we would care about (sub-micron at architectural scale) yet
   comfortably above rounding noise. Override per-assertion when a particular
   computation (e.g. spline arc-length quadrature) is legitimately coarser, or
   scale it by the magnitude of large results.

   See also: AABB_SAMPLES. =#
const MEASURE_ATOL = 1e-9
"Default absolute tolerance for backend-free geometric comparisons."
MEASURE_ATOL

#= How densely a path is sampled to bound it.

   The translate/scale AABB invariants are EXACT per sample — the same sample
   set is transformed identically on both sides of the assertion — so this
   count does not affect them. It only governs how closely a sampled AABB
   approaches the true extrema of a smooth curve for analytic-anchor
   comparisons. 64 matches the default curve-flattening density
   (path_smoothness_segments). =#
const AABB_SAMPLES = 64
"Number of parameter samples used to bound a path."
AABB_SAMPLES

"World-coordinate (x, y, z) tuple of a location."
world_xyz(p) = let q = in_world(p); (q.x, q.y, q.z) end

#= Axis-aligned bounding box of a path, as (lo, hi) world-coordinate triples,
   obtained by sampling location_at over the path's own domain. Used for the
   translation-equivariance invariant, where sampling error cancels exactly. =#
function sampled_world_aabb(path; n::Int=AABB_SAMPLES)
  (t0, t1) = KhepriBase.path_domain(path)
  pts = [world_xyz(location_at(path, t)) for t in range(t0, t1; length=n)]
  lo = (minimum(p[1] for p in pts), minimum(p[2] for p in pts), minimum(p[3] for p in pts))
  hi = (maximum(p[1] for p in pts), maximum(p[2] for p in pts), maximum(p[3] for p in pts))
  (lo, hi)
end

"Shift an (lo, hi) AABB by a vector."
aabb_shift((lo, hi), v) =
  ((lo[1]+v.x, lo[2]+v.y, lo[3]+v.z), (hi[1]+v.x, hi[2]+v.y, hi[3]+v.z))

"Approximate equality of two (lo, hi) AABBs."
aabb_isapprox(a, b; atol=MEASURE_ATOL) =
  all(isapprox.(a[1], b[1]; atol=atol)) && all(isapprox.(a[2], b[2]; atol=atol))

#= Signed polygon area (shoelace): CCW positive, CW negative.

   Sign-bearing — unlike the public abs() polygon_area — so it detects winding
   and handedness errors that an absolute-area oracle silently certifies as
   correct. Named locally so the (internal) source symbol is referenced in one
   place. =#
signed_area(pts) = KhepriBase._polygon_signed_area(pts)

#= Scale points about a center in the world frame. Lets tests assert that
   signed area scales by k^2 and lengths by k, independent of path
   parameterization (which would otherwise confound a per-sample comparison). =#
scale_points(pts, k::Real, center=u0()) =
  let c = in_world(center)
    [c + (in_world(p) - c)*k for p in pts]
  end

#= Reflect points across the world YZ plane (negate x). Flips handedness, so a
   correct signed area must negate while preserving magnitude. =#
reflect_x(pts) = [let q = in_world(p); xyz(-q.x, q.y, q.z) end for p in pts]

## ── Mesh topology predicates (faces are 0-indexed into vertices) ──────────

"Geometric area of a (possibly quad) face, triangulated as a fan from its first vertex."
face_area(m, f) =
  let vs = [in_world(m.vertices[i+1]) for i in f]
    sum(let n = cross(vs[k]-vs[1], vs[k+1]-vs[1]); 0.5*sqrt(max(dot(n, n), 0.0)) end
        for k in 2:length(vs)-1; init=0.0)
  end

"Normal of a face, from the first three vertices (un-normalized)."
face_normal(m, f) =
  let a = in_world(m.vertices[f[1]+1]),
      b = in_world(m.vertices[f[2]+1]),
      c = in_world(m.vertices[f[3]+1])
    cross(b-a, c-a)
  end

"True iff every face index references an existing vertex (0-indexed)."
face_indices_valid(m) = all(f -> all(i -> 0 <= i < length(m.vertices), f), m.faces)

"True iff any face repeats a vertex index (degenerate connectivity)."
has_repeated_face_indices(m) = any(f -> length(unique(f)) != length(f), m.faces)

"True iff any face encloses ~zero geometric area (sliver / collapsed)."
has_degenerate_faces(m; atol=MEASURE_ATOL) = any(f -> face_area(m, f) <= atol, m.faces)

#= Winding/normal consistency for a PLANAR patch.

   On a flat surface every face normal must be parallel and same-facing; a
   single flipped winding shows up as a reversed normal. Returns true iff all
   non-degenerate face normals agree in direction. Only meaningful for planar
   input (curved patches have legitimately varying normals). =#
function planar_normals_consistent(m; atol=1e-6)
  ns = [face_normal(m, f) for f in m.faces if face_area(m, f) > atol]
  isempty(ns) && return true
  n0 = ns[1]; l0 = sqrt(dot(n0, n0))
  l0 <= atol && return false
  all(n -> let l = sqrt(dot(n, n))
             l > atol && dot(n, n0)/(l*l0) > 1 - 1e-6
           end, ns)
end

end # module Oracles
