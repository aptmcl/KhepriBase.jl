# SceneExpectations.jl — analytic, hand-derived expectations for the visual
# test corpus (VisualTests.jl), for Tier-1 "parse and measure" oracles.
#
# Each entry states, in closed form, what geometry a scene MUST contain in
# world space — derived from the scene source in VisualTests.jl by hand, NOT
# from any backend or from KhepriBase's tessellation code.  Backend artifact
# parsers (first consumer: KhepriTikZ/src/TikZParser.jl via
# KhepriTikZ/test/TikZOracle.jl) compare their measures against these values.
# See Docs/TestingStrategy.md step 4.
#
# This module is deliberately dependency-free (plain Julia math, no
# KhepriBase): an expectation that reused vpol/division/in_world would
# certify those functions with themselves.  Where a scene builds geometry
# from KhepriBase constructors, the construction is re-derived here as
# explicit cos/sin/range arithmetic — replicating the scene's OWN formulas
# (the program text in VisualTests.jl is the specification), not the
# library's implementation.
#
# NOT included in KhepriBase's runtests.jl: backend test suites include it
# directly, like VisualTests.jl.  Include sites guard with
# `isdefined(Main, :SceneExpectations) || include(...)` (same reason as
# Oracles.jl: a module must sit at top level).
#
# Expectation fields (all optional; consumers check the ones present):
#   circles    :: [(center, r, filled)]
#   points     :: [(x,y,z)]                 zero-extent arcs / point shapes
#   arcs       :: [(center, r, ai, af, filled)]   degrees; ai ∈ [0,360),
#                 af = ai + arc width (an angle INTERVAL: representation-
#                 independent, so an emitted -180:0 equals (180, 360))
#   rectangles :: [(lo, hi)]                normalized opposite corners
#   polylines  :: [(pts, closed)]           vertex sequences, world space
#   splines    :: [(pts, kind)]             control points verbatim
#   nodes      :: [(anchor, text)]          text anchor positions
#   counts     :: NamedTuple                extra statement counts (e.g.
#                                           triangle counts) not implied by
#                 the lists above
#   vertex_set :: [(x,y,z)]                 exact distinct-vertex set of all
#                                           polyline-like statements
#   aabb       :: (lo, hi)                  over statement coordinate points
#   exhaustive :: Bool                      scene contains NOTHING beyond
#                                           what the fields above imply
#   on_circles :: (r, centers)              vouch: every vertex must lie on
#                                           one of these circles
# All coordinates are (x, y, z) world-space Float64 triples.

module SceneExpectations

export expected, tikz_parseable, TIKZ_WAVE1, TIKZ_WAVE2, TIKZ_VOUCH

# ── Small helpers (plain math, mirroring scene-source formulas) ────────

xyz3(x, y, z=0.0) = (Float64(x), Float64(y), Float64(z))

# The corpus builds points as p + vpol(ρ, ϕ); vpol uses sincos, replicated
# here so expectation floats are bit-identical to the scene's pre-rounding
# values (comparison tolerance then only has to absorb artifact rounding).
add_pol((x, y, z), ρ, ϕ) =
  let (s, c) = sincos(ϕ)
    (x + ρ*c, y + ρ*s, z)
  end

# division(t0, t1, n[, include_last]) from KhepriBase/src/Utils.jl,
# replicated: n+1 range values, optionally dropping the last.
divis(t0, t1, n, include_last=true) =
  let vs = collect(range(t0, t1, length=n + 1))
    include_last ? vs : vs[1:n]
  end

circle_entry(c, r; filled=false) = (center=c, r=Float64(r), filled=filled)

#=
Arcs are stated as angle intervals, the representation the corpus's emitters
normalize to: a Khepri arc(c, r, ai, da) covers [ai+min(0,da), ai+max(0,da)],
whose start we reduce into [0,360) keeping the width.  Both this module and
the parser-side measure normalize the same way, so (-180:0) ≡ (180, 360).
=#
arc_entry(c, r, ai, da; filled=false) =
  let lo = rad2deg(ai + min(0.0, da)), w = rad2deg(abs(da)), lon = mod(lo, 360)
    (center=c, r=Float64(r), ai=lon, af=lon + w, filled=filled)
  end

rect_entry(c0, c1) =
  (lo=(min(c0[1], c1[1]), min(c0[2], c1[2]), min(c0[3], c1[3])),
   hi=(max(c0[1], c1[1]), max(c0[2], c1[2]), max(c0[3], c1[3])))

node_entry(anchor, text) = (anchor=anchor, text=text)

# ── Wave 1: full-measure scenes ────────────────────────────────────────

#= circles: circle(u0(), r) for r in 1:5 — five concentric circles at the
   origin. =#
circles_expectation() =
  (circles=[circle_entry(xyz3(0, 0), r) for r in 1:5], exhaustive=true)

#= polygons: regular_polygon(n, x(3i), 1) for (i, n) in enumerate(3:8) —
   an n-gon inscribed in the unit circle at center (3i, 0), first vertex at
   angle 0: vertex k is (3i + cos(2πk/n), sin(2πk/n)). =#
polygons_expectation() =
  (polylines=[(pts=[add_pol(xyz3(3i, 0), 1, 2π*k/n) for k in 0:n-1], closed=true)
              for (i, n) in enumerate(3:8)],
   exhaustive=true)

#= arcs: arc(u0(),5,0,π/2) → interval (0°,90°); arc(u0(),4,π/4,π) →
   (45°,225°); circle(u0(),3). =#
arcs_expectation() =
  (arcs=[arc_entry(xyz3(0, 0), 5, 0.0, π/2),
         arc_entry(xyz3(0, 0), 4, π/4, Float64(π))],
   circles=[circle_entry(xyz3(0, 0), 3)],
   exhaustive=true)

#= spline_curve: spline through five literal control points; Tier 1 asserts
   the control points verbatim, not the curve between them (Tier 3). =#
spline_curve_expectation() =
  (splines=[(pts=[xyz3(0, 0), xyz3(1, 2), xyz3(3, 1), xyz3(5, 3), xyz3(7, 0)],
             kind=:spline)],
   exhaustive=true)

#= rectangles: rectangle(xy(0,0),4,3), rectangle(xy(5,0),2,5),
   rectangle(xy(8,0),3,3) — corners (c, c+(dx,dy)). =#
rectangles_expectation() =
  (rectangles=[rect_entry(xyz3(0, 0), xyz3(4, 3)),
               rect_entry(xyz3(5, 0), xyz3(7, 5)),
               rect_entry(xyz3(8, 0), xyz3(11, 3))],
   exhaustive=true)

#= lines_and_polygons: one open 3-vertex polyline, one closed pentagon —
   vertices literal in the scene. =#
lines_and_polygons_expectation() =
  (polylines=[(pts=[xyz3(0, 0), xyz3(5, 0), xyz3(5, 5)], closed=false),
              (pts=[xyz3(6, 0), xyz3(10, 0), xyz3(10, 4), xyz3(8, 5), xyz3(6, 3)],
               closed=true)],
   exhaustive=true)

#= surface_circles (top view): two filled discs. =#
surface_circles_expectation() =
  (circles=[circle_entry(xyz3(0, 0), 3, filled=true),
            circle_entry(xyz3(8, 0), 2, filled=true)],
   exhaustive=true)

#= surface_rectangles (top view): two filled rectangles, (0,0)+5×3 and
   (6,0)+2×4.  A backend may triangulate them (TikZ does: 2 triangles per
   convex quad), but triangulating a polygon introduces NO new vertices, so
   the distinct-vertex set must be exactly the 8 corners; the triangle count
   pins the decomposition arity. =#
surface_rectangles_expectation() =
  (vertex_set=[xyz3(0, 0), xyz3(5, 0), xyz3(5, 3), xyz3(0, 3),
               xyz3(6, 0), xyz3(8, 0), xyz3(8, 4), xyz3(6, 4)],
   counts=(triangle=4,),
   aabb=(lo=xyz3(0, 0), hi=xyz3(8, 4)),
   exhaustive=true)

# ── Wave 2: parameter/count/anchor scenes ──────────────────────────────

#= circulosRaios / circulosRaiosPolar: the same three circles and three
   labels, built polar-first and cartesian-first respectively —
   pol(ρ, π/4) = (ρ/√2, ρ/√2):
     pol(4, π/4) = (2√2, 2√2) ≈ (2.828, 2.828)
     pol(6, π/4) = (3√2, 3√2) ≈ (4.243, 4.243)
   Labels sit at center + (Δx, 0) with Δx = 5, 2.5, 1.25. =#
circulos_raios_expectation() =
  let c1 = xyz3(0, 0),
      c2 = add_pol(xyz3(0, 0), 4, π/4),
      c3 = add_pol(xyz3(0, 0), 6, π/4)
    (circles=[circle_entry(c1, 4), circle_entry(c2, 2), circle_entry(c3, 1)],
     nodes=[node_entry((c1[1] + 5.0, c1[2], 0.0), "Raio: 4"),
            node_entry((c2[1] + 2.5, c2[2], 0.0), "Raio: 2"),
            node_entry((c3[1] + 1.25, c3[2], 0.0), "Raio: 1")],
     exhaustive=true)
  end

# circulosRaiosPolar builds the same centers as xy(√8,√8)/xy(√18,√18);
# identical world geometry, shared expectation derived per its own formulas.
circulos_raios_polar_expectation() =
  let c1 = xyz3(0, 0), c2 = xyz3(sqrt(8), sqrt(8)), c3 = xyz3(sqrt(18), sqrt(18))
    (circles=[circle_entry(c1, 4), circle_entry(c2, 2), circle_entry(c3, 1)],
     nodes=[node_entry((c1[1] + 5.0, c1[2], 0.0), "Raio: 4"),
            node_entry((c2[1] + 2.5, c2[2], 0.0), "Raio: 2"),
            node_entry((c3[1] + 1.25, c3[2], 0.0), "Raio: 1")],
     exhaustive=true)
  end

#= autocad_text: circle((2,3),1) plus the literal "AutoCAD" anchored at
   its text corner (1,1). =#
autocad_text_expectation() =
  (circles=[circle_entry(xyz3(2, 3), 1)],
   nodes=[node_entry(xyz3(1, 1), "AutoCAD")],
   exhaustive=true)

#= textoRodado / textoRodadoEscalado: every text is anchored at
   loc_from_o_phi(u0(), ϕ) — the origin of a frame ROTATED about the world
   origin — so all world anchors are exactly (0,0); rotation and scale live
   in the artifact's transform, not the anchor.  10 and 11 texts
   (division(0,2π,10,false) and division(0,0.9,10)). =#
texto_rodado_expectation(n) =
  (nodes=[node_entry(xyz3(0, 0), "Rodado") for _ in 1:n], exhaustive=true)

#= setas: norte(p, ro, alfa) draws
     seta: the open 5-vertex polyline
       p, a, a+vpol(σ,alfa+π−β), a+vpol(σ,alfa+π+β), a
       with a = p+vpol(ro,alfa), σ = ro/2, β = π/4;
     label: text_centered("N", loc_from_o_phi(p+vpol(1.2ro, alfa), alfa−π/2), ro/2).
   text_centered(str, c, h) anchors the text at
   c + (−len(str)·h·0.85/2, −h/2) in c's frame (Shapes.jl), so the world
   anchor is o + R(alfa−π/2)·(−0.425·h, −0.5·h) with len("N") = 1, h = ro/2.
   Five calls: p = (3k,0) k=0..4, ro = 2, alfa = π/2, π/4, π/6, π/8, π/16. =#
setas_expectation() =
  let entries = [(xyz3(3k, 0), 2.0, alfa)
                 for (k, alfa) in zip(0:4, [π/2, π/4, π/6, π/8, π/16])],
      seta(p, ro, alfa, σ, β) =
        let a = add_pol(p, ro, alfa)
          (pts=[p, a, add_pol(a, σ, alfa + π - β), add_pol(a, σ, alfa + π + β), a],
           closed=false)
        end,
      anchor(p, ro, alfa) =
        let o = add_pol(p, 1.2*ro, alfa),
            h = ro*0.5,
            (dx, dy) = (-1*h*0.85/2, -h/2),
            (s, c) = sincos(alfa - π/2)
          (o[1] + c*dx - s*dy, o[2] + s*dx + c*dy, 0.0)
        end
    (polylines=[seta(p, ro, alfa, ro/2.0, π/4) for (p, ro, alfa) in entries],
     nodes=[node_entry(anchor(p, ro, alfa), "N") for (p, ro, alfa) in entries],
     exhaustive=true)
  end

#= ovos: ovo(p, r0, r1, h) is the classical four-arc egg —
     alfa = 2·atan(r0−r1, h−r0−r1)
     r2   = (r0 − r1·cos alfa)/(1 − cos alfa)
     arc(p, r0, 0, −π)                       bottom half circle
     arc(p+(r0−r2, 0), r2, 0, alfa)          right flank
     arc(p+(r2−r0, 0), r2, π−alfa, alfa)     left flank
     arc(p+(0, (r2−r1)·sin alfa), r1, alfa, π−2·alfa)   top cap
   over the grid p = (30·r1, 7·h), r1 ∈ division(0.6,1.8,6) (7 values),
   h ∈ division(4.5,6.5,2) (3 values): 21 eggs × 4 arcs = 84 arcs. =#
ovos_expectation() =
  let arcs = NamedTuple[]
    for r1 in divis(0.6, 1.8, 6), h in divis(4.5, 6.5, 2)
      let r0 = 2.0,
          p = xyz3(r1*30, 7h),
          alfa = 2*atan(r0 - r1, h - r0 - r1),
          r2 = (r0 - r1*cos(alfa))/(1 - cos(alfa))
        push!(arcs, arc_entry(p, r0, 0.0, -Float64(π)))
        push!(arcs, arc_entry((p[1] + (r0 - r2), p[2], 0.0), r2, 0.0, alfa))
        push!(arcs, arc_entry((p[1] + (r2 - r0), p[2], 0.0), r2, π - alfa, alfa))
        push!(arcs, arc_entry((p[1], p[2] + (r2 - r1)*sin(alfa), 0.0), r1,
                              alfa, π - alfa - alfa))
      end
    end
    (arcs=arcs, exhaustive=true)
  end

#= espiralB / espiralOuro: the recursive spiral of shrinking arc steps,
   replicated verbatim from the scene source (the recursion IS the closed
   form; only p+vpol becomes explicit cos/sin).  A zero-width final arc
   (a_fin − a_ini exactly 0) degenerates to a point at p+vpol(r, a_ini) —
   geometrically, an arc of zero extent is a point. =#
function espiral!(arco!, p, r, a_ini, a_inc, a_fin, f)
  if a_fin - a_ini < a_inc
    arco!(p, r, a_ini, a_fin - a_ini)
  else
    arco!(p, r, a_ini, a_inc)
    espiral!(arco!, add_pol(p, r*(1 - f), a_ini + a_inc), r*f,
             a_ini + a_inc, a_inc, a_fin, f)
  end
end

push_arc_or_point!(arcs, points, p, r, ai, da) =
  da == 0 ? push!(points, add_pol(p, r, ai)) :
            push!(arcs, arc_entry(p, r, ai, da))

espiralB_expectation() =
  let arcs = NamedTuple[], points = NTuple{3,Float64}[], lines = NamedTuple[],
      arco!(p, r, a_ini, a_inc) =
        begin
          push_arc_or_point!(arcs, points, p, r, a_ini, a_inc)
          push!(lines, (pts=[p, add_pol(p, r, a_ini)], closed=false))
          push!(lines, (pts=[p, add_pol(p, r, a_ini + a_inc)], closed=false))
        end
    espiral!(arco!, xyz3(0, 0), 10.0, π/2, π/2, π*6, 0.9)
    espiral!(arco!, xyz3(20, 0), 10.0, π/2, π/2, π*6, 0.7)
    espiral!(arco!, xyz3(40, 0), 10.0, π/2, π/2, π*6, 0.5)
    #= π is passed UN-converted (Irrational), exactly as the scene passes
       `pi`: the recursion's `a_fin - a_ini < a_inc` termination compares a
       Float64 against it, and `x < π` is true even when x == Float64(π) —
       converting to Float64 here could run the recursion one step longer
       than the scene and shift every remaining arc. =#
    espiral!(arco!, xyz3(0, -20), 10.0, π/2, π, π*6, 0.8)
    espiral!(arco!, xyz3(20, -20), 10.0, π/2, π/2, π*6, 0.8)
    espiral!(arco!, xyz3(40, -20), 10.0, π/2, π/4, π*6, 0.8)
    espiral!(arco!, xyz3(0, -40), 10.0, π/2, π, π*6, 0.8)
    espiral!(arco!, xyz3(20, -40), 10.0, π/2, π/2, π*6, 0.8^(1//2))
    espiral!(arco!, xyz3(40, -40), 10.0, π/2, π/4, π*6, 0.8^(1//4))
    (arcs=arcs, points=points, polylines=lines, exhaustive=true)
  end

espiralOuro_expectation() =
  let arcs = NamedTuple[], points = NTuple{3,Float64}[], rects = NamedTuple[],
      arco!(p, r, a_ini, a_inc) =
        begin
          push_arc_or_point!(arcs, points, p, r, a_ini, a_inc)
          push!(rects, rect_entry(p, add_pol(p, sqrt(2)*r, a_ini + a_inc/2)))
        end
    espiral!(arco!, xyz3(40, 0), 1.0, π/2, π/2, π*6, 1.618)
    (arcs=arcs, points=points, rectangles=rects, exhaustive=true)
  end

# ── Vouch-only: stale-golden extrusion scenes (3D views) ───────────────

#= extrusaoSuperficie / extrusaoSolido: a literal 11-gon extruded by
   height 1 along z.  Whatever the wall/cap decomposition (triangulated
   walls, fan caps, native polygons), decomposing a prism over a fixed
   polygon introduces NO vertices beyond the polygon's own at z = 0 and
   z = 1, and painter's order cannot change a SET — so the distinct-vertex
   set must equal these 22 points exactly. =#
const EXTRUSAO_BASE =
  [xyz3(0, 2), xyz3(0, 5), xyz3(5, 3), xyz3(13, 3), xyz3(13, 6), xyz3(3, 6),
   xyz3(6, 7), xyz3(15, 7), xyz3(15, 2), xyz3(1, 2), xyz3(1, 0)]

extrusao_expectation() =
  (vertex_set=[(p[1], p[2], Float64(z)) for p in EXTRUSAO_BASE for z in (0, 1)],)

#= florCirculosExtrudidos: extrusion(surface_circle(), vsph(20, ϕ, ψ)) for
   ψ ∈ division(π/8, π/2−π/16, 5) (6 values) and ϕ ∈ division(0, 2π, 16,
   false) (16 values) — 96 oblique cylinders sharing one base disc.  Every
   artifact vertex must lie on the shared bottom circle (origin, r=1, z=0)
   or on one of the 96 top circles (center vsph(20,ϕ,ψ) =
   (20 cosϕ sinψ, 20 sinϕ sinψ, 20 cosψ), r=1, in its own z plane): an
   extrusion translate maps the base ring vertex-for-vertex, so no
   decomposition vertex can leave the two rings.  The flat-cap sampling
   starts at angle 0, so each circle attains its ±x/±y extremes exactly and
   the vertex AABB equals the circle-set AABB. =#
flor_circulos_expectation() =
  let centers = vcat([xyz3(0, 0, 0)],
                     [let (sψ, cψ) = sincos(ψ), (sϕ, cϕ) = sincos(ϕ)
                        (20*cϕ*sψ, 20*sϕ*sψ, 20*cψ)
                      end
                      for ψ in divis(π/8, π/2 - π/16, 5)
                      for ϕ in divis(0, 2π, 16, false)]),
      r = 1.0
    (on_circles=(r=r, centers=centers),
     aabb=(lo=(minimum(c[1] for c in centers) - r,
               minimum(c[2] for c in centers) - r,
               minimum(c[3] for c in centers)),
           hi=(maximum(c[1] for c in centers) + r,
               maximum(c[2] for c in centers) + r,
               maximum(c[3] for c in centers))))
  end

# ── Registry ───────────────────────────────────────────────────────────

#= Wave membership mirrors the Tier-1 TikZ plan in PLANS.md: wave 1 =
   full-measure 2D scenes, wave 2 = parameter/count/anchor scenes, vouch =
   stale 3D extrusion goldens assessed by a standalone entry point only.
   predioSinusoidal is registered with NO expectation: its towers
   triangulate spline-sampled walls, which have no closed-form vertex set —
   recorded honestly rather than approximated. =#
const TIKZ_WAVE1 =
  ["circles", "polygons", "arcs", "spline_curve", "rectangles",
   "lines_and_polygons", "surface_circles", "surface_rectangles"]
const TIKZ_WAVE2 =
  ["circulosRaios", "circulosRaiosPolar", "ovos", "espiralB", "espiralOuro",
   "setas", "autocad_text", "textoRodado", "textoRodadoEscalado"]
const TIKZ_VOUCH =
  ["extrusaoSuperficie", "extrusaoSolido", "florCirculosExtrudidos",
   "predioSinusoidal"]

const EXPECTATIONS = Dict{String,Any}(
  "circles" => circles_expectation,
  "polygons" => polygons_expectation,
  "arcs" => arcs_expectation,
  "spline_curve" => spline_curve_expectation,
  "rectangles" => rectangles_expectation,
  "lines_and_polygons" => lines_and_polygons_expectation,
  "surface_circles" => surface_circles_expectation,
  "surface_rectangles" => surface_rectangles_expectation,
  "circulosRaios" => circulos_raios_expectation,
  "circulosRaiosPolar" => circulos_raios_polar_expectation,
  "autocad_text" => autocad_text_expectation,
  "textoRodado" => () -> texto_rodado_expectation(10),
  "textoRodadoEscalado" => () -> texto_rodado_expectation(11),
  "setas" => setas_expectation,
  "ovos" => ovos_expectation,
  "espiralB" => espiralB_expectation,
  "espiralOuro" => espiralOuro_expectation,
  "extrusaoSuperficie" => extrusao_expectation,
  "extrusaoSolido" => extrusao_expectation,
  "florCirculosExtrudidos" => flor_circulos_expectation,
)

"Analytic expectation for scene `name`, or `nothing` when none is defined."
expected(name) =
  haskey(EXPECTATIONS, name) ? EXPECTATIONS[name]() : nothing

"True when scene `name` has a full validate-time expectation (waves 1+2)."
tikz_parseable(name) = name in TIKZ_WAVE1 || name in TIKZ_WAVE2

end # module SceneExpectations
