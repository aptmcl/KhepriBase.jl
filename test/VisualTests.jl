# VisualTests.jl — Shared visual regression test module for Khepri backends
#
# Usage in a backend's test/runtests.jl:
#
#   include(joinpath(dirname(pathof(KhepriBase)), "..", "test", "VisualTests.jl"))
#   using .VisualTests
#
#   run_visual_tests(my_backend,
#     golden_dir = joinpath(@__DIR__, "golden"),
#     reset! = () -> begin reset_my_backend(); delete_all_shapes(); backend(my_backend) end,
#     compare = text_compare,
#   )
#
# Golden file workflow:
#   - If a golden file does not exist, the test creates it automatically.
#   - If a golden file exists, the test generates a new result and compares it.
#   - To regenerate golden files, delete them and re-run the tests.
#
# Backend setup requirements for consistent golden files:
#
# 1. Render size: rendering_with sets width/height to 16:9 FullHD ratio (1920x1080).
#    All backends that use render_width()/render_height() will respect this.
#
# 2. Multi-viewport backends (Rhino, AutoCAD):
#    - set_view() automatically maximizes the Perspective viewport in Rhino
#    - set_view_top() automatically maximizes the Top viewport in Rhino
#    - Use setup_backend callback for any additional one-time configuration

module VisualTests

using KhepriBase
using Test
export run_visual_tests, text_compare, pixel_diff_compare

# Detect unimplemented backend operations (thrown as ErrorException wrapping
# UnimplementedBackendOperationException, or as UndefVarError for undefined b_* ops)
is_unimplemented(e::ErrorException) =
  occursin("UnimplementedBackendOperationException", e.msg)
is_unimplemented(e::UndefVarError) =
  startswith(string(e.var), "b_")
is_unimplemented(::Any) = false

# ── Helpers ──────────────────────────────────────────────────────────

zoom_2d_top() = begin
  set_view_top()
  zoom_extents()
end

# ── Test categories ────────────────────────────────────────────────────

const CATEGORIES = [
  :primitives_2d,
  :primitives_3d,
  :surfaces,
  :extrusions,
  :csg,
  :parametric,
]

# ── Curated test examples ─────────────────────────────────────────────
# Each: (name, category, function).
# The function: creates shapes, sets view, calls raw_view(name), returns path.

const VISUAL_TESTS = Tuple{String, Symbol, Function}[]

register_test(fn, name, category) =
  push!(VISUAL_TESTS, (name, category, ()->(fn(); raw_view(name))))

# ── :primitives_2d ─────────────────────────────────────────────────────

register_test("circles", :primitives_2d) do
  for r in 1:5
    circle(u0(), r)
  end
  zoom_2d_top()
end

register_test("polygons", :primitives_2d) do
  for (i, n) in enumerate(3:8)
    regular_polygon(n, x(i * 3), 1)
  end
  zoom_2d_top()
end

register_test("arcs", :primitives_2d) do
  arc(u0(), 5, 0, pi/2)
  arc(u0(), 4, pi/4, pi)
  circle(u0(), 3)
  zoom_2d_top()
end

register_test("spline_curve", :primitives_2d) do
  spline([xy(0, 0), xy(1, 2), xy(3, 1), xy(5, 3), xy(7, 0)])
  zoom_2d_top()
end

register_test("rectangles", :primitives_2d) do
  rectangle(xy(0, 0), 4, 3)
  rectangle(xy(5, 0), 2, 5)
  rectangle(xy(8, 0), 3, 3)
  zoom_2d_top()
end

register_test("lines_and_polygons", :primitives_2d) do
  line(xy(0, 0), xy(5, 0), xy(5, 5))
  polygon(xy(6, 0), xy(10, 0), xy(10, 4), xy(8, 5), xy(6, 3))
  zoom_2d_top()
end

register_test("autocad_text", :primitives_2d) do
  circle(xy(2, 3), 1)
  text("AutoCAD", xy(1, 1), 1)
  zoom_2d_top()
end

register_test("circulosRaios", :primitives_2d) do
  circle(pol(0, 0), 4)
  text("Raio: 4", pol(0, 0)+vpol(5, 0), 1)
  circle(pol(4, pi/4), 2)
  text("Raio: 2", pol(4, pi/4)+vpol(2.5, 0), 0.5)
  circle(pol(6, pi/4), 1)
  text("Raio: 1", pol(6, pi/4)+vpol(1.25, 0), 0.25)
  zoom_2d_top()
end

register_test("circulosRaiosPolar", :primitives_2d) do
  circle(xy(0, 0), 4)
  text("Raio: 4", xy(0, 0)+vxy(5, 0), 1)
  circle(xy(sqrt(8), sqrt(8)), 2)
  text("Raio: 2", xy(sqrt(8), sqrt(8))+vxy(2.5, 0), 0.5)
  circle(xy(sqrt(18), sqrt(18)), 1)
  text("Raio: 1", xy(sqrt(18), sqrt(18))+vxy(1.25, 0), 0.25)
  zoom_2d_top()
end

register_test("setas", :primitives_2d) do
  seta(p, ro, alfa, sigma, beta) =
    line(p,
         p+vpol(ro, alfa),
         p+vpol(ro, alfa)+vpol(sigma, alfa+pi+-beta),
         p+vpol(ro, alfa)+vpol(sigma, alfa+pi+beta),
         p+vpol(ro, alfa))

  norte(p, ro, alfa) =
    begin
      seta(p, ro, alfa, ro/2.0, pi/4)
      let p = p+vpol(1.2*ro, alfa)
        text_centered("N", loc_from_o_phi(p, alfa-pi/2), ro*0.5)
      end
    end
  norte(xy(0, 0), 2, pi/2)
  norte(xy(3, 0), 2, pi/4)
  norte(xy(6, 0), 2, pi/6)
  norte(xy(9, 0), 2, pi/8)
  norte(xy(12, 0), 2, pi/16)
  zoom_2d_top()
end

register_test("espiralB", :primitives_2d) do
  espiral(p, r, a_ini, a_inc, a_fin, f) =
    if a_fin-a_ini < a_inc
      arco_espiral(p, r, a_ini, a_fin-a_ini)
    else
      arco_espiral(p, r, a_ini, a_inc)
      espiral(p+vpol(r*(1-f), a_ini+a_inc), r*f, a_ini+a_inc, a_inc, a_fin, f)
    end
  arco_espiral(p, r, a_ini, a_inc) =
    begin
      arc(p, r, a_ini, a_inc)
      line(p, p+vpol(r, a_ini))
      line(p, p+vpol(r, a_ini+a_inc))
    end

  espiral(xy(0, 0), 10, pi/2, pi/2, pi*6, 0.9)
  espiral(xy(20, 0), 10, pi/2, pi/2, pi*6, 0.7)
  espiral(xy(40, 0), 10, pi/2, pi/2, pi*6, 0.5)
  espiral(xy(0, -20), 10, pi/2, pi, pi*6, 0.8)
  espiral(xy(20, -20), 10, pi/2, pi/2, pi*6, 0.8)
  espiral(xy(40, -20), 10, pi/2, pi/4, pi*6, 0.8)
  espiral(xy(0, -40), 10, pi/2, pi, pi*6, 0.8)
  espiral(xy(20, -40), 10, pi/2, pi/2, pi*6, 0.8^(1//2))
  espiral(xy(40, -40), 10, pi/2, pi/4, pi*6, 0.8^(1//4))
  zoom_2d_top()
end


register_test("espiralOuro", :primitives_2d) do
  espiral(p, r, a_ini, a_inc, a_fin, f) =
    if a_fin-a_ini < a_inc
      arco_espiral(p, r, a_ini, a_fin-a_ini)
    else
      arco_espiral(p, r, a_ini, a_inc)
      espiral(p+vpol(r*(1-f), a_ini+a_inc), r*f, a_ini+a_inc, a_inc, a_fin, f)
    end

  arco_espiral(p, r, a_ini, a_inc) =
    begin
      arc(p, r, a_ini, a_inc)
      rectangle(p, p+vpol(sqrt(2)*r, a_ini+a_inc/2))
    end

  espiral(xy(40, 0), 1, pi/2, pi/2, pi*6, 1.618)
  zoom_2d_top()
end
  
register_test("ovos", :primitives_2d) do
  ovo(p, r0, r1, h) =
    begin
      alfa = 2*atan(r0-r1, h-r0-r1)
      r2 = (r0-r1*cos(alfa))/(1-cos(alfa))
      arc(p, r0, 0, -pi)
      arc(p+vx(r0-r2), r2, 0, alfa)
      arc(p+vx(r2-r0), r2, pi-alfa, alfa)
      arc(p+vy((r2-r1)*sin(alfa)), r1, alfa, pi-alfa-alfa)
    end
  map(r1 -> map(h -> ovo(xy(r1*30, 7*h), 2, r1, h), division(4.5, 6.5, 2)),
      division(0.6, 1.8, 6))
  zoom_2d_top()
end

register_test("textoRodado", :primitives_2d) do
  for fraccao in division(0, 2*pi, 10, false)
    text("Rodado", loc_from_o_phi(u0(), fraccao), 10)
  end
  zoom_2d_top()
end


register_test("textoRodadoEscalado", :primitives_2d) do
  for fraccao in division(0, 0.9, 10)
    text("Rodado", loc_from_o_phi(u0(), fraccao*2*pi), (1-fraccao)*10)
  end
  zoom_2d_top()
end

# ── :primitives_3d ─────────────────────────────────────────────────────

register_test("boxes", :primitives_3d) do
  box(xyz(1, 1, 0), xyz(0.5, 0.5, 0.5))
  box(xyz(1.7, 1.7, 0), xyz(1.5, 1.5, 0.5))
  box(xyz(2, 2, 0), xyz(2.5, 2.5, 0.5))
  set_view(xyz(-0.498875, -0.825617, 0.681738),
           xyz(0.484146, 0.459262, 0.329734), 50)
end

register_test("spheres", :primitives_3d) do
  sphere(xyz(1, 2, 3), 4)
  sphere(xyz(5, 2, 3), 2)
  sphere(xyz(7, 2, 3), 1)
  set_view(xyz(71.5486, 50.8126, 50.9644),
           xyz(-243.179, -168.982, -167.24), 197)
end

register_test("solids", :primitives_3d) do
  box(xyz(2, 1, 1), xyz(3, 4, 5))
  cone(xyz(6, 0, 0), 1, xyz(8, 1, 5))
  cone_frustum(xyz(11, 1, 0), 2, xyz(10, 0, 5), 1)
  sphere(xyz(8, 4, 5), 2)
  cylinder(xyz(8, 7, 0), 1, xyz(6, 8, 7))
  regular_pyramid(5, xyz(-2, 1, 0), 1, 0, xyz(2, 7, 7))
  torus(xyz(14, 6, 5), 2, 1)
  set_view(xyz(8.057, -23.2615, 9.5719),
           xyz(6.30023, 7.5729, 1.71575), 50)
end

register_test("cylinders", :primitives_3d) do
  cylinder(xyz(0, 0, 0), 1, 5)
  cylinder(xyz(4, 0, 0), 0.5, 3)
  cylinder(xyz(8, 0, 0), 2, 1)
  set_view(xyz(20, 20, 15), xyz(4, 0, 1.5), 50)
  zoom_extents()
end

register_test("cones", :primitives_3d) do
  cone(xyz(0, 0, 0), 2, 4)
  cone_frustum(xyz(6, 0, 0), 2, 4, 1)
  cone_frustum(xyz(12, 0, 0), 1, 3, 2)
  set_view(xyz(20, 20, 15), xyz(6, 0, 2), 50)
  zoom_extents()
end

register_test("pyramids", :primitives_3d) do
  regular_pyramid(3, xyz(0, 0, 0), 2, 0, 3)
  regular_pyramid(4, xyz(6, 0, 0), 2, 0, 3)
  regular_pyramid(6, xyz(12, 0, 0), 2, 0, 3)
  set_view(xyz(20, 20, 15), xyz(6, 0, 1.5), 50)
  zoom_extents()
end

register_test("tori", :primitives_3d) do
  torus(xyz(0, 0, 0), 3, 1)
  torus(xyz(10, 0, 0), 2, 0.5)
  torus(xyz(18, 0, 0), 4, 0.3)
  set_view(xyz(20, 20, 15), xyz(9, 0, 0), 50)
  zoom_extents()
end

register_test("prisms", :primitives_3d) do
  regular_pyramid_frustum(3, xyz(0, 0, 0), 0.4, 0, xyz(0, 0, 5), 0.4, true)
  regular_pyramid_frustum(5, xyz(-2, 0, 0), 0.4, 0, xyz(-1, 1, 5), 0.4, true)
  regular_pyramid_frustum(4, xyz(0, 2, 0), 0.4, 0, xyz(1, 1, 5), 0.4, true)
  regular_pyramid_frustum(6, xyz(2, 0, 0), 0.4, 0, xyz(1, -1, 5), 0.4, true)
  regular_pyramid_frustum(7, xyz(0, -2, 0), 0.4, 0, xyz(-1, -1, 5), 0.4, true)
  set_view(xyz(3.5049, -12.3042, 15.1668),
           xyz(-0.604036, 0.523037, 1.60932), 50)
end

register_test("searsTower", :primitives_3d) do
  bloco_sears(p, i, j, l, h) =
    begin
      l2 = l/2
      r = sqrt(2)*l2
      pij = p+vxy(i*l+l2, j*l+l2)
      regular_pyramid_frustum(4, pij, r, pi/4, pij+vz(h), r, true)
    end
  torre_sears(p, l, h00, h01, h02, h10, h11, h12, h20, h21, h22) =
    begin
      l3 = l/3
      bloco_sears(p, 0, 0, l3, h00)
      bloco_sears(p, 0, 1, l3, h01)
      bloco_sears(p, 0, 2, l3, h02)
      bloco_sears(p, 1, 0, l3, h10)
      bloco_sears(p, 1, 1, l3, h11)
      bloco_sears(p, 1, 2, l3, h12)
      bloco_sears(p, 2, 0, l3, h20)
      bloco_sears(p, 2, 1, l3, h21)
      bloco_sears(p, 2, 2, l3, h22)
    end
  
  torre_sears(u0(), 68.7, 270, 442, 205, 368, 442, 368, 205, 368, 270)
  set_view(xyz(-75.043, -17.081, 637.179),
           xyz(85.0492, 79.1629, 153.154),
           30.0)
end

register_test("cruzPapal", :primitives_3d) do
  cruz_papal(p, raio) =
    begin
      cylinder(p, raio, p+vxyz(0, 0, 20*raio))
      cylinder(p+vxyz(-7*raio, 0, 9*raio), raio, p+vxyz(7*raio, 0, 9*raio))
      cylinder(p+vxyz(-5*raio, 0, 13*raio), raio, p+vxyz(5*raio, 0, 13*raio))
      cylinder(p+vxyz(-3*raio, 0, 17*raio), raio, p+vxyz(3*raio, 0, 17*raio))
    end
  cruz_papal(xyz(0, 0, 0), 1)
  set_view(xyz(28.1736, -50.1646, 26.8575),
           xyz(-2.04815, 5.662, 8.4588),
           50)
end

register_test("cruzetas", :primitives_3d) do
  cruzeta(p, rb, rt, c) =
    begin
      cone_frustum(p, rb, p+vx(c), rt)
      cone_frustum(p, rb, p+vy(c), rt)
      cone_frustum(p, rb, p+vz(c), rt)
      cone_frustum(p, rb, p+vx(-c), rt)
      cone_frustum(p, rb, p+vy(-c), rt)
      cone_frustum(p, rb, p+vz(-c), rt)
    end
  cruzeta(xyz(0, 0, 0), 1, 2, 5)
  cruzeta(xyz(12, 0, 0), 2, 1, 5)
  cruzeta(xyz(24, 0, 0), 2, 4, 2)
  set_view(xyz(2.79, -35.61, 23.97), xyz(2.96, -34.78, 23.44), 50)
end

register_test("cuboids", :primitives_3d) do
  cuboid(xyz(0, 0, 0),
         xyz(2, 0, 0),
         xyz(2, 2, 0),
         xyz(0, 2, 0),
         xyz(0, 0, 2),
         xyz(2, 0, 2),
         xyz(2, 2, 2),
         xyz(0, 2, 2))
  cuboid(xyz(4, 0, 0),
         xyz(5, 0, 0),
         xyz(5, 2, 0),
         xyz(4, 2, 0),
         xyz(3, 1, 2),
         xyz(5, 1, 2),
         xyz(5, 2, 2),
         xyz(3, 2, 2))
  cuboid(xyz(7, 2, 0),
         xyz(8, 0, 0),
         xyz(8, 3, 0),
         xyz(6, 3, 0),
         xyz(7, 2, 2),
         xyz(8, 0, 2),
         xyz(8, 3, 2),
         xyz(6, 3, 2))
  set_view(xyz(-10.1817, -42.536, 22.7753),
           xyz(3.25268, -1.27072, 2.07017),
           200.0)
end

register_test("tronco_rectangular", :primitives_3d) do
  tronco_rectangular(p, cb, lb, ct, lt, h) =
    cuboid(p+vxyz(-(cb/2), -(lb/2), 0),
           p+vxyz(cb/2, -(lb/2), 0),
           p+vxyz(cb/2, lb/2, 0),
           p+vxyz(-(cb/2), lb/2, 0),
           p+vxyz(-(ct/2), -(lt/2), h),
           p+vxyz(ct/2, -(lt/2), h),
           p+vxyz(ct/2, lt/2, h),
           p+vxyz(-(ct/2), lt/2, h))
  tronco_rectangular(xyz(0, 0, 0), 80.8, 50.3, 55.1, 34.3, 344)
  tronco_rectangular(xyz(200, 0, 0), 80, 50, 10, 30, 300)
  tronco_rectangular(xyz(400, 0, 0), 50, 50, 70, 70, 350)
  tronco_rectangular(xyz(600, 0, 0), 20, 40, 90, 10, 320)
  set_view(xyz(-1224.75, -4154.03, 528.056),
           xyz(252.093, -38.3401, 171.981),
           200.0)
end

register_test("pilone", :primitives_3d) do
  porta(p, c, h, e) =
    begin
      box(p+vx(-(c/2)-e), e, e, h)
      box(p+vx(c/2), e, e, h)
      box(p+vxz(-(c/2)-e, h), e+c+e, e, e)
    end

  torre(p, cb, lb, ct, lt, h) =
    cuboid(p+vxyz(-(cb/2), -(lb/2), 0),
           p+vxyz(cb/2, -(lb/2), 0),
           p+vxyz(cb/2, lb/2, 0),
           p+vxyz(-(cb/2), lb/2, 0),
           p+vxyz(-(ct/2), -(lt/2), h),
           p+vxyz(ct/2, -(lt/2), h),
           p+vxyz(ct/2, lt/2, h),
           p+vxyz(-(ct/2), lt/2, h))

  pilone(p, cb, lb, ct, lt, h, cp, hp, ep) =
    begin
      torre(p+vx(-((cp+cb)/2)), cb, lb, ct, lt, h)
      torre(p+vx((cp+cb)/2), cb, lb, ct, lt, h)
      porta(p+vy(-(lb/2)), cp, hp, ep)
    end

  pilone(xyz(0, 0, 0), 50, 20, 30, 10, 50, 6, 20, 8)
  set_view(xyz(28.9092, -109.67, 1.13012),
           xyz(0.461597, 10.8795, 28.6872),
           35.0)
end

register_test("piramidesRomboides", :primitives_3d) do
  piramide_romboide(p, l, h, a0, a1) =
    begin
      l0 = (h-l*tan(a1))/(tan(a0)-tan(a1))
      l1 = l-l0
      h0 = l0*tan(a0)
      h1 = h-h0
      regular_pyramid_frustum(4, p, l, 0, h0, l1)
      regular_pyramid(4, p+vz(h0), l1, 0, h1)
    end
  radianos_from_graus(graus) = pi*graus/180.0

  piramide_romboide(xyz(0, 0, 0),
                    186.6/2,
                    101.1,
                    radianos_from_graus(55),
                    radianos_from_graus(43))
  piramide_romboide(xyz(300, 0, 0),
                    186.6/2,
                    101.1,
                    radianos_from_graus(75),
                    radianos_from_graus(40))
  piramide_romboide(xyz(600, 0, 0),
                    186.6/2,
                    101.1,
                    radianos_from_graus(95),
                    radianos_from_graus(37))
  set_view(xyz(12.1513, 394.954, 38.4843),
           xyz(255.793, -174.573, 4.66954),
           20.0)
end

register_test("washingtonMonument", :primitives_3d) do
  obelisco(p, b, h, bp, hp) =
    begin
      l = b/2
      l1 = bp/2
      h0 = h-hp
      regular_pyramid_frustum(4, p, l, 0, h0, l1)
      regular_pyramid(4, p+vz(h0), l1, 0, hp)
    end
  obelisco(u0(), 16.8, 169.3, 10.5, 16.9)
  set_view(xyz(-32.7059, 169.073, 4.69462),
           xyz(66.8827, -256.645, 166.228),
           20.0)
end

register_test("arcoFalso", :primitives_3d) do
  arco_falso(p, c, e, de, l) =
    if e <= 0
      box(p+vxy(-(c/2), -(l/2)), p+vxyz(c/2, l/2, l))
    else
      box(p+vxy(-(c/2), -(l/2)), p+vxyz(-(e/2), l/2, l))
      box(p+vxy(e/2, -(l/2)), p+vxyz(c/2, l/2, l))
      arco_falso(p+vz(l), c, e-de-de, de, l)
    end
  arco_falso(u0(), 6, 4, 0.3, 0.5)
  set_view(xyz(0.0395904, -6.68326, 1.36019),
           xyz(0.038299, -6.46529, 1.38105),
           28.0)
end

register_test("urbeRecursiva", :primitives_3d) do
  urbe(p, a, c, n) =
    let p1 = p+vpol(c, a)
      right_cuboid(p, 10, 0.5, p1)
      if n == 0
        right_cuboid(p1+vcyl(10, a, 0), 20, 20, p1+vcyl(10, a, 40))
      else
        urbe(p1, a-pi/2, c/2.0, n-1)
        urbe(p1, a, c/2.0, n-1)
        urbe(p1, a+pi/2, c/2.0, n-1)
      end
    end

  urbe(xyz(0, 0, 0), 0, 500.0, 4)
  set_view(xyz(-268.61, -736.92, 801.01),
           xyz(3297.9, 2901.32, -3254.52),
           50)
end

register_test("cidadeEspacial", :primitives_3d) do
  cidade_espacial(p, raio) =
    begin
      sphere(p, raio/8.0)
      if raio < 1
        true
      else
        r2 = raio/2.0
        let (_px, ppx, _py, ppy, _pz, ppz) = (p+vxyz(-r2, 0, 0), p+vxyz(+r2, 0, 0), p+vxyz(0, -r2, 0), p+vxyz(0, +r2, 0), p+vxyz(0, 0, -r2), p+vxyz(0, 0, +r2))
          cylinder(_px, raio/32.0, ppx)
          cylinder(_py, raio/32.0, ppy)
          cylinder(_pz, raio/32.0, ppz)
          cidade_espacial(_px, r2)
          cidade_espacial(ppx, r2)
          cidade_espacial(_py, r2)
          cidade_espacial(ppy, r2)
          cidade_espacial(_pz, r2)
          cidade_espacial(ppz, r2)
        end
      end
    end
  cidade_espacial(xyz(0, 0, 0), 8)
  set_view(xyz(3.79426, 3.1758, 2.4499),
           xyz(-7.76258, -2.69934, -0.806126),
           35)
end

register_test("piramideCilindros", :primitives_3d) do
  piramide_cilindros(p, l, r, f, a, d, da) =
    if r < 0.01
      true
    else
      cylinder(p+vpol(l, a+-d), r, p+vpol(l, a+d+pi/2))
      cylinder(p+vpol(l, a+pi-d), r, p+vpol(l, a+d+3*pi/2))
      piramide_cilindros(p+vz((1+f)*r), l*f, r*f, f, a+pi/2+da, d, da)
    end

  piramide_cilindros(xyz(0, 0, 0), 10, 1, 0.92, 0, 0.15, 0.1)
  set_view(xyz(63.6514, -282.353, 71.9217),
           xyz(-6.64998, 24.4478, 4.91676),
           200)
end

register_test("arvores3D", :primitives_3d) do
  arvore(base, comprimento, angulo_fi, angulo_psi, min_fi, max_fi, min_psi, max_psi, min_f, max_f) =
    let topo = base+vsph(comprimento, angulo_fi, angulo_psi)
      ramo(base, topo)
      if comprimento < 2
        folha(topo)
      else
        arvore(topo,
               comprimento*random_range(min_f, max_f),
               angulo_fi+random_range(min_fi, max_fi),
               angulo_psi+random_range(min_psi, max_psi),
               min_fi,
               max_fi,
               min_psi,
               max_psi,
               min_f,
               max_f)
        arvore(topo,
               comprimento*random_range(min_f, max_f),
               angulo_fi-random_range(min_fi, max_fi),
               angulo_psi-random_range(min_psi, max_psi),
               min_fi,
               max_fi,
               min_psi,
               max_psi,
               min_f,
               max_f)
      end
    end

  ramo(base, topo) =
    let raio = distance(base, topo)/10.0
      cone_frustum(base, raio, topo, raio*0.9)
    end

  folha(topo) = sphere(topo, 0.5)

  set_random_seed(54320)
  arvore(xy(0, 0), 20, pi/2, 0.0, -(pi/2), pi/2, pi/12, pi/5, 0.6, 0.9)
  arvore(xy(100, 0), 20, pi/2, 0.0, -(pi/2), pi/2, pi/12, pi/5, 0.6, 0.9)
  arvore(xy(0, 100), 20, pi/2, 0.0, -(pi/2), pi/2, pi/12, pi/5, 0.6, 0.9)
  arvore(xy(100, 100), 20, pi/2, 0.0, -(pi/2), pi/2, pi/12, pi/5, 0.6, 0.9)
  set_view(xyz(745.4952, 1172.5071, 516.1298),
           xyz(-8.4988, -36.4129, 0.8168),
           200)
end

register_test("cilindrosAleatorios", :primitives_3d) do
  ponto_aleatorio() =
    xyz(random_range(0, 200), random_range(0, 100), random_range(0, 100))

  cilindros_aleatorios(n) =
    if n == 0
      true
    else
      cylinder(ponto_aleatorio(), random_range(1, 10), ponto_aleatorio())
      cilindros_aleatorios(n-1)
    end

  set_random_seed(12345)
  cilindros_aleatorios(100)
  set_view(xyz(-690.817, 1254.8545, 509.5468),
           xyz(160.033, -40.8755, 8.8878),
           200)
end

register_test("abacus", :primitives_3d) do
  abacus(p, l, ht, hb, e, n_bars, n_beads_t, n_beads_b) =
    let sep = l/(n_bars-1),
        rc = e/4,
        tre = 2//5*sep,
        tri = 11//10*rc,
        tr = (tre-tri)/2
      box(p, p+vxyz(e, e, e+hb+e+ht+e))
      box(p+vx(e+l), p+vxyz(e+l+e, e, e+hb+e+ht+e))
      box(p, p+vxyz(e+l+e, e, e))
      box(p+vz(e+hb+e+ht), p+vxyz(e+l+e, e, e+hb+e+ht+e))
      box(p+vz(e+hb), p+vxyz(e+l+e, e, e+hb+e))
      for x in division(e+sep/2, e+l+-(sep/2), n_bars-1)
        cylinder(p+vxyz(x, e/2, e/2), rc, p+vxyz(x, e/2, e+hb+e+ht+e/2))
        for i in 0:n_beads_b-1
          torus(p+vxyz(x, e/2, e+tr+i*2*tr), tre-tr, tr)
        end
        for i in 0:(n_beads_t-1)-1
          torus(p+vxyz(x, e/2, e+hb+e+tr+i*2*tr), tre-tr, tr)
        end
      end
    end

  with(current_cs, cs_from_o_vx_vy(xyz(0, 0, 0), vxyz(1, 0, 0), vxyz(0, 0, -1))) do
    abacus(u0(),
           15,
           1.5,
           3.5,
           0.4,
           15,
           2,
           5)
  end
  set_view(xyz(2.27621, -8.7908, 27.7904),
           xyz(5.80925, 0.431481, 6.06085),
           50)
end

register_test("escadasNCaracol", :primitives_3d) do
  escada_n_caracol(p, m, ri, re, l, e, a, da, n) =
    if n == 0
      []
    else
      for b in division(0, 2*pi, m, false)
        right_cuboid(p+vpol(ri, a+b), l, e, p+vpol(re, a+b))
      end
      escada_n_caracol(p+vxyz(0, 0, e), m, ri, re, l, e, a+da, da, n-1)
    end
  escada_n_caracol(xyz(0, 0, 0), 1, 6, 7, 2, 0.2, 0, pi/15, 60)
  escada_n_caracol(xyz(13, 13, 0), 3, 4, 5, 1, 0.3, pi, pi/20, 50)
  escada_n_caracol(xyz(26, 26, 0), 5, 4, 5, 1, 0.3, pi, pi/20, 50)
  set_view(xyz(231.77, -188.56, 197.25), xyz(7.19, 17.11, 2.61), 200.0)
end

# ── :surfaces ──────────────────────────────────────────────────────────

register_test("surface_polygons", :surfaces) do
  surface_polygon(xy(0, 0), xy(4, 0), xy(2, 3))
  surface_polygon(xy(5, 0), xy(9, 0), xy(9, 3), xy(5, 3))
  set_view(xyz(15, 15, 10), xyz(4, 0, 1), 50)
  zoom_extents()
end

register_test("surface_circles", :surfaces) do
  surface_circle(u0(), 3)
  surface_circle(x(8), 2)
  zoom_2d_top()
end

register_test("surface_grid_sin", :surfaces) do
  let n = 20,
      pts = [xyz(x, y, sin(x) * cos(y))
             for x in range(0, 4pi, length=n),
                 y in range(0, 4pi, length=n)]
    surface_grid(pts)
  end
  set_view(xyz(30, 30, 20), xyz(6, 6, 0), 50)
  zoom_extents()
end

register_test("surface_grid_ripple", :surfaces) do
  let n = 20,
      pts = [let r = sqrt(x^2 + y^2)
               xyz(x, y, r < 0.01 ? 2.0 : 2 * sin(r) / r)
             end
             for x in range(-6, 6, length=n),
                 y in range(-6, 6, length=n)]
    surface_grid(pts)
  end
  set_view(xyz(20, 20, 15), u0(), 50)
  zoom_extents()
end

register_test("surface_rectangles", :surfaces) do
  surface_rectangle(xy(0, 0), 5, 3)
  surface_rectangle(xy(6, 0), 2, 4)
  zoom_2d_top()
end

# ── :extrusions ────────────────────────────────────────────────────────

register_test("extrusaoSuperficie", :extrusions) do
  extrusion(
    polygon(
      reverse([xy(0, 2), xy(0, 5), xy(5, 3), xy(13, 3), xy(13, 6),
               xy(3, 6), xy(6, 7), xy(15, 7), xy(15, 2), xy(1, 2), xy(1, 0)])),
    1)
  set_view(xyz(-8.1453, -54.7236, 71.7539),
           xyz(8.2449, 8.2065, -5.1442), 199)
end

register_test("extrusaoSolido", :extrusions) do
  extrusion(
    surface_polygon(
      reverse([xy(0, 2), xy(0, 5), xy(5, 3), xy(13, 3), xy(13, 6),
               xy(3, 6), xy(6, 7), xy(15, 7), xy(15, 2), xy(1, 2), xy(1, 0)])),
    1)
  set_view(xyz(-8.1453, -54.7236, 71.7539),
           xyz(8.2449, 8.2065, -5.1442), 199)
end

register_test("extrusion_circle", :extrusions) do
  extrusion(region(circular_path(u0(), 2)), vz(5))
  set_view(xyz(15, 15, 10), xyz(0, 0, 2.5), 50)
  zoom_extents()
end

register_test("florCilindros", :extrusions) do
  for psi in division(pi/8, pi/2-pi/16, 5)
    for phi in division(0, 2*pi, 16, false)
      cylinder(u0(), 1, sph(20, phi, psi))
    end
  end
  set_view(xyz(-288.894, 23.9087, 75.961),
           xyz(-15.0015, -0.329266, 11.7649),
           199.0)
end

register_test("florCirculosExtrudidos", :extrusions) do
  for psi in division(pi/8, pi/2-pi/16, 5)
    for phi in division(0, 2*pi, 16, false)
      extrusion(surface_circle(), vsph(20, phi, psi))
    end
  end
  set_view(xyz(-288.894, 23.9087, 75.961),
           xyz(-15.0015, -0.329266, 11.7649),
           199.0)
end

register_test("revolve_line", :extrusions) do
  revolve(line(xyz(0, 0, 0), xyz(1, 1, 1), xyz(2, 0, 2)),
          x(5), vz(), 1//4*2*pi, 3//4*2*pi)
  set_view(xyz(19.6195, 51.9061, 19.8688),
           xyz(-0.3397, -0.6222, -1.0769), 199)
end

register_test("loft_circles", :extrusions) do
  loft([circle(xyz(0, 0, 0), 4),
        circle(xyz(0, 0, 2), 2),
        circle(xyz(0, 0, 4), 3)])
  loft([surface_circle(xyz(0, 9, 0), 4),
        surface_circle(xyz(0, 9, 2), 2),
        surface_circle(xyz(0, 9, 4), 3)])
  set_view(xyz(76.69, 3.10, 63.36), xyz(75.91, 3.11, 62.73), 200)
end

#ERROR
register_test("sweep_circle_path", :extrusions) do
  sweep_path(
    open_spline_path([xyz(0,0,0), xyz(2,3,1), xyz(5,2,3), xyz(8,0,2)]),
    circular_path(u0(), 0.5))
  set_view(xyz(15, 15, 10), xyz(4, 1, 1.5), 50)
  zoom_extents()
end

register_test("abrigoEsfericoTubos", :extrusions) do
  pontos_meridiano(p, r, fi, psi0, psi1, n) =
    map_division(psi -> p+vsph(r, fi, psi), psi0, psi1, n)

  cilindrifica(curva, r) = sweep(curva, surface_circle(u0(), r))
  
  map_division(fi -> cilindrifica(spline(pontos_meridiano(u0(), 10, fi, 0, pi/2, 30)), 0.5),
               0,
               pi,
               20)
  set_view(xyz(40.3121, -115.849, 53.2611),
           xyz(-1.26541, 5.06909, 2.59963),
           197.051)
end

register_test("predioSinusoidal", :extrusions) do
  sinusoide(a, omega, fi, x) = a*sin(omega*x+fi)

  pontos_sinusoide(p, a, omega, fi, x0, x1, dx) =
    x0 > x1 ? 
      [] : 
      [p+vxy(x0, sinusoide(a, omega, fi, x0)),
       pontos_sinusoide(p, a, omega, fi, x0+dx, x1, dx)...]

  laje(p, a, omega, fi, lx, dx, ly, lz) =
    let pontos = pontos_sinusoide(p, a, omega, fi, 0, lx, dx)
      extrusion(surface(spline(pontos),
                        line(pontos[end], p+vxy(lx, ly), p+vxy(0, ly), pontos[1])),
                lz)
    end

  parede_sinusoidal(p, a, omega, fi, x0, x1, dx, e, h) =
    let (pts_1, pts_2) = (pontos_sinusoide(p, a, omega, fi, x0, x1, dx), pontos_sinusoide(p+vy(e), a, omega, fi, x0, x1, dx))
      extrusion(surface(spline(pts_1),
                        spline([pts_1[end], pts_2[end]]),
                        spline(reverse(pts_2)),
                        spline([pts_2[1], pts_1[1]])),
                h)
    end
    
  corrimao(p, a, omega, fi, lx, dx, l_corrimao, a_corrimao) =
    parede_sinusoidal(p, a, omega, fi, 0, lx, dx, l_corrimao, a_corrimao)

  prumos(p, a, omega, fi, lx, dx, altura, raio) =
    map(ponto -> cylinder(ponto, raio, ponto+vxyz(0, 0, altura)),
        pontos_sinusoide(p, a, omega, fi, 0, lx, dx))

  guarda(p, a, omega, fi, lx, dx, a_guarda, l_corrimao, a_corrimao, d_prumos) =
    begin
      corrimao(p+vxyz(0, l_corrimao/-2.0, a_guarda),
               a, omega, fi, lx, d_prumos,
               #using d-prumos due to rounding errors dx,
               l_corrimao, a_corrimao)
      prumos(p, a, omega, fi, lx, d_prumos, a_guarda, l_corrimao/3.0)
    end

  piso(p, a, omega, fi, lx, dx, ly, a_laje, a_guarda, l_corrimao, a_corrimao, d_prumos) =
    begin
      laje(p, a, omega, fi, lx, d_prumos,
           #using d-prumos due to rounding errors dx,
           ly, a_laje)
      guarda(p+vxyz(0, l_corrimao, a_laje), a, omega, fi, lx, dx,
             a_guarda, l_corrimao, a_corrimao, d_prumos)
    end

  predio(p, a, omega, fi, lx, dx, ly, a_laje, a_guarda, l_corrimao, a_corrimao, d_prumos, a_andar, n_andares) =
    if n_andares == 0
      nothing
    else
      piso(p, a, omega, fi, lx, dx, ly, a_laje, a_guarda, l_corrimao, a_corrimao, d_prumos)
      predio(p+vxyz(0, 0, a_andar), a, omega, fi, lx, dx, ly, a_laje, a_guarda, l_corrimao, a_corrimao, d_prumos, a_andar, n_andares-1)
    end

  predio(xy(0, 0), 1.0, 1.0, 0, 20*pi, 0.5, 20, 0.2, 1, 0.06, 0.02, 0.4, 4, 2)
  set_view(xyz(-201.0679, -343.0591, 115.357),
           xyz(20.6981, -5.0361, 17.5824),
           200)
end

register_test("corrimaoCaracol", :extrusions) do
  escada_caracol(p, ri, re, l, e, a, da, n) =
    n == 0 ? 
      [] : 
      [right_cuboid(p+vpol(ri, a), l, e, p+vpol(re, a)),
       escada_caracol(p+vz(e), ri, re, l, e, a+da, da, n-1)...]
  
  corrimao(pts, r, h) =
    begin
      sweep(spline(pts), surface_circle(u0(), r))
      map(pt -> cylinder(pt, r, pt+vz(-h)),
          pts)
    end
  
  corrimao_caracol(p, r, e, a, da, n, h) =
    n == 0 ? 
      [] : 
      [p+vcyl(r, a, h),
       corrimao_caracol(p+vxyz(0, 0, e), r, e, a+da, da, n-1, h)...]
       
  escada_caracol(xyz(0, 0, 0), 5, 7, 2, 0.15, 0, pi/16, 80)
  escada_caracol(xyz(13, 13, 0), 4, 5, 1, 0.3, pi, pi/20, 50)
  corrimao(corrimao_caracol(xyz(0, 0, 0), 6.8, 0.15, 0, pi/16, 80, 1), 0.05, 1)
  corrimao(corrimao_caracol(xyz(0, 0, 0), 5.2, 0.15, 0, pi/16, 80, 1), 0.05, 1)
  corrimao(corrimao_caracol(xyz(13, 13, 0), 4.9, 0.3, pi, pi/20, 50, 1), 0.05, 1)
  set_view(xyz(161.3483, -93.4412, 47.977),
           xyz(2.0903, 9.1328, 5.2713),
           200)
end

# ── :csg ───────────────────────────────────────────────────────────────

register_test("csg_union", :csg) do
  union(box(xyz(0, 0, 0), xyz(1, 1, 1)),
        sphere(xyz(1, 0, 1), 0.5))
  set_view(xyz(16.3249, -21.2313, 16.2413),
           xyz(2.1267, 1.3586, -0.1368), 200)
end

register_test("csg_subtraction", :csg) do
  subtraction(box(xyz(4, 0, 0), xyz(5, 1, 1)),
              sphere(xyz(5, 0, 1), 0.5))
  set_view(xyz(16.3249, -21.2313, 16.2413),
           xyz(2.1267, 1.3586, -0.1368), 200)
end

register_test("csg_intersection", :csg) do
  intersection(box(xyz(2, 0, 0), xyz(3, 1, 1)),
               sphere(xyz(3, 0, 1), 0.5))
  set_view(xyz(16.3249, -21.2313, 16.2413),
           xyz(2.1267, 1.3586, -0.1368), 200)
end

register_test("banheira", :csg) do
  banheira(p, r, e, l) =
    let re = r+e
      subtraction(box(p+vxyz(-re, -re, -re), p+vxyz(l+re, +re, 0)),
                  cylinder(p, r, p+vx(l)),
                  sphere(p+vx(l), r),
                  sphere(p, r))
    end
  banheira(xyz(0, 0, 0), 10, 2, 15)
  set_view(xyz(168.7713, -256.5569, 179.258),
           xyz(5.8413, 2.6721, -8.688),
           200)
end

register_test("cilindrosUniaoInterseccao", :csg) do
  union([cylinder(xyz(-1, 0, 0), 1, xyz(1, 0, 0)),
         cylinder(xyz(0, -1, 0), 1, xyz(0, 1, 0)),
         cylinder(xyz(0, 0, -1), 1, xyz(0, 0, 1))])
  intersection([cylinder(xyz(3, 0, 0), 1, xyz(5, 0, 0)),
                cylinder(xyz(4, -1, 0), 1, xyz(4, 1, 0)),
                cylinder(xyz(4, 0, -1), 1, xyz(4, 0, 1))])
  set_view(xyz(20.1789, -29.4306, 21.3149),
           xyz(1.1787, 0.7996, -0.6025),
           200)
end

register_test("csg_compound", :csg) do
  subtraction(
    intersection([cylinder(xyz(-1, 0, 0), 1, xyz(1, 0, 0)),
                  cylinder(xyz(0, -1, 0), 1, xyz(0, 1, 0)),
                  cylinder(xyz(0, 0, -1), 1, xyz(0, 0, 1))]),
    sphere(xyz(0, 0, 0), 1.01))
  set_view(xyz(4.7056, -21.2502, 7.6505),
           xyz(-0.0838, 0.2606, -0.1048), 200)
end

register_test("predioCircularSinB", :csg) do
  piso(p, r_laje, e_laje, r_pilares, r_pilar, n_pilares, h_pilar) =
    union(union(map_division(a -> cylinder(p+vpol(r_pilares, a), r_pilar, h_pilar),
                             0,
                             2*pi,
                             n_pilares,
                             false)),
          cylinder(p+vz(h_pilar), r_laje, e_laje))
  torre_conica_oscilante(p, r_laje, e_laje, r_pilares, r_pilar, n_pilares, h_pilar, n_pisos, a, omega, f) =
    map((i, f) -> piso(p+vxz(a*sin(omega*i/n_pisos*2*pi), i*(h_pilar+e_laje)),
                             f*r_laje,
                             e_laje,
                             f*r_pilares,
                             r_pilar,
                             n_pilares,
                             h_pilar),
        division(0, n_pisos, n_pisos, false),
        division(1, f, n_pisos, false))
  torre_conica_oscilante(u0(), 12, 0.3, 8, 0.2, 10, 3, 16, 2, 1, 0.2)
  torre_conica_oscilante(x(40), 15, 0.25, 13, 0.3, 15, 2.8, 20, 3, 2, 1.3)
  torre_conica_oscilante(x(80), 10, 0.25, 8, 0.1, 25, 2.7, 21, 4, 1, 0.6)
  set_view(xyz(67.1643, -128.435, 68.2267),
           xyz(-11.7019, 348.709, -76.4216),
           35.0)
end

register_test("folios", :csg) do
  n_folio(p, re, n) =
    union(folhas_folio(p, re, n), circulo_interior_folio(p, re, n))

  uniao_circulos(p, ro, fi, d_fi, rf, n) =
    n == 1 ? 
      surface(circle(p+vpol(ro, fi), rf)) : 
      union(surface(circle(p+vpol(ro, fi), rf)),
            uniao_circulos(p, ro, fi+d_fi, d_fi, rf, n-1))

  folhas_folio(p, re, n) =
    uniao_circulos(p, re/(1+sin(pi/n)), 0, 2*pi/n, re/(1+1/sin(pi/n)), n)

  circulo_interior_folio(p, re, n) =
    n == 2 ? empty_shape() : surface_circle(p, re*cos(pi/n)/(1+sin(pi/n)))

  n_folio(xy(0, 0), 1, 3)
  n_folio(xy(2.5, 0), 1, 4)
  zoom_2d_top()
end

register_test("poligonosRecursivos", :csg) do
  poligonos_recursivos(p, r, fi, alfa_r, n, nivel) =
    if nivel == 0
      empty_shape()
    else
      pontos = regular_polygon_vertices(n, p, r, fi, true)
      union([surface_polygon(pontos),
         lista_poligonos_recursivos(pontos, r*alfa_r, fi, alfa_r, n, nivel-1)...])
    end

  lista_poligonos_recursivos(pontos, r, fi, alfa_r, n, nivel) =
    pontos == [] ? 
      [] : 
      [poligonos_recursivos(pontos[1], r, fi, alfa_r, n, nivel),
       lista_poligonos_recursivos(pontos[2:end], r, fi, alfa_r, n, nivel)...]

  poligonos_recursivos(xy(0, 0), 1, 0, 0.3, 4, 2)
  poligonos_recursivos(xy(3, 0), 1, pi/3, 0.3, 3, 3)
  poligonos_recursivos(xy(6, 0), 1, 0, 0.3, 5, 4)
  zoom_2d_top()
end

register_test("abobadasRomanas", :csg) do
  raios_cilindro(p, r, a, da, rc, n) =
    n == 0 ?
      empty_shape() :
      union(cylinder(p, rc, p+vpol(r, a)),
            raios_cilindro(p, r, a+da, da, rc, n-1))


  cobertura_arcos_romanos(p, r, e, n) =
    let da = 2*pi/n,
        #subtraction(a, b) = true,
        (lc, rc) = (r*cos(da/2), r*sin(da/2))
        subtraction(subtraction(raios_cilindro(p, lc, 0, da, rc, n),
                                raios_cilindro(p, r, 0, da, rc-e, n)),
                    cylinder(p, r, p+vz(-rc)))
    end
  
  cobertura_arcos_romanos(xyz(-8, -8, 0), 5, 0.8, 3)
  cobertura_arcos_romanos(xyz(0, 0, 0), 5, 0.2, 4)
  cobertura_arcos_romanos(xyz(8, 8, 0), 4, 0.2, 6)
  cobertura_arcos_romanos(xyz(16, 16, 0), 8, 0.1, 10)
  set_view(xyz(41.6558, -84.6266, 28.0611),
           xyz(2.5878, 5.7118, -0.0298),
           193)
end

register_test("cascasPerfuradas", :csg) do
  casca_esferica_perfurada(p, r, e, rc) =
    subtraction(
      sphere(p, r),
      vcat(map(psi -> let n = floor(15*sin(pi*psi))+1
                        map(i -> cone(p+vsph(1.2*r, (i*2*pi)/n, pi*psi), rc, p), 0:n-1)
                      end,
               0:1//10:1)...)...,
               sphere(p, r-e))

  casca_esferica_perfurada(xyz(0, 0, 0), 1, 0.05, 0.1)
  casca_esferica_perfurada(xyz(3, 0, -0.2), 0.8, 0.04, 0.09)
  casca_esferica_perfurada(xyz(6, 0, 0.2), 1.2, 0.02, 0.2)
  set_view(xyz(2.3014, -27.0913, 29.4911),
           xyz(3.0671, 0.1468, -0.169),
           183)
end

register_test("coberturaTubos", :csg) do
  malha_tubos(p, c, r, n, m, sp, sr) =
    m == 0 ? 
      empty_shape() : 
      (linha_tubos(p, c, r, n, sp, sr), 
      malha_tubos(p+vz(2*r), c, r, n, m-1, sp, sr))

  linha_tubos(p, c, r, n, sp, sr) =
    n == 0 ? 
      empty_shape() : 
      (subtraction(cylinder(p, r, p+vy(c)), 
                  cylinder(p-vy(1), 0.9*r, p+vy(c+1)),
                  distance(p, sp) < sr + r ? sphere(sp, sr) : empty_shape()),
       linha_tubos(p+vx(2*r), c, r, n-1, sp, sr))
       
  cobertura_tubos(p, h, n) =
    let r1 = h/2.0/n,
        r0 = h-r1
      malha_tubos(p+vxyz(-r0, 0, r1), h, r1, 2*n, n, p, r0)
    end
  cobertura_tubos(xyz(0, 0, 0), 5, 9)
  set_view(xyz(-2.9726, -16.1778, 1.2609),
           xyz(0.3387, 5.3931, 2.5914),
           50)
end

register_test("coberturaTubos2", :csg) do
  malha_tubos(p, c, r, n, m, sp, sr) =
    m == 0 ?
      empty_shape() :
      (linha_tubos(p, c, r, n, sp, sr),
       malha_tubos(p+vz(2*r), c, r, n, m-1, sp, sr))

  linha_tubos(p, c, r, n, sp, sr) =
    n == 0 ?
      empty_shape() :
      (subtraction(cylinder(p, r, p+vx(c)),
                   cylinder(p-vx(1), 0.9*r, p+vx(c+1)),
                   distance(p+vx(c/2), sp) < sr + r ? sphere(sp, sr) : empty_shape()),
       linha_tubos(p+vy(2*r), c, r, n-1, sp, sr))

  cobertura_tubos(p, h, n) =
    let r1 = h/2.0/n,
        r0 = h-r1
      malha_tubos(p+vxyz(-h, r1, r1), 2*h, r1, n, n, p, r0)
    end
  cobertura_tubos(xyz(0, 0, 0), 5, 9)
  set_view(xyz(-2.9726, -16.1778, 1.2609),
           xyz(0.3387, 5.3931, 2.5914),
           50)
end

register_test("coberturaTubos3", :csg) do
  malha_tubos(p, c, r, n, m, sp, sr) =
    m == 0 ?
      empty_shape() :
      (linha_tubos(p, c, r, n, sp, sr),
       malha_tubos(p+vy(2*r), c, r, n, m-1, sp, sr))
  
  linha_tubos(p, c, r, n, sp, sr) =
    n == 0 ?
      empty_shape() :
      (subtraction(cylinder(p, r, p+vz(c)), 
                   cylinder(p-vz(1), 0.9*r, p+vz(c+1)),
                   distance(p, sp) < sr + r ? sphere(sp, sr) : empty_shape()),
       linha_tubos(p+vx(2*r), c, r, n-1, sp, sr))
  
  cobertura_tubos(p, h, n) =
    let r1 = h/2.0/n,
        r0 = h-r1
      malha_tubos(p+vxyz(-r0, r1, 0), h, r1, 2*n, n, p, r0)
    end
  cobertura_tubos(xyz(0, 0, 0), 5, 9)
  set_view(xyz(-2.9726, -16.1778, 1.2609),
           xyz(0.3387, 5.3931, 2.5914),
           50)
end

normal_pontos(p0, p1, p2, p3) =
  let (v0, v1) = (p1-p0, p2-p0)
    n = cross(v0, v1)
    dot(n, p3-p0) < 0 ? n : n*-1
  end

register_test("tetraedroEvol", :csg) do

  tetraedro(i, p0, p1, p2, p3) =
    let (pmin, pmax) = (xyz(min(p0.x, p1.x, p2.x, p3.x), min(p0.y, p1.y, p2.y, p3.y), min(p0.z, p1.z, p2.z, p3.z)), xyz(max(p0.x, p1.x, p2.x, p3.x), max(p0.y, p1.y, p2.y, p3.y), max(p0.z, p1.z, p2.z, p3.z)))
      solido = box(pmin, pmax)
      solido = i > 0 ? slice(solido, p0, normal_pontos(p0, p1, p2, p3)) : solido
      solido = i > 1 ? slice(solido, p1, normal_pontos(p1, p2, p3, p0)) : solido
      solido = i > 2 ? slice(solido, p2, normal_pontos(p2, p3, p0, p1)) : solido
      solido = i > 3 ? slice(solido, p3, normal_pontos(p3, p0, p1, p2)) : solido
      solido
    end
  vertices = [xyz(1, 1, 1), xyz(-1, -1, 1), xyz(-1, 1, -1), xyz(1, -1, -1)]
  for i in 0:4
    tetraedro(i, map(p -> p+vy(i*4), vertices)...)
  end
  set_view(xyz(88.2914, -20.1148, 25.2363),
           xyz(0.9228, 7.5387, 0.1266),
           200.0)
end

register_test("duploTetraedro", :csg) do
  tetraedro(p0, p1, p2, p3) =
    let (pmin, pmax) = (xyz(min(p0.x, p1.x, p2.x, p3.x), min(p0.y, p1.y, p2.y, p3.y), min(p0.z, p1.z, p2.z, p3.z)), xyz(max(p0.x, p1.x, p2.x, p3.x), max(p0.y, p1.y, p2.y, p3.y), max(p0.z, p1.z, p2.z, p3.z)))
      solido = box(pmin, pmax)
      solido = slice(solido, p0, normal_pontos(p0, p1, p2, p3))
      solido = slice(solido, p1, normal_pontos(p1, p2, p3, p0))
      solido = slice(solido, p2, normal_pontos(p2, p3, p0, p1))
      solido = slice(solido, p3, normal_pontos(p3, p0, p1, p2))
      solido
    end
  union(tetraedro(u0(), uxy(), uyz(), uxz()),
        tetraedro(ux(), uy(), uz(), uxyz()))
  set_view(xyz(15.7731, -4.34902, 7.3827),
           xyz(0.742007, 0.373327, 0.550518),
           200.0)
end


register_test("octahedro", :csg) do
  tetraedro(p0, p1, p2, p3) =
    let (pmin, pmax) = (xyz(min(p0.x, p1.x, p2.x, p3.x), min(p0.y, p1.y, p2.y, p3.y), min(p0.z, p1.z, p2.z, p3.z)), xyz(max(p0.x, p1.x, p2.x, p3.x), max(p0.y, p1.y, p2.y, p3.y), max(p0.z, p1.z, p2.z, p3.z)))
      solido = box(pmin, pmax)
      solido = slice(solido, p0, normal_pontos(p0, p1, p2, p3))
      solido = slice(solido, p1, normal_pontos(p1, p2, p3, p0))
      solido = slice(solido, p2, normal_pontos(p2, p3, p0, p1))
      solido = slice(solido, p3, normal_pontos(p3, p0, p1, p2))
      solido
    end
  intersection(tetraedro(u0(), uxy(), uyz(), uxz()),
               tetraedro(ux(), uy(), uz(), uxyz()))
  set_view(xyz(15.7731, -4.34902, 7.3827),
           xyz(0.742007, 0.373327, 0.550518),
           200.0)
end

register_test("octaedroEstrelado", :csg) do
  tetraedro(p0, p1, p2, p3) =
    let (pmin, pmax) = (xyz(min(p0.x, p1.x, p2.x, p3.x), min(p0.y, p1.y, p2.y, p3.y), min(p0.z, p1.z, p2.z, p3.z)), xyz(max(p0.x, p1.x, p2.x, p3.x), max(p0.y, p1.y, p2.y, p3.y), max(p0.z, p1.z, p2.z, p3.z)))
      solido = box(pmin, pmax)
      solido = slice(solido, p0, normal_pontos(p0, p1, p2, p3))
      solido = slice(solido, p1, normal_pontos(p1, p2, p3, p0))
      solido = slice(solido, p2, normal_pontos(p2, p3, p0, p1))
      solido = slice(solido, p3, normal_pontos(p3, p0, p1, p2))
      solido
    end
  octaedro_estrelado(p, l) =
    union(tetraedro(p+vxyz(-(l/2), -(l/2), -(l/2)),
                  p+vxyz(l/2, l/2, -(l/2)),
                  p+vxyz(-(l/2), l/2, l/2),
                  p+vxyz(l/2, -(l/2), l/2)),
          tetraedro(p+vxyz(l/2, -(l/2), -(l/2)),
                    p+vxyz(-(l/2), l/2, -(l/2)),
                    p+vxyz(-(l/2), -(l/2), l/2),
                    p+vxyz(l/2, l/2, l/2)))
  octaedro_estrelado(u0(), 1)
  set_view(xyz(15.7731, -4.34902, 7.3827),
           xyz(0.742007, 0.373327, 0.550518),
           200.0)
end

register_test("calotaEsferica", :csg) do
  calota_esferica(p0, p1, d) =
    let h = distance(p0, p1),
        n = p0-p1,
        r = h/2+(d*d)/(8*h),
        c = p1+n*r/h
      slice(sphere(c, r), p0, n)
    end

  calota_esferica(xyz(0, 0, 0), xyz(0, 1, 1), 4)
  set_view(xyz(-25.081, -22.1148, 8.7675),
           xyz(7.8206, 6.4513, -2.5557),
           190)
end

# ── :parametric ────────────────────────────────────────────────────────

register_test("sierpinski", :parametric) do
  meio_xyz(p1, p2) = p1 + (p2 - p1)/2
  tetraedro(p0, p1, p2, p3) = begin
    surface_polygon(p0, p1, p2)
    surface_polygon(p0, p1, p3)
    surface_polygon(p0, p2, p3)
    surface_polygon(p1, p2, p3)
  end
  function sierpinski(p0, p1, p2, p3, n)
    if n == 0
      tetraedro(p0, p1, p2, p3)
    else
      let (p0p1, p0p2, p0p3, p1p2, p1p3, p2p3) =
            (meio_xyz(p0,p1), meio_xyz(p0,p2), meio_xyz(p0,p3),
             meio_xyz(p1,p2), meio_xyz(p1,p3), meio_xyz(p2,p3))
        sierpinski(p0, p0p1, p0p2, p0p3, n-1)
        sierpinski(p0p1, p1, p1p2, p1p3, n-1)
        sierpinski(p0p2, p1p2, p2, p2p3, n-1)
        sierpinski(p0p3, p1p3, p2p3, p3, n-1)
      end
    end
  end
  for i in 0:4
    p = xyz(i*2.2, i*2.2, 0)
    sierpinski(p+vxyz(-1, -1, -1),
               p+vxyz(1, 1, -1),
               p+vxyz(1, -1, 1),
               p+vxyz(-1, 1, 1),
               i)
  end
  set_view(xyz(102.2687, -93.3245, 382.1695),
           xyz(4.402, 4.3919, -0.3285), 1000)
end

register_test("column_grid", :parametric) do
  for x in 0:4:16
    for y in 0:4:16
      cylinder(xyz(x, y, 0), 0.3, 5)
    end
  end
  set_view(xyz(30, 30, 20), xyz(8, 8, 2.5), 50)
  zoom_extents()
end

register_test("helix_boxes", :parametric) do
  for i in 0:30
    let a = i * pi/8,
        r = 5,
        z = i * 0.3
      box(xyz(r*cos(a), r*sin(a), z), 0.5, 0.5, 0.5)
    end
  end
  set_view(xyz(20, 20, 15), u0(), 50)
  zoom_extents()
end


# ── Comparison functions ──────────────────────────────────────────────

text_compare(test_path, golden_path; kwargs...) =
  read(test_path) == read(golden_path)

function pixel_diff_compare(test_path, golden_path; threshold=0.01)
  test_bytes = read(test_path)
  golden_bytes = read(golden_path)
  test_bytes == golden_bytes && return true
  # Pixel-level comparison using PNGFiles (if available)
  try
    PNGFiles = Base.loaded_modules[Base.PkgId(Base.UUID("f57f5aa1-a3ce-4bc8-8ab9-96f992907883"), "PNGFiles")]
    test_img = PNGFiles.load(test_path)
    golden_img = PNGFiles.load(golden_path)
    size(test_img) != size(golden_img) && return false
    n = length(test_img)
    ndiff = count(i -> !isapprox(test_img[i], golden_img[i]; atol=0.02), 1:n)
    ndiff / n <= threshold
  catch
    false
  end
end

# ── Test runner ───────────────────────────────────────────────────────

function run_visual_tests(b;
    golden_dir::String,
    reset!::Function,
    setup_backend::Function = () -> setup_raw_view(b),
    width::Int = 1920,
    height::Int = 1080,
    compare::Function = text_compare,
    skip::Vector{Symbol} = Symbol[])

  rendering_with(dir=mktempdir(), width=width, height=height) do
    setup_backend()
    for (name, category, test_fn) in VISUAL_TESTS
      category in skip && continue
      @testset "$name" begin
        reset!()
        test_path = try
          test_fn()
        catch e
          if is_unimplemented(e)
            @warn "Skipping $name — unimplemented backend operation"
            @test_broken false
            continue
          end
          @error "Visual test failed to run" name exception=(e, catch_backtrace())
          @test false
          continue
        end
        ext = splitext(test_path)[2]
        golden_path = joinpath(golden_dir, "$(name)$(ext)")

        if isfile(golden_path)
          if compare(test_path, golden_path)
            @test true
          else
            @test false
            @warn "Visual regression detected for $name" test_path golden_path
          end
        else
          mkpath(golden_dir)
          cp(test_path, golden_path, force=true)
          @info "Created golden file: $golden_path"
          @test true
        end
      end
    end
  end
end

end # module
