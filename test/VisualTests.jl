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
#   KHEPRI_UPDATE_GOLDEN=1 julia --project -e "using Pkg; Pkg.test()"
#
# Backend setup requirements for consistent golden files:
#
# 1. Render size: rendering_with sets width/height to 16:9 FullHD ratio (960x540).
#    All backends that use render_width()/render_height() will respect this.
#
# 2. Multi-viewport backends (Rhino, AutoCAD):
#    - set_view() automatically maximizes the Perspective viewport in Rhino
#    - set_view_top() automatically maximizes the Top viewport in Rhino
#    - Use setup_backend callback for any additional one-time configuration
#
# 3. After changing view parameters or test content, regenerate golden files:
#    KHEPRI_UPDATE_GOLDEN=1 julia --project -e "using Pkg; Pkg.test()"

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

function register_test(name, category, fn)
  push!(VISUAL_TESTS, (name, category, fn))
end

# ── :primitives_2d ─────────────────────────────────────────────────────

register_test("circles", :primitives_2d, () -> begin
  for r in 1:5
    circle(u0(), r)
  end
  zoom_2d_top()
  raw_view("circles")
end)

register_test("polygons", :primitives_2d, () -> begin
  for (i, n) in enumerate(3:8)
    regular_polygon(n, x(i * 3), 1)
  end
  zoom_2d_top()
  raw_view("polygons")
end)

register_test("arcs", :primitives_2d, () -> begin
  arc(u0(), 5, 0, pi/2)
  arc(u0(), 4, pi/4, pi)
  circle(u0(), 3)
  zoom_2d_top()
  raw_view("arcs")
end)

register_test("spline_curve", :primitives_2d, () -> begin
  spline([xy(0, 0), xy(1, 2), xy(3, 1), xy(5, 3), xy(7, 0)])
  zoom_2d_top()
  raw_view("spline_curve")
end)

register_test("rectangles", :primitives_2d, () -> begin
  rectangle(xy(0, 0), 4, 3)
  rectangle(xy(5, 0), 2, 5)
  rectangle(xy(8, 0), 3, 3)
  zoom_2d_top()
  raw_view("rectangles")
end)

register_test("lines_and_polygons", :primitives_2d, () -> begin
  line(xy(0, 0), xy(5, 0), xy(5, 5))
  polygon(xy(6, 0), xy(10, 0), xy(10, 4), xy(8, 5), xy(6, 3))
  zoom_2d_top()
  raw_view("lines_and_polygons")
end)

# ── :primitives_3d ─────────────────────────────────────────────────────

register_test("boxes", :primitives_3d, () -> begin
  box(xyz(1, 1, 0), xyz(0.5, 0.5, 0.5))
  box(xyz(1.7, 1.7, 0), xyz(1.5, 1.5, 0.5))
  box(xyz(2, 2, 0), xyz(2.5, 2.5, 0.5))
  set_view(xyz(-0.498875, -0.825617, 0.681738),
           xyz(0.484146, 0.459262, 0.329734), 50)
  raw_view("boxes")
end)

register_test("spheres", :primitives_3d, () -> begin
  sphere(xyz(1, 2, 3), 4)
  sphere(xyz(5, 2, 3), 2)
  sphere(xyz(7, 2, 3), 1)
  set_view(xyz(71.5486, 50.8126, 50.9644),
           xyz(-243.179, -168.982, -167.24), 197)
  raw_view("spheres")
end)

register_test("solids", :primitives_3d, () -> begin
  box(xyz(2, 1, 1), xyz(3, 4, 5))
  cone(xyz(6, 0, 0), 1, xyz(8, 1, 5))
  cone_frustum(xyz(11, 1, 0), 2, xyz(10, 0, 5), 1)
  sphere(xyz(8, 4, 5), 2)
  cylinder(xyz(8, 7, 0), 1, xyz(6, 8, 7))
  regular_pyramid(5, xyz(-2, 1, 0), 1, 0, xyz(2, 7, 7))
  torus(xyz(14, 6, 5), 2, 1)
  set_view(xyz(8.057, -23.2615, 9.5719),
           xyz(6.30023, 7.5729, 1.71575), 50)
  raw_view("solids")
end)

register_test("cylinders", :primitives_3d, () -> begin
  cylinder(xyz(0, 0, 0), 1, 5)
  cylinder(xyz(4, 0, 0), 0.5, 3)
  cylinder(xyz(8, 0, 0), 2, 1)
  set_view(xyz(20, 20, 15), xyz(4, 0, 1.5), 50)
  zoom_extents()
  raw_view("cylinders")
end)

register_test("cones", :primitives_3d, () -> begin
  cone(xyz(0, 0, 0), 2, 4)
  cone_frustum(xyz(6, 0, 0), 2, 4, 1)
  cone_frustum(xyz(12, 0, 0), 1, 3, 2)
  set_view(xyz(20, 20, 15), xyz(6, 0, 2), 50)
  zoom_extents()
  raw_view("cones")
end)

register_test("pyramids", :primitives_3d, () -> begin
  regular_pyramid(3, xyz(0, 0, 0), 2, 0, 3)
  regular_pyramid(4, xyz(6, 0, 0), 2, 0, 3)
  regular_pyramid(6, xyz(12, 0, 0), 2, 0, 3)
  set_view(xyz(20, 20, 15), xyz(6, 0, 1.5), 50)
  zoom_extents()
  raw_view("pyramids")
end)

register_test("tori", :primitives_3d, () -> begin
  torus(xyz(0, 0, 0), 3, 1)
  torus(xyz(10, 0, 0), 2, 0.5)
  torus(xyz(18, 0, 0), 4, 0.3)
  set_view(xyz(20, 20, 15), xyz(9, 0, 0), 50)
  zoom_extents()
  raw_view("tori")
end)

register_test("prisms", :primitives_3d, () -> begin
  regular_pyramid_frustum(3, xyz(0, 0, 0), 0.4, 0, xyz(0, 0, 5), 0.4, true)
  regular_pyramid_frustum(5, xyz(-2, 0, 0), 0.4, 0, xyz(-1, 1, 5), 0.4, true)
  regular_pyramid_frustum(4, xyz(0, 2, 0), 0.4, 0, xyz(1, 1, 5), 0.4, true)
  regular_pyramid_frustum(6, xyz(2, 0, 0), 0.4, 0, xyz(1, -1, 5), 0.4, true)
  regular_pyramid_frustum(7, xyz(0, -2, 0), 0.4, 0, xyz(-1, -1, 5), 0.4, true)
  set_view(xyz(3.5049, -12.3042, 15.1668),
           xyz(-0.604036, 0.523037, 1.60932), 50)
  raw_view("prisms")
end)

# ── :surfaces ──────────────────────────────────────────────────────────

register_test("surface_polygons", :surfaces, () -> begin
  surface_polygon(xy(0, 0), xy(4, 0), xy(2, 3))
  surface_polygon(xy(5, 0), xy(9, 0), xy(9, 3), xy(5, 3))
  set_view(xyz(15, 15, 10), xyz(4, 0, 1), 50)
  zoom_extents()
  raw_view("surface_polygons")
end)

register_test("surface_circles", :surfaces, () -> begin
  surface_circle(u0(), 3)
  surface_circle(x(8), 2)
  zoom_2d_top()
  raw_view("surface_circles")
end)

register_test("surface_grid_sin", :surfaces, () -> begin
  let n = 20,
      pts = [xyz(x, y, sin(x) * cos(y))
             for x in range(0, 4pi, length=n),
                 y in range(0, 4pi, length=n)]
    surface_grid(pts)
  end
  set_view(xyz(30, 30, 20), xyz(6, 6, 0), 50)
  zoom_extents()
  raw_view("surface_grid_sin")
end)

register_test("surface_grid_ripple", :surfaces, () -> begin
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
  raw_view("surface_grid_ripple")
end)

register_test("surface_rectangles", :surfaces, () -> begin
  surface_rectangle(xy(0, 0), 5, 3)
  surface_rectangle(xy(6, 0), 2, 4)
  zoom_2d_top()
  raw_view("surface_rectangles")
end)

# ── :extrusions ────────────────────────────────────────────────────────

register_test("extrusion_rectangle", :extrusions, () -> begin
  extrusion(
    surface_polygon(
      reverse([xy(0, 2), xy(0, 5), xy(5, 3), xy(13, 3), xy(13, 6),
               xy(3, 6), xy(6, 7), xy(15, 7), xy(15, 2), xy(1, 2), xy(1, 0)])),
    1)
  set_view(xyz(-8.1453, -54.7236, 71.7539),
           xyz(8.2449, 8.2065, -5.1442), 199)
  raw_view("extrusion_rectangle")
end)

register_test("extrusion_circle", :extrusions, () -> begin
  extrusion(region(circular_path(u0(), 2)), vz(5))
  set_view(xyz(15, 15, 10), xyz(0, 0, 2.5), 50)
  zoom_extents()
  raw_view("extrusion_circle")
end)

register_test("revolve_line", :extrusions, () -> begin
  revolve(line(xyz(0, 0, 0), xyz(1, 1, 1), xyz(2, 0, 2)),
          x(5), vz(), 1//4*2*pi, 3//4*2*pi)
  set_view(xyz(19.6195, 51.9061, 19.8688),
           xyz(-0.3397, -0.6222, -1.0769), 199)
  raw_view("revolve_line")
end)

register_test("loft_circles", :extrusions, () -> begin
  loft([circle(xyz(0, 0, 0), 4),
        circle(xyz(0, 0, 2), 2),
        circle(xyz(0, 0, 4), 3)])
  loft([surface_circle(xyz(0, 9, 0), 4),
        surface_circle(xyz(0, 9, 2), 2),
        surface_circle(xyz(0, 9, 4), 3)])
  set_view(xyz(-77.8119, 4.1524, -50.357),
           xyz(-186.379, 4.89835, -120.792), 199.0)
  raw_view("loft_circles")
end)

register_test("sweep_circle_path", :extrusions, () -> begin
  sweep_path(
    open_spline_path([xyz(0,0,0), xyz(2,3,1), xyz(5,2,3), xyz(8,0,2)]),
    circular_path(u0(), 0.5))
  set_view(xyz(15, 15, 10), xyz(4, 1, 1.5), 50)
  zoom_extents()
  raw_view("sweep_circle_path")
end)

# ── :csg ───────────────────────────────────────────────────────────────

register_test("csg_union", :csg, () -> begin
  union(box(xyz(0, 0, 0), xyz(1, 1, 1)),
        sphere(xyz(1, 0, 1), 0.5))
  set_view(xyz(16.3249, -21.2313, 16.2413),
           xyz(2.1267, 1.3586, -0.1368), 200)
  raw_view("csg_union")
end)

register_test("csg_subtraction", :csg, () -> begin
  subtraction(box(xyz(4, 0, 0), xyz(5, 1, 1)),
              sphere(xyz(5, 0, 1), 0.5))
  set_view(xyz(16.3249, -21.2313, 16.2413),
           xyz(2.1267, 1.3586, -0.1368), 200)
  raw_view("csg_subtraction")
end)

register_test("csg_intersection", :csg, () -> begin
  intersection(box(xyz(2, 0, 0), xyz(3, 1, 1)),
               sphere(xyz(3, 0, 1), 0.5))
  set_view(xyz(16.3249, -21.2313, 16.2413),
           xyz(2.1267, 1.3586, -0.1368), 200)
  raw_view("csg_intersection")
end)

register_test("csg_compound", :csg, () -> begin
  subtraction(
    intersection([cylinder(xyz(-1, 0, 0), 1, xyz(1, 0, 0)),
                  cylinder(xyz(0, -1, 0), 1, xyz(0, 1, 0)),
                  cylinder(xyz(0, 0, -1), 1, xyz(0, 0, 1))]),
    sphere(xyz(0, 0, 0), 1.01))
  set_view(xyz(4.7056, -21.2502, 7.6505),
           xyz(-0.0838, 0.2606, -0.1048), 200)
  raw_view("csg_compound")
end)

# ── :parametric ────────────────────────────────────────────────────────

register_test("sierpinski", :parametric, () -> begin
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
  raw_view("sierpinski")
end)

register_test("column_grid", :parametric, () -> begin
  for x in 0:4:16
    for y in 0:4:16
      cylinder(xyz(x, y, 0), 0.3, 5)
    end
  end
  set_view(xyz(30, 30, 20), xyz(8, 8, 2.5), 50)
  zoom_extents()
  raw_view("column_grid")
end)

register_test("helix_boxes", :parametric, () -> begin
  for i in 0:30
    let a = i * pi/8,
        r = 5,
        z = i * 0.3
      box(xyz(r*cos(a), r*sin(a), z), 0.5, 0.5, 0.5)
    end
  end
  set_view(xyz(20, 20, 15), u0(), 50)
  zoom_extents()
  raw_view("helix_boxes")
end)

register_test("tower_floors", :parametric, () -> begin
  for i in 0:9
    let z = i * 3.0,
        s = 8 - i * 0.5,
        a = i * pi/20
      box(xyz(-s/2 * cos(a) - s/2 * sin(a),
              -s/2 * sin(a) + s/2 * cos(a),
              z),
          s, s, 0.3)
      for c in 0:3
        let ca = a + c * pi/2,
            cx = (s/2 - 0.3) * cos(ca),
            cy = (s/2 - 0.3) * sin(ca)
          cylinder(xyz(cx, cy, z), 0.2, 3)
        end
      end
    end
  end
  set_view(xyz(30, 30, 30), xyz(0, 0, 15), 50)
  zoom_extents()
  raw_view("tower_floors")
end)

# ── Comparison functions ──────────────────────────────────────────────

function text_compare(test_path, golden_path; kwargs...)
  read(test_path) == read(golden_path)
end

function pixel_diff_compare(test_path, golden_path; threshold=0.01)
  test_bytes = read(test_path)
  golden_bytes = read(golden_path)
  test_bytes == golden_bytes && return true
  # Future: pixel-level comparison using PNGFiles.jl
  # For now, byte-level comparison only
  false
end

# ── Test runner ───────────────────────────────────────────────────────

function run_visual_tests(b;
    golden_dir::String,
    reset!::Function,
    setup_backend::Function = () -> nothing,
    update_golden::Bool = get(ENV, "KHEPRI_UPDATE_GOLDEN", "0") == "1",
    width::Int = 960,
    height::Int = 540,
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

        if update_golden
          mkpath(golden_dir)
          cp(test_path, golden_path, force=true)
          @info "Updated golden file: $golden_path"
          @test true
        elseif isfile(golden_path)
          if compare(test_path, golden_path)
            @test true
          else
            @test false
            @warn "Visual regression detected for $name" test_path golden_path
          end
        else
          @warn "No golden file for $name — run with KHEPRI_UPDATE_GOLDEN=1"
          @test_broken false
        end
      end
    end
  end
end

end # module
