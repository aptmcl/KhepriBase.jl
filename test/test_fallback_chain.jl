# test_fallback_chain.jl - Verify that the fallback algebra correctly
# decomposes higher-level operations down to b_trig.
#
# A MinimalTriangleBackend implements ONLY b_trig. All higher-level
# operations (surfaces, solids) must decompose through the fallback chain.

using KhepriBase
import KhepriBase: Backend, void_ref, new_refs, backend_name,
  current_transaction, Transaction, AutoCommitTransaction,
  References, GenericRef, NativeRef, b_trig,
  b_surface_polygon, b_surface_circle, b_surface_rectangle,
  b_box, b_sphere, b_cylinder, b_cone, b_torus,
  b_cuboid, b_prism, b_regular_pyramid, b_regular_prism,
  b_delete_ref, b_delete_refs, b_delete_all_shape_refs,
  b_layer, b_current_layer_ref, b_delete_all_shapes_in_layer,
  b_create_layer_from_ref_value,
  b_get_material, b_material

# Minimal backend key
struct MinTriKey end
const MinTriId = Int

mutable struct MinimalTriangleBackend <: Backend{MinTriKey, MinTriId}
  triangles::Vector{Tuple{Loc, Loc, Loc}}
  next_id::Int
  all_refs::Vector{MinTriId}
  transaction::Parameter{Transaction}
  refs::References{MinTriKey, MinTriId}
end

MinimalTriangleBackend() = MinimalTriangleBackend(
  Tuple{Loc,Loc,Loc}[],
  1,
  MinTriId[],
  Parameter{Transaction}(AutoCommitTransaction()),
  References{MinTriKey, MinTriId}())

function next_ref!(b::MinimalTriangleBackend)
  id = b.next_id
  b.next_id += 1
  push!(b.all_refs, id)
  id
end

# Required backend protocol
KhepriBase.backend_name(b::MinimalTriangleBackend) = "MinimalTriangle"
KhepriBase.void_ref(b::MinimalTriangleBackend) = 0
KhepriBase.new_refs(b::MinimalTriangleBackend) = MinTriId[]
KhepriBase.current_transaction(b::MinimalTriangleBackend) = b.transaction

# The ONLY geometry operation: b_trig
b_trig(b::MinimalTriangleBackend, p1, p2, p3) = begin
  push!(b.triangles, (p1, p2, p3))
  next_ref!(b)
end

# Deletion stubs
b_delete_ref(b::MinimalTriangleBackend, r::MinTriId) = filter!(x -> x != r, b.all_refs)
b_delete_refs(b::MinimalTriangleBackend, rs::Vector{MinTriId}) = filter!(x -> !(x in rs), b.all_refs)

# Layer stubs
KhepriBase.b_layer(b::MinimalTriangleBackend, name, active, color) = next_ref!(b)
KhepriBase.b_current_layer_ref(b::MinimalTriangleBackend) = 0
KhepriBase.b_current_layer_ref(b::MinimalTriangleBackend, r) = nothing
KhepriBase.b_delete_all_shapes_in_layer(b::MinimalTriangleBackend, layer) = nothing
KhepriBase.b_create_layer_from_ref_value(b::MinimalTriangleBackend, r) = layer("Default")

# Material stubs
KhepriBase.b_get_material(b::MinimalTriangleBackend, spec::Nothing) = 0
KhepriBase.b_get_material(b::MinimalTriangleBackend, spec) = 0
KhepriBase.b_get_material(b::MinimalTriangleBackend, ::BackendDefault) = 0
KhepriBase.b_material(b::MinimalTriangleBackend, name, base_color) =
  next_ref!(b)

function reset!(b::MinimalTriangleBackend)
  empty!(b.triangles)
  empty!(b.all_refs)
  empty!(b.refs.shapes)
  empty!(b.refs.materials)
  empty!(b.refs.layers)
  empty!(b.refs.annotations)
  empty!(b.refs.families)
  empty!(b.refs.levels)
  b.next_id = 1
  b
end

@testset "Fallback Chain (MinimalTriangleBackend)" begin
  b = MinimalTriangleBackend()
  vr = void_ref(b)

  @testset "b_trig directly" begin
    reset!(b)
    r = b_trig(b, u0(), ux(), uy())
    @test r != vr
    @test length(b.triangles) == 1
  end

  @testset "b_surface_polygon → triangles" begin
    reset!(b)
    # A square polygon should decompose into 2 triangles
    rs = b_surface_polygon(b, [u0(), ux(), uxy(), uy()], vr)
    @test length(b.triangles) == 2
  end

  @testset "b_surface_polygon pentagon → triangles" begin
    reset!(b)
    pts = [xyz(cos(2π*i/5), sin(2π*i/5), 0) for i in 0:4]
    rs = b_surface_polygon(b, pts, vr)
    @test length(b.triangles) == 3  # 5-gon → 3 triangles
  end

  @testset "b_box → triangles" begin
    reset!(b)
    r = b_box(b, u0(), 5.0, 5.0, 5.0, vr)
    # A box has 6 faces, each decomposed to 2 triangles = 12 minimum
    # (quad_strip_closed for sides = 4 faces × 2 trigs = 8, plus 2 caps × 2 trigs = 4)
    @test length(b.triangles) >= 12
  end

  @testset "b_sphere → triangles" begin
    reset!(b)
    r = b_sphere(b, u0(), 5.0, vr)
    # A 32-segment sphere should produce many triangles
    # Top cap: 32 trigs, bottom cap: 32 trigs, 31 bands × 32 quads × 2 = 1984
    # Total should be well over 100
    @test length(b.triangles) > 100
  end

  @testset "b_cylinder → triangles" begin
    reset!(b)
    r = b_cylinder(b, u0(), 3.0, 10.0, vr, vr, vr)
    # Cylinder with 32 segments: top cap + bottom cap + 32 side quads
    @test length(b.triangles) > 60
  end

  @testset "b_cone → triangles" begin
    reset!(b)
    r = b_cone(b, u0(), 5.0, 10.0, vr, vr)
    # Cone: base cap (32 trigs from ngon) + side (32 trigs from ngon)
    @test length(b.triangles) >= 32
  end

  @testset "b_torus → triangles" begin
    reset!(b)
    r = b_torus(b, u0(), 10.0, 3.0, vr)
    # Torus should produce a grid of quads decomposed to triangles
    @test length(b.triangles) > 100
  end

  @testset "b_prism → triangles" begin
    reset!(b)
    r = b_prism(b, [u0(), ux(), uxy()], vz(5.0), vr, vr, vr)
    # Triangle prism: 2 triangular caps + 3 quad sides = 2 + 6 = 8 triangles
    @test length(b.triangles) >= 8
  end

  @testset "high-level sphere() → triangles" begin
    reset!(b)
    backend(b)
    s = sphere(u0(), 5)
    @test length(b.triangles) > 100
  end

  @testset "high-level box() → triangles" begin
    reset!(b)
    backend(b)
    s = box(u0(), 5, 5, 5)
    @test length(b.triangles) >= 12
  end

  @testset "high-level cylinder() → triangles" begin
    reset!(b)
    backend(b)
    s = cylinder(u0(), 3, 10)
    @test length(b.triangles) > 60
  end

  @testset "high-level surface_polygon() → triangles" begin
    reset!(b)
    backend(b)
    s = surface_polygon([u0(), ux(), uxy(), uy()])
    @test length(b.triangles) == 2
  end

  @testset "triangle count consistency" begin
    # Running the same operation twice should produce the same triangle count
    reset!(b)
    b_sphere(b, u0(), 1.0, vr)
    count1 = length(b.triangles)

    reset!(b)
    b_sphere(b, u0(), 1.0, vr)
    count2 = length(b.triangles)

    @test count1 == count2
    @test count1 > 0
  end
end
