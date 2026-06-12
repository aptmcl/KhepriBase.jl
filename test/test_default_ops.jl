# test_default_ops.jl — rank-6 default b_* operation regression tests.
#
# Exercise the layered default operations against MockBackend (brought into scope
# by runtests.jl's `include("TestMockBackend.jl")` and `@import_backend_api`).
using Test

#=
Probe backends for the DEFAULT b_delete_all_shapes_in_layer.  The generic
fallback used to call `b_delete_shapes`, a function that exists nowhere, so
every backend that relied on the default chain (Revit, TikZ, POVRay, ...)
crashed with UndefVarError on the first delete_all_shapes_in_layer() call —
masked in the test suite because MockBackend and MinimalTriangleBackend both
override the op.  These probes deliberately do NOT override it, so the
default method body itself runs.

Two flavors, matching the two production paths through the default:
- LayerProbeGenericBackend exercises the plain-Backend legs
  (b_all_shapes_in_layer via `b.layers`, b_delete_shape via maybe_delete);
  the generic default assumes the backend populates `b.layers` itself, so
  the test does that explicitly.
- LayerProbeLocalBackend (a LocalBackend) exercises the LocalBackend legs
  (local layer index + filter!-based b_delete_shape) that file backends
  like TikZ and POVRay actually use.
=#
if !@isdefined(LayerProbeGenericKey)
  abstract type LayerProbeGenericKey end
  mutable struct LayerProbeGenericBackend <: KhepriBase.Backend{LayerProbeGenericKey, Int}
    layers::Dict{Any, Vector{KhepriBase.Proxy}}
    all_refs::Vector{Int}
    next_id::Int
    transaction::Parameter{KhepriBase.Transaction}
    refs::KhepriBase.References{LayerProbeGenericKey, Int}
  end
  LayerProbeGenericBackend() =
    LayerProbeGenericBackend(Dict{Any, Vector{KhepriBase.Proxy}}(), Int[], 1,
                             Parameter{KhepriBase.Transaction}(KhepriBase.AutoCommitTransaction()),
                             KhepriBase.References{LayerProbeGenericKey, Int}())
  KhepriBase.backend_name(b::LayerProbeGenericBackend) = "LayerProbeGeneric"
  KhepriBase.void_ref(b::LayerProbeGenericBackend) = 0
  KhepriBase.b_point(b::LayerProbeGenericBackend, p, mat) =
    let id = b.next_id
      b.next_id += 1
      push!(b.all_refs, id)
      id
    end
  KhepriBase.b_delete_ref(b::LayerProbeGenericBackend, r::Int) = filter!(!=(r), b.all_refs)
  KhepriBase.b_delete_refs(b::LayerProbeGenericBackend, rs::Vector{Int}) = filter!(r -> !(r in rs), b.all_refs)
  KhepriBase.b_get_material(b::LayerProbeGenericBackend, spec::Nothing) = 0
  KhepriBase.b_get_material(b::LayerProbeGenericBackend, spec) = 0
  KhepriBase.b_get_material(b::LayerProbeGenericBackend, ::KhepriBase.BackendDefault) = 0

  KhepriBase.@defbackend LayerProbeLocal LayerProbeLocal begin
    id_type = Any
    void_ref = -1
    parent = KhepriBase.LocalBackend
    mixin(local_shapes)
    mixin(io)
  end
end

@testset "Default ops (rank 6)" begin
  @testset "b_unite_refs on empty/single/multi (BACKEND-6/CONSIST-5)" begin
    b = MockBackend()
    # An empty union (degenerate ContourPath/Region/Mesh reaching here via
    # b_stroke_unite) used to crash with "reducing over an empty collection";
    # now it yields the empty-ref identity new_refs(b).
    @test KhepriBase.b_unite_refs(b, MockId[]) == KhepriBase.new_refs(b)
    # Single element is returned unchanged (the bare scalar ref).
    @test KhepriBase.b_unite_refs(b, MockId[7]) == 7
    # Multiple elements are collected (default b_unite_ref = vcat).
    @test KhepriBase.b_unite_refs(b, MockId[7, 8, 9]) == MockId[7, 8, 9]
  end

  @testset "b_text / b_text_size degrade on non-ASCII & control chars (BACKEND-7)" begin
    b = MockBackend()
    m = KhepriBase.void_ref(b)
    @test haskey(KhepriBase.letter_glyph, '?')   # the .notdef fallback glyph exists
    # Unknown characters (accented, Greek, tab, newline) must degrade to '?'
    # instead of raising KeyError, in BOTH the draw and measure ops.
    for s in ("é", "café", "\t", "\n", "α")
      @test (KhepriBase.b_text(b, s, u0(), 1.0, m); true)
      @test (KhepriBase.b_text_size(b, s, 1.0, m); true)
    end
    # A single unknown char measures exactly like the '?' fallback.
    @test KhepriBase.b_text_size(b, "é", 1.0, m) == KhepriBase.b_text_size(b, "?", 1.0, m)
  end

  @testset "default b_delete_all_shapes_in_layer (generic Backend chain)" begin
    let b = LayerProbeGenericBackend()
      with(KhepriBase.current_backends, (b,)) do
        let lay = KhepriBase.b_layer(b, "A", true, rgba(1, 1, 1, 1)),
            p1 = point(u0()),
            p2 = point(ux())
          b.layers[lay] = KhepriBase.Proxy[p1, p2]
          @test length(b.all_refs) == 2
          @test !KhepriBase.marked_deleted(b, p1)
          # Must not raise UndefVarError (the old body called the nonexistent
          # b_delete_shapes) and must delete both native and proxy refs.
          KhepriBase.b_delete_all_shapes_in_layer(b, lay)
          @test isempty(b.all_refs)
          @test KhepriBase.marked_deleted(b, p1)
          @test KhepriBase.marked_deleted(b, p2)
        end
      end
    end
  end

  @testset "default b_delete_all_shapes_in_layer (LocalBackend chain)" begin
    let b = LayerProbeLocal()
      with(KhepriBase.current_backends, (b,)) do
        let la = layer("ProbeA"),
            lb = layer("ProbeB")
          current_layer(la)
          point(u0())
          point(ux())
          current_layer(lb)
          let p3 = point(uy()),
              la_ref = KhepriBase.ref_value(b, la),
              lb_ref = KhepriBase.ref_value(b, lb)
            @test length(KhepriBase.b_all_shapes_in_layer(b, la_ref)) == 2
            # Frontend entry point; dispatches to the DEFAULT op (LayerProbeLocal
            # overrides neither b_delete_all_shapes_in_layer nor b_delete_shape).
            delete_all_shapes_in_layer(la)
            @test isempty(KhepriBase.b_all_shapes_in_layer(b, la_ref))
            # The other layer and its shape must be untouched.
            @test length(KhepriBase.b_all_shapes_in_layer(b, lb_ref)) == 1
            @test KhepriBase.local_shape_storage(b) == [p3]
          end
        end
      end
    end
  end
end
