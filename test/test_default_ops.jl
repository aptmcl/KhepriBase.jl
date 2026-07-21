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

#=
Probe backend for the DEFAULT b_spline chain. The default (Backend.jl) draws
the canonical chord-parameterized cubic via open_spline_bezier_path (Paths.jl)
and bottoms out at b_line through sampled_curve. The hook and the helpers once
went out of lockstep — a committed default referenced open_spline_* that
existed only in an uncommitted working tree, so every backend without a native
b_spline override threw UndefVarError at draw time and no test noticed,
because MockBackend overrides b_spline. This probe deliberately does NOT
override it, so the default method body itself runs.
=#
if !@isdefined(SplineProbeKey)
  abstract type SplineProbeKey end
  mutable struct SplineProbeBackend <: KhepriBase.Backend{SplineProbeKey, Int}
    lines::Vector{Vector{KhepriBase.Loc}}
    refs::KhepriBase.References{SplineProbeKey, Int}
  end
  SplineProbeBackend() =
    SplineProbeBackend(Vector{KhepriBase.Loc}[], KhepriBase.References{SplineProbeKey, Int}())
  KhepriBase.backend_name(b::SplineProbeBackend) = "SplineProbe"
  KhepriBase.void_ref(b::SplineProbeBackend) = 0
  KhepriBase.b_line(b::SplineProbeBackend, ps, mat) = begin
    push!(b.lines, collect(ps))
    length(b.lines)
  end
end

@testset "default b_spline draws the canonical curve (helper/hook lockstep)" begin
  let ps = [xy(0, 0), xy(1, 2), xy(3, 1), xy(5, 3), xy(7, 0)],
      sample_step = 9.5 / max(path_smoothness_segments(), (length(ps) - 1) * 8)
    # No tangents: must realize through open_spline_bezier_path without error
    # and pass near every interpolation point (within the sampling step).
    let b = SplineProbeBackend()
      KhepriBase.b_spline(b, ps, false, false, 0)
      @test length(b.lines) == 1
      let vs = b.lines[1]
        @test length(vs) >= length(ps)
        for p in ps
          @test minimum(distance(p, v) for v in vs) < sample_step
        end
        # Independent oracle: every drawn vertex lies on the Dierckx
        # chord-parameterized interpolant that location_at/sweeps follow.
        let ci = KhepriBase.curve_interpolator(ps, false),
            dense = [xyz(KhepriBase.Dierckx.evaluate(ci, t)..., world_cs)
                     for t in 0:0.00005:1]
          @test maximum(minimum(distance(v, q) for q in dense) for v in vs) < 1e-3
        end
      end
    end
    # Supplied FORWARD tangents: the drawn polyline leaves along v0 and
    # arrives along v1 (the end-of-spline wiggle regression, default chain).
    # The sampled vertices are oriented frames (z along the curve tangent), so
    # direction math must happen on their WORLD positions.
    let b = SplineProbeBackend()
      KhepriBase.b_spline(b, ps, vxy(1, 0), vxy(0, -1), 0)
      let vs = [KhepriBase.in_world(v) for v in b.lines[1]],
          d0 = KhepriBase.unitized(vs[2] - vs[1]),
          d1 = KhepriBase.unitized(vs[end] - vs[end-1])
        @test d0.x > 0.98
        @test d1.y < -0.98
      end
    end
  end
end

#=
Family profiles are section data: a profile path constructed under a non-world
current_cs (a group factory body, a with(current_cs, ...) block) used to drag
the ambient transform into every member placement — path_on/path_vertices_on
replant a profile by its WORLD coordinates, so the captured cs was applied ON
TOP of the member frame (a group's member column landed at instance offset
DOUBLED). family_profile now normalizes through section_in_world.
=#
@testset "family profiles are cs-free section data" begin
  measured_centroids(f) =
    let mb = KhepriBase.measure_backend(),
        prev = KhepriBase.current_backends()
      KhepriBase.current_backend(mb)
      try
        f()
      finally
        KhepriBase.current_backends(prev)
      end
      [m.measurement.centroid for m in KhepriBase.measured_shapes(mb)
       if m.measurement.n_trigs > 0]
    end
  # Direct: family (with profile) created inside a translated cs.
  let cs = KhepriBase.translated_cs(KhepriBase.world_cs, 2.0, 1.0, 0.0),
      centroids = measured_centroids() do
        with(KhepriBase.current_cs, cs) do
          column(xy(0.2, 0.2), 0, level(0.0), level(3.0),
                 column_family("cf", rectangular_path(xy(-0.1, -0.1), 0.2, 0.2)))
        end
      end
    @test length(centroids) == 1
    @test KhepriBase.distance(centroids[1], xyz(2.2, 1.2, 1.5)) < 1e-9
  end
  # Through the group machinery (the emitted-program shape: inline family in the factory).
  let prog = Meta.parseall("""
        function fac()
          column(xy(0.2, 0.2), 0, level(0.0), level(3.0),
                 column_family("cf", rectangular_path(xy(-0.1, -0.1), 0.2, 0.2)))
        end
        g = group("desk", factory=fac)
        group_instance(g, xyz(2.0, 1.0, 0.0))
        group_instance(g, xyz(8.0, 1.0, 0.0))
        finalize_groups()
        """),
      mb = KhepriBase.measure_backend(),
      rows = KhepriBase.measured_statements(prog; b=mb),
      centroids = sort([m.measurement.centroid for m in KhepriBase.measured_shapes(mb)
                        if m.measurement.n_trigs > 0], by=p -> KhepriBase.cx(p))
    @test all(r -> r.error === nothing, rows)
    @test length(centroids) == 2
    @test KhepriBase.distance(centroids[1], xyz(2.2, 1.2, 1.5)) < 1e-9
    @test KhepriBase.distance(centroids[2], xyz(8.2, 1.2, 1.5)) < 1e-9
  end
end
