# test_profile_consumption.jl — regression tests for construction-time
# consumption of extrusion/sweep profile shapes.
#
# The Union-typed profile fields (705d556) keep the input Shape alive until
# realize time so native backends can map_ref it. On LocalBackend that used
# to (a) LEAK the profile — it realized standalone, ahead of its consumer —
# and (b) DROP later shapes — the realize-time convert → and_delete_shape
# filter!ed local_shape_storage while realize_shapes iterated it, shifting
# indices and skipping the shapes that followed. These tests pin the fixed
# contract from both sides: storage backends consume at construction, native
# backends still see a live, realized profile.
using Test

#=
Probe backends.
- ProfileProbeLocal (a LocalBackend, like TikZ/POVRay) exercises the
  queue-then-realize path where the leak and the drop manifested. b_trig /
  b_line record realization counts so a leaked profile (extra cap trigs) or
  a skipped shape (missing line) is detected arithmetically.
- ProfileNativeProbeBackend (a plain AutoCommit Backend, like AutoCAD/Rhino)
  proves the other side of the contract: consumption must NOT destroy the
  profile's refs — the native b_extruded_surface override must still receive
  the live Shape proxy and walk its existing refs via map_ref, mirroring
  acad_extrude (KhepriAutoCAD/src/AutoCAD.jl).
=#
if !@isdefined(ProfileProbeLocal)
  KhepriBase.@defbackend ProfileProbeLocal ProfileProbeLocal begin
    id_type = Any
    void_ref = -1
    parent = KhepriBase.LocalBackend
    mixin(local_shapes)
    mixin(io)
  end
  const PROFILE_PROBE_TRIGS = Ref(0)
  const PROFILE_PROBE_LINES = Ref(0)
  KhepriBase.b_trig(b::ProfileProbeLocal, p1, p2, p3, mat) =
    (PROFILE_PROBE_TRIGS[] += 1)
  KhepriBase.b_line(b::ProfileProbeLocal, ps, mat) =
    (PROFILE_PROBE_LINES[] += 1)

  struct ProfileNativeProbeKey end
  mutable struct ProfileNativeProbeBackend <: KhepriBase.Backend{ProfileNativeProbeKey, Int}
    next_id::Int
    extruded_profile_was_realized::Vector{Bool}
    transaction::Parameter{KhepriBase.Transaction}
    refs::KhepriBase.References{ProfileNativeProbeKey, Int}
  end
  ProfileNativeProbeBackend() =
    ProfileNativeProbeBackend(1, Bool[],
                              Parameter{KhepriBase.Transaction}(KhepriBase.AutoCommitTransaction()),
                              KhepriBase.References{ProfileNativeProbeKey, Int}())
  KhepriBase.backend_name(b::ProfileNativeProbeBackend) = "ProfileNativeProbe"
  KhepriBase.void_ref(b::ProfileNativeProbeBackend) = 0
  profile_native_ref!(b::ProfileNativeProbeBackend) =
    let id = b.next_id
      b.next_id += 1
      id
    end
  KhepriBase.b_trig(b::ProfileNativeProbeBackend, p1, p2, p3) =
    profile_native_ref!(b)
  # Native shape-aware extrusion, mirroring acad_extrude: receive the LIVE
  # Shape2D proxy, walk its refs via map_ref, and mark it deleted afterwards.
  KhepriBase.b_extruded_surface(b::ProfileNativeProbeBackend, s::KhepriBase.Shape2D, v, cb, bmat, tmat, smat) =
    begin
      push!(b.extruded_profile_was_realized, KhepriBase.realized(b, s))
      KhepriBase.and_mark_deleted(b,
        KhepriBase.map_ref((r) -> profile_native_ref!(b), b, KhepriBase.ref(b, s)),
        s)
    end
end

@testset "Profile consumption (extrusion/sweep)" begin
  @testset "LocalBackend: profile consumed at construction, no leak" begin
    let b = ProfileProbeLocal()
      with(KhepriBase.current_backends, (b,)) do
        PROFILE_PROBE_TRIGS[] = 0
        let m = 11,
            p = surface_polygon([xyz(cos(2π*i/m), sin(2π*i/m), 0) for i in 0:m-1]),
            e = extrusion(p, vz(1))
          # The proxy keeps the live Shape (the 705d556 native-backend win) ...
          @test e.profile === p
          # ... but the profile is unqueued at construction: it must not sit in
          # the realize queue ahead of its consumer.
          @test !any(q -> q === p, KhepriBase.local_shape_storage(b))
          @test any(q -> q === e, KhepriBase.local_shape_storage(b))
          KhepriBase.realize_shapes(b)
          # The profile never realizes standalone ...
          @test !KhepriBase.realized(b, p)
          @test KhepriBase.realized(b, e)
          # ... and the output holds EXACTLY the extrusion's triangles:
          # 11 wall quads (12-vertex open path) ×2 + two 11-gon caps ×(11-2).
          # A leaked profile would add another 9-trig cap.
          @test PROFILE_PROBE_TRIGS[] == 22 + 2*9
        end
      end
    end
  end

  @testset "LocalBackend: realize-time frontier deletion skips nothing" begin
    let b = ProfileProbeLocal()
      with(KhepriBase.current_backends, (b,)) do
        PROFILE_PROBE_LINES[] = 0
        # predioSinusoidal minimal pattern: a surface() whose realize deletes
        # its boundary curves from the queue MID-PASS, followed by shapes that
        # the index shift used to skip silently.
        let l1 = line([u0(), ux(), uy()]),
            l2 = line([uy(), u0()]),
            s = surface([l1, l2]),
            l4 = line([xyz(2, 0, 0), xyz(3, 0, 0)]),
            l5 = line([xyz(4, 0, 0), xyz(5, 0, 0)])
          @test length(KhepriBase.local_shape_storage(b)) == 5
          KhepriBase.realize_shapes(b)
          # b_surface's and_delete_shapes removed the frontier from the queue
          # (blessed behavior — their draws appear, then the shapes unqueue) ...
          @test length(KhepriBase.local_shape_storage(b)) == 3
          # ... but NOTHING still queued may have been skipped by the shift.
          @test all(q -> KhepriBase.realized(b, q), KhepriBase.local_shape_storage(b))
          # All four lines realized: 2 frontier curves + the 2 trailing lines
          # that the unguarded iteration used to drop.
          @test PROFILE_PROBE_LINES[] == 4
        end
      end
    end
  end

  @testset "Native backend: consumed profile keeps live refs for map_ref" begin
    let b = ProfileNativeProbeBackend()
      with(KhepriBase.current_backends, (b,)) do
        let m = 5,
            p = surface_polygon([xyz(cos(2π*i/m), sin(2π*i/m), 0) for i in 0:m-1]),
            e = extrusion(p, vz(1))
          @test e.profile === p
          # b_consume_shape is a no-op here: the override received the live,
          # already-realized proxy and map_ref over its refs succeeded.
          @test b.extruded_profile_was_realized == [true]
          @test KhepriBase.realized(b, e)
          # and_mark_deleted ran: the profile is marked consumed, not destroyed.
          @test KhepriBase.marked_deleted(b, p)
        end
      end
    end
  end
end
