using KhepriBase
using Test

# Verify KhepriBase's two-tier API split.
# - User-facing names: `export`ed. End users see these via `using KhepriBackend`.
# - Developer-facing names: `public` (not exported). Pulled into backend modules
#   by `@import_backend_api`. Users of backends do NOT see these.

@testset "API surface: KhepriBase export/public split" begin
  @testset "User-facing names are exported" begin
    for n in (:sphere, :box, :cylinder, :cone, :torus,
              :point, :line, :polygon, :circle, :arc,
              :surface_polygon, :surface_rectangle,
              :xyz, :u0, :ux, :uy, :uz,
              :RGB, :RGBA, :rgb, :rgba,
              :material, :layer,
              :render_view, :set_view, :zoom_extents,
              :delete_all_shapes, :delete_shape,
              :is_sphere, :sphere_center, :sphere_radius,
              :is_box,
              :with, :Parameter,
              :random, :division, :series,
              Symbol("@backend"), Symbol("@backends"),
              Symbol("@remote"), Symbol("@get_remote"),
              :autocad_port, :rhino_port)
      @test Base.isexported(KhepriBase, n)
    end
  end

  @testset "Developer-facing names are public but not exported" begin
    for n in (:b_sphere, :b_trig, :b_quad, :b_point, :b_line,
              :b_polygon, :b_surface_polygon, :b_surface_grid,
              :b_box, :b_cylinder, :b_cone, :b_torus,
              :b_material, :b_layer, :b_current_layer_ref,
              :b_render_view, :b_zoom_extents, :b_set_view,
              :realize, :mark_deleted, :force_realize, :ref_value,
              :NativeRef, :GenericRef, :NativeRefs, :References,
              :SocketBackend, :WebSocketBackend, :RemoteBackend,
              :LocalBackend, :IOBackend, :LazyBackend,
              :void_ref, :new_refs,
              Symbol("@remote_api"), Symbol("@encode_decode_as"),
              Symbol("@defproxy"), Symbol("@deffamily"), Symbol("@defbackend"),
              Symbol("@bdef"), Symbol("@ifbackend"),
              :parse_signature, :encode, :decode,
              :current_backends, :top_backend, :purge_backends)
      @test Base.ispublic(KhepriBase, n)
      @test !Base.isexported(KhepriBase, n)
    end
  end

  @testset "@import_backend_api brings dev names into scope" begin
    # _BACKEND_API_NAMES must be non-empty
    @test !isempty(KhepriBase._BACKEND_API_NAMES)
    # A representative sample should be there
    for n in (:b_sphere, :b_trig, :realize, :NativeRef, :SocketBackend)
      @test n in KhepriBase._BACKEND_API_NAMES
    end
    # Nothing exported should be in there
    @test !(:sphere in KhepriBase._BACKEND_API_NAMES)
    @test !(:xyz in KhepriBase._BACKEND_API_NAMES)
    @test !(:material in KhepriBase._BACKEND_API_NAMES)
  end
end
