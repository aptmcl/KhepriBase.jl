# test_render_view_options.jl — canonical visual-style vocab + RenderViewOptions contract.

using KhepriBase
using Test

# Extend MockBackend with a render method that records what happened so we can
# assert the default 3-arg → 2-arg delegation threads Parameters correctly.
if !@isdefined(_mock_render_calls)
  const _mock_render_calls = Ref{Vector{NamedTuple}}(NamedTuple[])
end

KhepriBase.b_render_and_save_view(b::MockBackend, path::String) =
  push!(_mock_render_calls[],
        (path=path,
         width=render_width(),
         height=render_height(),
         quality=Float64(render_quality()),
         exposure=Float64(render_exposure()),
         kind=render_kind()))

KhepriBase.b_render_pathname(::MockBackend, name) = "/tmp/khepri_test_$(name).png"

@testset "canonical_visual_styles vocabulary" begin
  expected = (:wireframe, :shaded, :realistic, :arctic,
              :technical, :pen, :sketchy, :xray, :ghosted)
  @test canonical_visual_styles == expected
  @test :arctic in canonical_visual_styles
  @test :realistic in canonical_visual_styles
end

@testset "validate_visual_style" begin
  # All canonical styles pass.
  for s in canonical_visual_styles
    @test KhepriBase.validate_visual_style(s) == true || KhepriBase.validate_visual_style(s) === nothing
  end
  # Unknown style raises.
  @test_throws ErrorException KhepriBase.validate_visual_style(:bogus)
  @test_throws ErrorException KhepriBase.validate_visual_style(:cartoon)
end

@testset "RenderViewOptions defaults and overrides" begin
  # Defaults come from the current Parameters.
  opts = RenderViewOptions()
  @test opts.width == render_width()
  @test opts.height == render_height()
  @test opts.quality == Float64(render_quality())
  @test opts.exposure == Float64(render_exposure())
  @test opts.visual_style == :shaded
  @test opts.kind == render_kind()
  @test opts.extra == Dict{Symbol,Any}()

  # Explicit overrides.
  opts2 = RenderViewOptions(width=1920, height=1080, quality=0.5,
                            exposure=-1.0, visual_style=:arctic, kind=:white,
                            extra=Dict{Symbol,Any}(:samples => 256))
  @test opts2.width == 1920
  @test opts2.height == 1080
  @test opts2.quality == 0.5
  @test opts2.exposure == -1.0
  @test opts2.visual_style == :arctic
  @test opts2.kind == :white
  @test opts2.extra[:samples] == 256
end

@testset "Defaults track Parameter overrides via with()" begin
  # When a caller has wrapped rendering_with(...) / with(render_width, ...),
  # the struct must see the overridden values.
  with(render_width, 4096) do
    with(render_height, 2160) do
      with(render_quality, 0.9) do
        opts = RenderViewOptions()
        @test opts.width == 4096
        @test opts.height == 2160
        @test opts.quality == 0.9
      end
    end
  end
end

@testset "3-arg b_render_and_save_view threads opts into Parameters" begin
  # Backend implementing only the legacy 2-arg method must still receive the
  # opts via Parameters when called via the 3-arg default.
  b = mock_backend()
  reset_mock_backend!(b)
  empty!(_mock_render_calls[])

  opts = RenderViewOptions(width=640, height=480, quality=-0.5,
                           exposure=1.5, kind=:white, visual_style=:shaded)
  KhepriBase.b_render_and_save_view(b, "/tmp/out.png", opts)

  @test length(_mock_render_calls[]) == 1
  c = _mock_render_calls[][1]
  @test c.path == "/tmp/out.png"
  @test c.width == 640
  @test c.height == 480
  @test c.quality == -0.5
  @test c.exposure == 1.5
  @test c.kind == :white

  # After the call, the Parameters are restored.
  @test render_width() == 1024
  @test render_height() == 768
  @test render_kind() == :realistic
end

@testset "b_render_view builds default opts when none given" begin
  b = mock_backend()
  reset_mock_backend!(b)
  empty!(_mock_render_calls[])

  KhepriBase.b_render_view(b, "no_opts")

  @test length(_mock_render_calls[]) == 1
  c = _mock_render_calls[][1]
  @test c.path == "/tmp/khepri_test_no_opts.png"
  @test c.width == render_width()   # took default
  @test c.kind == render_kind()
end

@testset "b_render_view rejects unknown visual_style" begin
  b = mock_backend()
  reset_mock_backend!(b)
  opts = RenderViewOptions(visual_style=:bogus)
  @test_throws ErrorException KhepriBase.b_render_view(b, "bad", opts)
end

@testset "render_view kwargs thread through" begin
  b = mock_backend()
  reset_mock_backend!(b)
  empty!(_mock_render_calls[])

  # User-facing render_view accepts any RenderViewOptions kwargs.
  with(current_backend, b) do
    render_view("kw_thread", b; width=800, height=600,
                 visual_style=:arctic, quality=0.3, exposure=-0.5)
  end

  @test length(_mock_render_calls[]) == 1
  c = _mock_render_calls[][1]
  @test c.width == 800
  @test c.height == 600
  @test c.quality == 0.3
  @test c.exposure == -0.5
end
