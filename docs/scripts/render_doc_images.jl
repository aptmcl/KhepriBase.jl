#=
Driver for generating every figure referenced in docs/src/*.md.

Each entry in `SCENES` is a self-contained recipe: the backend it
targets (:svg for schematic 2D, :blender for 3D), the target
filename under docs/src/assets, a view preset, and a builder
function that constructs the scene.

Run:
    julia.exe --project=docs/scripts docs/scripts/render_doc_images.jl [section-pattern]

With no argument, renders every scene.  With an argument, renders
only scenes whose id matches (substring).  Example:

    julia.exe --project=docs/scripts docs/scripts/render_doc_images.jl concepts

Blender runs once and is kept alive for the whole batch; the SVG
backend writes file output per scene and needs no setup.
=#

const ASSETS = abspath(joinpath(@__DIR__, "..", "src", "assets"))

using Dates
using KhepriBase
using KhepriSVG
using KhepriBlender

include("presets.jl")

#=
Reset the backend to an empty-scene state between scenes.  Going through
the top-level `delete_all_shapes()` hits a pre-compiled LocalBackend
branch that assumes a `b.shapes` field; SVGBackend exposes that via
getproperty-forwarding and the inlined call site misses it.  Resetting
the mixin fields directly is equivalent and backend-agnostic.
=#
reset_backend_state!(b) = begin
  if hasfield(typeof(b), :_local_shapes)
    empty!(b._local_shapes.shapes)
    empty!(b._local_shapes.layers)
  end
  if hasfield(typeof(b), :refs) && hasfield(typeof(b.refs), :shapes)
    empty!(b.refs.shapes)
  end
  try
    delete_all_shapes()
  catch err
    @debug "delete_all_shapes fallthrough errored, ignored" err
  end
  nothing
end

include("scenes/concepts.jl")
include("scenes/bim.jl")
include("scenes/tutorials.jl")
include("scenes/reference.jl")

# ---------------------------------------------------------------
# Runner
# ---------------------------------------------------------------

function render_svg_scenes(scenes)
  isempty(scenes) && return
  println("--- Rendering ", length(scenes), " SVG scenes ---")
  # Pin the SVG instance as the sole current backend for every scene.
  with(KhepriBase.current_backends, (KhepriSVG.svg,)) do
    for s in scenes
      render_one(s)
    end
  end
end

function render_blender_scenes(scenes)
  isempty(scenes) && return
  println("--- Rendering ", length(scenes), " Blender scenes ---")
  # KhepriBlender uses a reverse connection: start_blender() spawns Blender
  # which connects back to our TCP server on blender_port.  Wait for the
  # live SocketBackend to show up in current_backends(), then pin it.
  KhepriBlender.start_blender()
  deadline = time() + 180.0
  live = nothing
  # Read the global default directly — task-local `current_backends()` is
  # pinned to the SVG-phase value by the time we get here, and wouldn't
  # see the Blender backend added by add_global_backend in the server task.
  while time() < deadline
    bs = KhepriBase.current_backends.value
    for b in bs
      if b isa KhepriBase.SocketBackend
        live = b
        break
      end
    end
    live !== nothing && break
    sleep(0.5)
  end
  live === nothing && error("Blender connection never landed after 180s.")
  println("Blender connected: ", KhepriBase.backend_name(live))
  with(KhepriBase.current_backends, (live,)) do
    for s in scenes
      render_one(s)
    end
  end
end

#=
Per-scene lighting for Blender.  The spawned headless Blender has an
empty world with no sun, no environment HDR — every shot would be
black without a manual rig.  `b_realistic_sky` installs a Blender
Sky Texture with a sun disc at a fixed elevation + rotation, which
matches what the user sees in an interactive Blender session with
"Sky" enabled.  Angles are in degrees (the backend converts).

`render_exposure=1.2` adds ~1.2 stops of post-exposure to compensate
for the Eevee clay defaults; `render_quality=0.5` lifts the Eevee
sample count out of the muddy-noise range without pushing render
time past a few seconds per scene.
=#
#=
Headless Blender ships a dark default world.  We try to install a
Sky Texture with a sun disc first; when the Blender build is old
enough that `NISHITA` is unavailable (only `PREETHAM`/`HOSEK_WILKIE`
exist in ≤ 3.x), we fall back to an explicit sun-light + pointlight
rig that's independent of world-shader features.
=#
function setup_blender_lighting!(b)
  try
    KhepriBase.b_realistic_sky(b, 55, 135, 3, true)
    return
  catch
    # fall through to explicit lights
  end
  try
    KhepriBase.b_pointlight(b, xyz(10, -10, 15), 8000.0,
                             RGB(1.0, 0.96, 0.90))
    KhepriBase.b_pointlight(b, xyz(-6, -8, 12), 4000.0,
                             RGB(0.85, 0.9, 1.0))
  catch err
    @warn "fallback lighting setup failed" err
  end
end

function render_one(s)
  dir = abspath(joinpath(ASSETS, s.section))
  mkpath(dir)
  name = replace(s.filename, r"\.(svg|png)$" => "")
  println("  [", s.backend, "] ", s.section, "/", s.filename)
  with(render_dir, dir) do
  with(render_user_dir, ".") do
  with(render_backend_dir, ".") do
  with(render_kind_dir, ".") do
  with(render_color_dir, ".") do
  with(render_width, s.width) do
  with(render_height, s.height) do
  with(render_exposure, s.backend == :blender ? 1.2 : 0.0) do
  with(render_quality, s.backend == :blender ? 0.5 : 0.0) do
    try
      b = KhepriBase.top_backend()
      reset_backend_state!(b)
      if s.backend == :blender
        setup_blender_lighting!(b)
      end
      s.build()
      if s.view !== nothing
        try
          set_view(s.view.eye, s.view.target)
        catch err
          @debug "set_view failed (expected on SVG)" err
        end
      end
      if s.backend == :blender
        render_view(name; visual_style=:shaded)
      else
        render_view(name)
      end
    catch err
      @error "scene failed" id=s.id exception=(err, catch_backtrace())
    end
  end; end; end; end; end; end; end; end; end
end

function main()
  only_svg     = "--svg"     in ARGS
  only_blender = "--blender" in ARGS
  positional = filter(a -> !startswith(a, "--"), ARGS)
  pattern = isempty(positional) ? "" : first(positional)

  selected = isempty(pattern) ? SCENES :
             filter(s -> occursin(pattern, string(s.id)) ||
                         occursin(pattern, s.section), SCENES)
  if only_svg
    selected = filter(s -> s.backend == :svg, selected)
  elseif only_blender
    selected = filter(s -> s.backend == :blender, selected)
  end
  if isempty(selected)
    println("No scenes match.")
    return
  end
  println("Matched ", length(selected), " scene(s) of ", length(SCENES), " total.")
  svg     = filter(s -> s.backend == :svg,     selected)
  blender = filter(s -> s.backend == :blender, selected)
  render_svg_scenes(svg)
  render_blender_scenes(blender)
  println("Done in ", round(time() - T0; digits=1), "s.")
end

const T0 = time()
main()
