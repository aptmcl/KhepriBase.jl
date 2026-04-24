#=
Smoke test v2: verify both backends render one tiny scene.

Writes output directly to a log file to avoid pipe-buffering issues.
=#

log_path = joinpath(@__DIR__, "_smoke.log")
io = open(log_path, "w")
atexit(() -> close(io))
logp(args...) = (println(io, args...); flush(io); println(args...))

using Dates
outdir = abspath(joinpath(@__DIR__, "..", "src", "assets"))
mkpath(outdir)

logp("[", Dates.now(), "] Starting smoke test")
logp("==== SVG smoke test ====")

using KhepriSVG

with(render_dir, outdir) do
with(render_user_dir, ".") do
with(render_backend_dir, ".") do
with(render_kind_dir, ".") do
with(render_color_dir, ".") do
with(render_width, 400) do
with(render_height, 300) do
  delete_all_shapes()
  surface_rectangle(xy(0, 0), 2, 1)
  surface_rectangle(xy(2.5, 0), 2, 1)
  surface_circle(xy(1, 2), 0.5)
  result = render_view("_smoke_svg")
  path = result isa AbstractString ? result : result.path
  logp("SVG render returned: ", path, " isfile=", isfile(path))
end; end; end; end; end; end; end

logp()
logp("==== Blender smoke test ====")

using KhepriBlender

try
  # Confirm server is running on 12345 before we spawn Blender.
  logp("Starting Blender…")
  start_blender()
  deadline = time() + 120.0
  connected = false
  last_bs = nothing
  while time() < deadline
    bs = KhepriBase.current_backends()
    if bs !== last_bs && !isempty(bs)
      logp("current_backends() now has ", length(bs), " entry(ies): ",
           [typeof(b) for b in bs])
      last_bs = bs
    end
    # Accept any socket backend (name may differ)
    if any(b -> b isa KhepriBase.SocketBackend, bs)
      connected = true
      break
    end
    sleep(0.5)
  end
  logp("Connected: ", connected)
  if connected
    logp("top_backend is: ", KhepriBase.top_backend())
    delete_all_shapes()
    sphere(xyz(0, 0, 0), 1)
    box(xyz(2, 0, 0), 1, 1, 1)
    set_view(xyz(5, -5, 4), xyz(1, 0, 0.5))
    with(render_dir, outdir) do
    with(render_user_dir, ".") do
    with(render_backend_dir, ".") do
    with(render_kind_dir, ".") do
    with(render_color_dir, ".") do
    with(render_width, 600) do
    with(render_height, 450) do
      logp("Calling render_view…")
      result = render_view("_smoke_blender")
      logp("Blender render returned: ", result)
    end; end; end; end; end; end; end
  end
catch err
  logp("!! Blender smoke test threw: ", sprint(showerror, err))
end

logp("[", Dates.now(), "] Done.")
