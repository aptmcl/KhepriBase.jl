# Doc image rendering

All figures embedded in the KhepriBase docs live under
`docs/src/assets/` and are produced by `render_doc_images.jl`.  The
script is kept out of the Documenter build on purpose: it needs a
running Blender on the host and would slow every doc build to a
crawl.  Regenerate images on demand, commit the PNG/SVG artefacts.

## Requirements

- `julia.exe` on the Windows side (WSL cannot reach Windows localhost
  ports reliably — run the renderer via `julia.exe`).
- Blender installed at `C:\Program Files\Blender Foundation\…` —
  `start_blender()` auto-spawns a headless instance.
- The `KhepriBlender` and `KhepriSVG` packages, both `dev`-ed to the
  sibling checkouts (see `Project.toml`).

## Usage

```bash
cd docs/scripts

# Render everything
julia.exe --project=. render_doc_images.jl

# Render one section / one scene via substring pattern
julia.exe --project=. render_doc_images.jl concepts
julia.exe --project=. render_doc_images.jl bim_wallgraph
julia.exe --project=. render_doc_images.jl tutorials_rendering

# Backend filter (useful to skip Blender when only touching SVG scenes)
julia.exe --project=. render_doc_images.jl --svg
julia.exe --project=. render_doc_images.jl --svg concepts
julia.exe --project=. render_doc_images.jl --blender bim
```

## Layout

- `render_doc_images.jl` — driver.
- `presets.jl` — view / camera presets and the `Scene` registry.
- `scenes/concepts.jl` — schematic 2D diagrams for `docs/src/concepts/`.
- `scenes/bim.jl` — 3D element catalogue for `docs/src/bim/`.
- `scenes/tutorials.jl` — 3D progression shots for tutorials.
- `scenes/reference.jl` — worked examples for the reference pages.

Adding a figure is one `register_scene(...)` call.  The driver then
renders it in the next run and writes to
`docs/src/assets/<section>/<filename>`.

## SVG vs Blender

- **SVG (KhepriSVG)** — schematic 2D.  Composition operators,
  subdivision patterns, adjacency diagrams, leaf-type illustrations,
  constraint violations.  Crisp at any zoom, small files, no external
  tool required.
- **Blender (KhepriBlender)** — 3D renders.  BIM elements, tutorials,
  building hero shots.  PNG output, photoreal via Cycles / Eevee.

Each `Scene` declares `backend = :svg` or `:blender`.  The driver
runs SVG scenes first (they're file-output and instantaneous), then
brings up Blender once for the batch.

## Notes

- The renderer pins each scene's `current_backends` via `with(...)`
  so cross-backend interference is impossible.
- Blender's connection uses a reverse socket: Julia is the server on
  port 12345; `start_blender()` spawns Blender which connects back.
  If port 12345 is held by a stale Julia (e.g., after a SIGKILL),
  the next run fails with `EADDRINUSE` — identify and terminate the
  zombie with `netstat.exe -ano | grep ":12345"` then `taskkill.exe
  /F /PID …`.
- Image assets are committed artefacts.  Documenter does not
  regenerate them, CI does not need Blender, and regular contributors
  only need to run this script when changing a figure.
