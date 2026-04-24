#=
Camera / render presets + the `Scene` record.  Every image the doc
site needs is one `Scene` pushed into `SCENES`.  The driver
iterates the registry once per backend.

View helpers take the *content's* bounding information so the
camera actually frames the geometry.  Earlier presets fixed the
camera at the world origin, which left parts of the scene outside
the view for any geometry whose corner — not centre — was at the
origin.  `iso_view(cx, cy, cz, r)` places the camera on a standard
isometric ray away from (cx, cy, cz) and targets that same point,
guaranteeing the content lands in frame regardless of where its
corner is anchored.
=#

struct Scene
  id::Symbol
  section::String    # "concepts" | "bim" | "tutorials" | "reference"
  filename::String   # "concepts-composition-beside_x.svg"
  backend::Symbol    # :svg or :blender
  width::Int
  height::Int
  view               # nothing | (eye, target)
  build::Function
end

const SCENES = Scene[]

register_scene(; id, section, filename, backend, width=800, height=600,
               view=nothing, build) =
  push!(SCENES, Scene(Symbol(id), section, filename, backend,
                      width, height, view, build))

#=
Standard isometric view for a scene centred at (cx, cy, cz) with a
characteristic radius `r` (the longest plan half-diagonal plus a
bit of height clearance).  The camera is on a 45°/45°/30° ray — far
enough out that the content sits comfortably inside the frame with
room for the sky/ground above and below.
=#
iso_view(cx, cy, cz, r) = (
  eye    = xyz(cx + 1.3 * r, cy - 1.3 * r, cz + 0.9 * r),
  target = xyz(cx, cy, cz))

#=
Top-down plan view, useful for floorplan-style images.  The `z`
distance is generous so the camera stays above any roof element.
=#
top_view(cx, cy, r) = (
  eye    = xyz(cx, cy, 2 * r + 8),
  target = xyz(cx, cy, 0))

#=
Dead-on front elevation along −y, looking toward +y.  `r` sets
both camera distance and framing height.
=#
front_view(cx, cy, cz, r) = (
  eye    = xyz(cx, cy - 1.7 * r, cz),
  target = xyz(cx, cy, cz))

# ---- Back-compat aliases (a handful of early scenes reference these) ----
const VIEW_ISO_SMALL  = iso_view(3.0, 1.5, 1.5,  7.5)
const VIEW_ISO_MEDIUM = iso_view(5.0, 4.0, 1.5, 10.0)
const VIEW_ISO_LARGE  = iso_view(8.0, 5.0, 2.5, 16.0)
const VIEW_TOP_SMALL  = top_view(3.0, 1.5,  7.5)
const VIEW_TOP_MEDIUM = top_view(5.0, 4.0, 10.0)
