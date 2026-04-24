#=
Camera / render presets + the `Scene` record.  Every image the doc
site needs is one `Scene` pushed into `SCENES`.  The driver
iterates the registry once per backend.
=#

struct Scene
  id::Symbol
  section::String    # "concepts" | "bim" | "tutorials" | "reference"
  filename::String   # "concepts-composition-beside_x.svg"
  backend::Symbol    # :svg or :blender
  width::Int
  height::Int
  view                # nothing | (eye, target)
  build::Function
end

const SCENES = Scene[]

register_scene(; id, section, filename, backend, width=800, height=600,
               view=nothing, build) =
  push!(SCENES, Scene(Symbol(id), section, filename, backend,
                      width, height, view, build))

# ---- Common view presets (3D) ----

const VIEW_ISO_SMALL =
  (eye=xyz(14, -14, 10), target=xyz(0, 0, 1))
const VIEW_ISO_MEDIUM =
  (eye=xyz(22, -22, 14), target=xyz(0, 0, 1.5))
const VIEW_ISO_LARGE =
  (eye=xyz(35, -35, 22), target=xyz(5, 5, 2))
const VIEW_TOP_SMALL =
  (eye=xyz(0, 0, 30), target=xyz(0, 0, 0))
const VIEW_TOP_MEDIUM =
  (eye=xyz(4, 3, 40), target=xyz(4, 3, 0))
