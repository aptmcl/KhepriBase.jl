using KhepriBase
using Documenter

makedocs(;
    modules=[KhepriBase],
    authors="António Menezes Leitão <antonio.menezes.leitao@gmail.com>",
    repo="https://github.com/aptmcl/KhepriBase.jl/blob/{commit}{path}#L{line}",
    sitename="KhepriBase.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://aptmcl.github.io/KhepriBase.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => [
            "Installation" => "getting_started/installation.md",
            "Coordinates" => "getting_started/coordinates.md",
            "Paths" => "getting_started/paths.md",
        ],
        "Concepts" => [
            "Levels & Families" => "concepts/levels_and_families.md",
            "Shapes" => "concepts/shapes.md",
            "Parameters" => "concepts/parameters.md",
            "Backends" => "concepts/backends.md",
        ],
        "BIM Elements" => [
            "Horizontal Elements" => "bim/horizontal_elements.md",
            "Vertical Elements" => "bim/vertical_elements.md",
            "Structural Elements" => "bim/structural_elements.md",
            "Circulation" => "bim/circulation.md",
            "Furnishings & Lights" => "bim/furnishings_and_lights.md",
            "Spaces" => "bim/spaces.md",
        ],
        "Tutorials" => [
            "Building a Complete Building" => "tutorials/building_tutorial.md",
            "Space-First Layout Design" => "tutorials/spaces_tutorial.md",
            "Rendering & Animation" => "tutorials/rendering_tutorial.md",
            "Algorithmic Design" => "tutorials/algorithmic_tutorial.md",
        ],
        "Reference" => [
            "Shapes & Geometry" => "reference/shapes_geometry.md",
            "Utilities" => "reference/utilities.md",
            "Camera & Rendering" => "reference/camera_rendering.md",
            "Backend Operations Matrix" => "reference/backend_operations.md",
            "Realize & Ref Protocol" => "reference/realize_and_ref.md",
            "Implementing a Backend" => "reference/implementing_backend.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/aptmcl/KhepriBase.jl",
    devbranch="master",
)
