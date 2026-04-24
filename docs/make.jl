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
        size_threshold=300 * 1024,  # 300 KiB — api.md is large
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => [
            "Installation" => "getting_started/installation.md",
            "Coordinates" => "getting_started/coordinates.md",
            "Paths" => "getting_started/paths.md",
        ],
        "Concepts" => [
            "Levels of Abstraction" => "concepts/levels_of_abstraction.md",
            "Levels & Families" => "concepts/levels_and_families.md",
            "Shapes" => "concepts/shapes.md",
            "Parameters" => "concepts/parameters.md",
            "Backends" => "concepts/backends.md",
            "Constraints" => "concepts/constraints.md",
            "Designs (Level 2)" => "concepts/designs.md",
            "Space Descriptions" => "concepts/space-descriptions.md",
            "Composition Operators" => "concepts/composition-operators.md",
            "Top-Down Subdivision" => "concepts/subdivision.md",
        ],
        "BIM Elements" => [
            "Horizontal Elements" => "bim/horizontal_elements.md",
            "Vertical Elements" => "bim/vertical_elements.md",
            "Structural Elements" => "bim/structural_elements.md",
            "Circulation" => "bim/circulation.md",
            "Furnishings & Lights" => "bim/furnishings_and_lights.md",
            "Spaces" => "bim/spaces.md",
            "Wall Graph" => "bim/wall_graph.md",
        ],
        "Tutorials" => [
            "Building a Complete Building" => "tutorials/building_tutorial.md",
            "Space-First Layout Design" => "tutorials/spaces_tutorial.md",
            "Wall Graph Networks" => "tutorials/wall_graph_tutorial.md",
            "Rendering & Animation" => "tutorials/rendering_tutorial.md",
            "Algorithmic Design" => "tutorials/algorithmic_tutorial.md",
            "Isenberg (Bottom-Up)" => "tutorials/isemberg_bottom_up.md",
            "Isenberg (Top-Down)" => "tutorials/isemberg_top_down.md",
        ],
        "Reference" => [
            "Shapes & Geometry" => "reference/shapes_geometry.md",
            "Utilities" => "reference/utilities.md",
            "Camera & Rendering" => "reference/camera_rendering.md",
            "Backend Operations Matrix" => "reference/backend_operations.md",
            "Realize & Ref Protocol" => "reference/realize_and_ref.md",
            "Implementing a Backend" => "reference/implementing_backend.md",
            "Layout Engine" => "reference/layout-engine.md",
            "Adjacencies" => "reference/adjacencies.md",
            "Constraints" => "reference/constraints.md",
            "Design Types" => "reference/design-types.md",
            "Leaf Constructors" => "reference/design-leaves.md",
            "Combinators" => "reference/design-combinators.md",
            "Subdivision" => "reference/design-subdivision.md",
            "Annotations" => "reference/design-annotations.md",
            "Tree Queries" => "reference/design-queries.md",
            "API -- Shapes" => "reference/api.md",
            "API -- BIM" => "reference/api_bim.md",
            "API -- Infrastructure" => "reference/api_other.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/aptmcl/KhepriBase.jl",
    devbranch="master",
)
