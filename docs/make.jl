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
        "Concepts" => [
            "Levels & Families" => "concepts/levels_and_families.md",
        ],
        "BIM Elements" => [
            "Horizontal Elements" => "bim/horizontal_elements.md",
            "Vertical Elements" => "bim/vertical_elements.md",
            "Structural Elements" => "bim/structural_elements.md",
            "Circulation" => "bim/circulation.md",
            "Furnishings & Lights" => "bim/furnishings_and_lights.md",
        ],
        "Tutorials" => [
            "Building a Complete Building" => "tutorials/building_tutorial.md",
        ],
        "Reference" => [
            "Backend Operations Matrix" => "reference/backend_operations.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/aptmcl/KhepriBase.jl",
)
