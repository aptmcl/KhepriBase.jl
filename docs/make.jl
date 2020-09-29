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
    ],
)

deploydocs(;
    repo="github.com/aptmcl/KhepriBase.jl",
)
