using MarkovKernels
using Documenter

DocMeta.setdocmeta!(MarkovKernels, :DocTestSetup, :(using MarkovKernels); recursive = true)

makedocs(;
    modules = [MarkovKernels],
    authors = "Filip Tronarp <filip.tronarp@uni-tuebingen.de> and contributors",
    repo = "https://github.com/filtron/MarkovKernels.jl/blob/{commit}{path}#{line}",
    sitename = "MarkovKernels.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://filtron.github.io/MarkovKernels.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Distributions" => [
            "Normal distributions" => "normal.md"
        ],
        "Tutorials" => [
            "Sampling from Probabilistic state-space models" => "tutorial_pomp_sampling.md"
            "Implementing a Kalman filter" => "tutorial_kalman_filter.md"
        ],
        ],
)

deploydocs(;
    repo = "github.com/filtron/MarkovKernels.jl",
    devbranch = "main",
    push_preview = true,
)
