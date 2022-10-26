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
        "Affine maps" => "affinemap.md",
        "Covariance parametrisations" => "covariance_parameter.md",
        "Distributions" => ["Normal" => "normal.md", "Dirac" => "dirac.md"],
        "Kernels" => ["normalkernel.md", "dirackernel.md"],
        "LogLikelihoods" => "likelihoods.md",
        "Binary operators" => "binary_operators.md",
        "Tutorials" => [
            "Sampling from probabilistic state-space models" => "tutorial_pomp_sampling.md"
            "Implementing a Kalman filter" => "tutorial_kalman_filter.md"
            "Implementing a Rauch-Tung-Striebel smoother" => "tutorial_rts.md"
        ],
    ],
)

deploydocs(;
    repo = "github.com/filtron/MarkovKernels.jl",
    devbranch = "main",
    push_preview = true,
)
