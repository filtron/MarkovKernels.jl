using MarkovKernels
using Documenter

DocMeta.setdocmeta!(MarkovKernels, :DocTestSetup, :(using MarkovKernels); recursive = true)

makedocs(;
    modules = [MarkovKernels],
    authors = "Filip Tronarp <filip.tronarp@matstat.lu.se> and contributors",
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
        "Affine maps" => "affinemaps/affinemaps.md",
        "PSDParametrizations" => [
            "General" => "PSDParametrizations/general.md",
            "Self-adjoint" => "PSDParametrizations/selfadjoint.md",
            "Cholesky" => "PSDParametrizations/cholesky.md",
            "Scalar" => "PSDParametrizations/scalar.md",
            "Internal" => "PSDParametrizations/internal.md",
        ],
        "Distributions" => [
            "General" => "distributions/general.md",
            "Categorical" => "distributions/categorical.md",
            "Normal" => "distributions/normal.md",
            "Dirac" => "distributions/dirac.md",
        ],
        "Kernels" => ["kernels/normalkernel.md", "kernels/dirackernel.md"],
        "Likelihoods" => "likelihoods.md",
        "Binary operators" => "binary_operators.md",
        "Tutorials" => [
            "Sampling from probabilistic state-space models" => "tutorial_pomp_sampling.md"
            "Kalman filtering and smoothing" => "tutorial_kalman_filter.md"
        ],
    ],
)

deploydocs(;
    repo = "github.com/filtron/MarkovKernels.jl",
    devbranch = "main",
    push_preview = true,
)
