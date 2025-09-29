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
            "Internal" => "PSDParametrizations/internal.md",
        ],
        "Distributions" => [
            "General" => "distributions/distribution_general.md",
            "Probability vector" => "distributions/probability_vector.md",
            "Normal" => "distributions/normal.md",
            "Dirac" => "distributions/dirac.md",
        ],
        "Kernels" => [
            "kernels/kernel_general.md",
            "kernels/normalkernel.md",
            "kernels/dirackernel.md",
            "kernels/stochasticmatrix.md",
        ],
        "Likelihoods" => [
            "likelihoods/likelihood_general.md",
            "likelihoods/likelihood_vector.md",
            "likelihoods/likelihood.md",
            "likelihoods/logquadratic.md",
            "likelihoods/flatlikelihood.md",
        ],
        "Binary operators" => "binary_operators.md",
        "Tutorials" => [
            "Sampling and inference in Hidden Markov models" => "tutorials/hidden_markov_model.md",
            "Sampling and inference in Gauss-Markov models" => "tutorials/gauss_markov_regression.md",
        ],
    ],
)

deploydocs(;
    repo = "github.com/filtron/MarkovKernels.jl",
    devbranch = "main",
    push_preview = true,
)
