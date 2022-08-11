var documenterSearchIndex = {"docs":
[{"location":"normal/#Normal","page":"Normal distributions","title":"Normal","text":"","category":"section"},{"location":"normal/","page":"Normal distributions","title":"Normal distributions","text":"CurrentModule = MarkovKernels","category":"page"},{"location":"normal/","page":"Normal distributions","title":"Normal distributions","text":"The normal distribution is denoted by","category":"page"},{"location":"normal/","page":"Normal distributions","title":"Normal distributions","text":"pi(x) = mathcalN(x  mu   Sigma )","category":"page"},{"location":"normal/","page":"Normal distributions","title":"Normal distributions","text":"The exact expression for the probabiltiy density function depends on whether x is vector with real or complex values, both are supported.","category":"page"},{"location":"normal/","page":"Normal distributions","title":"Normal distributions","text":"Types:","category":"page"},{"location":"normal/","page":"Normal distributions","title":"Normal distributions","text":"abstract type AbstractNormal{T<:Number}  <: AbstractDistribution end # normal distributions with realisations in real / complex Euclidean spaces\nNormal{T} <: AbstractNormal{T} # mean vector / covariance matrix parametrisation of normal distributions\nDirac{T}  <: AbstractNormal{T} # normal distribution with zero covariance","category":"page"},{"location":"normal/","page":"Normal distributions","title":"Normal distributions","text":"Functionality:","category":"page"},{"location":"tutorial_kalman_filter/#Implementing-a-Kalman-filter","page":"Implementing a Kalman filter","title":"Implementing a Kalman filter","text":"","category":"section"},{"location":"tutorial_kalman_filter/","page":"Implementing a Kalman filter","title":"Implementing a Kalman filter","text":"This tutorial describes how to implement a Kalman filter for the following state-space model","category":"page"},{"location":"tutorial_kalman_filter/","page":"Implementing a Kalman filter","title":"Implementing a Kalman filter","text":"beginaligned\nx_0 sim mathcalN(mu_0 Sigma_0) \nx_n mid x_n-1 sim mathcalN(Phi  x_n-1 Q)\nz_n mid x_n sim mathcalN(Cx_nR)\nendaligned","category":"page"},{"location":"tutorial_kalman_filter/","page":"Implementing a Kalman filter","title":"Implementing a Kalman filter","text":"given a measurement sequence z_0N.","category":"page"},{"location":"tutorial_kalman_filter/#Kalman-filter-implementation","page":"Implementing a Kalman filter","title":"Kalman filter implementation","text":"","category":"section"},{"location":"tutorial_pomp_sampling/#Sampling-from-Markov-realisable-processes","page":"Sampling from Probabilistic state-space models","title":"Sampling from Markov-realisable processes","text":"","category":"section"},{"location":"tutorial_pomp_sampling/","page":"Sampling from Probabilistic state-space models","title":"Sampling from Probabilistic state-space models","text":"This tutorial describes how to sample from the probabilistic state-space model given by","category":"page"},{"location":"tutorial_pomp_sampling/","page":"Sampling from Probabilistic state-space models","title":"Sampling from Probabilistic state-space models","text":"beginaligned\nx_0 sim mathcalN(mu_0 Sigma_0) \nx_n mid x_n-1 sim mathcalN(Phi  x_n-1 Q)\ny_n = C x_n\nendaligned","category":"page"},{"location":"tutorial_pomp_sampling/","page":"Sampling from Probabilistic state-space models","title":"Sampling from Probabilistic state-space models","text":"where x and y are referred to as the latent Gauss-Markov process and the output process, respectively. Additionally, noisy measurements of the output process will be generated according to","category":"page"},{"location":"tutorial_pomp_sampling/","page":"Sampling from Probabilistic state-space models","title":"Sampling from Probabilistic state-space models","text":"z_n mid x_n sim mathcalN(Cx_nR)","category":"page"},{"location":"tutorial_pomp_sampling/#Sampling-from-the-latent-Gauss-Markov-process","page":"Sampling from Probabilistic state-space models","title":"Sampling from the latent Gauss-Markov process","text":"","category":"section"},{"location":"tutorial_pomp_sampling/","page":"Sampling from Probabilistic state-space models","title":"Sampling from Probabilistic state-space models","text":"using MarkovKernels, LinearAlgebra, Plots\n\n\nN = 2^9\nns = 0:N\n\n# define a Markov kernel for a homogeneous Markov proces\nλ = 0.9\nσ = 1.0\ndimx = 2\n\nΦ = [λ 0.0; 1 - λ^2 λ]\nQ = (1-λ^2)*(1+λ^2) * 1.0*I(dimx)\n\nforward_kernel = NormalKernel(Φ, Q)\n\n# define initial distribution\ninit = Normal(zeros(dimx), 1.0*I(dimx))\n\n# sample Gauss-Markov process and plot\nxs = rand(init, forward_kernel, N)\nplot(\n    ns,\n    xs,\n    layout=(dimx,1),\n    xlabel = [\"\" \"t\"],\n    label = [\"x0\" \"x1\"],\n    title = [\"Latent Gauss-Markov process\" \"\"]\n)","category":"page"},{"location":"tutorial_pomp_sampling/#Sampling-output-and-measurements","page":"Sampling from Probabilistic state-space models","title":"Sampling output and measurements","text":"","category":"section"},{"location":"tutorial_pomp_sampling/","page":"Sampling from Probabilistic state-space models","title":"Sampling from Probabilistic state-space models","text":"# define output process\nC = σ*[1.0 -1.0]\noutput_kernel = DiracKernel(C)\n\n# define measurements of output process\nR = fill(0.1,1,1)\nmeasurement_kernel = NormalKernel(C,R)\n\n# sample outputs, measurements and plot\noutput = rand(output_kernel,xs)\nzs = rand(measurement_kernel,xs)\n\nplot(\n    ns,\n    output,\n    xlabel = \"t\",\n    ylabel = \"y\",\n    label = \"output process\",\n    title = \"Output process and measurements\"\n)\n\nscatter!(\n    ns,\n    zs,\n    label = \"measurements\",\n    color=\"black\"\n)","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = MarkovKernels\nusing MarkovKernels","category":"page"},{"location":"#MarkovKernels","page":"Home","title":"MarkovKernels","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for MarkovKernels.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [MarkovKernels]\nPages   = [\"normal.jl\",\"normal_generic.jl\",\"normalkernel.jl\",\"normalkernel_generic.jl\"]","category":"page"},{"location":"#MarkovKernels.AbstractNormal","page":"Home","title":"MarkovKernels.AbstractNormal","text":"AbstractNormal{T<:Number}\n\nAbstract type for representing normal distributed random vectors taking values in T.\n\n\n\n\n\n","category":"type"},{"location":"#MarkovKernels.Normal","page":"Home","title":"MarkovKernels.Normal","text":"Normal{T,U,V}\n\nStandard parametrisation of the normal distribution with element type T.\n\n\n\n\n\n","category":"type"},{"location":"#Base.rand-Tuple{AbstractNormal}","page":"Home","title":"Base.rand","text":"rand(N::AbstractNormal)\n\nDraws one random vector using the random number generator GLOBAL_RNG.\n\n\n\n\n\n","category":"method"},{"location":"#Base.rand-Tuple{Random.AbstractRNG, AbstractNormal}","page":"Home","title":"Base.rand","text":"rand(RNG::AbstractRNG, N::AbstractNormal)\n\nDraws one random vector from N using the random number generator RNG.\n\n\n\n\n\n","category":"method"},{"location":"#MarkovKernels.dim-Tuple{AbstractNormal}","page":"Home","title":"MarkovKernels.dim","text":"dim(N::AbstractNormal)\n\nReturns the dimension of the random vector represented by the distribution N.\n\n\n\n\n\n","category":"method"},{"location":"#MarkovKernels.entropy-Tuple{AbstractNormal}","page":"Home","title":"MarkovKernels.entropy","text":"entropy(N::AbstractNormal)\n\nComputes the entropy of the distribution N.\n\n\n\n\n\n","category":"method"},{"location":"#MarkovKernels.kldivergence-Tuple{AbstractNormal, AbstractNormal}","page":"Home","title":"MarkovKernels.kldivergence","text":"kldivergence(N1::AbstractNormal, N2::AbstractNormal)\n\nComputes the Kullback-Leibler divergence between N1 and N2.\n\n\n\n\n\n","category":"method"},{"location":"#MarkovKernels.logpdf-Tuple{AbstractNormal, Any}","page":"Home","title":"MarkovKernels.logpdf","text":"logpdf(N::AbstractNormal,x)\n\nEvaluates the logarithm  of the probability density of N at x.\n\n\n\n\n\n","category":"method"},{"location":"#MarkovKernels.residual-Tuple{AbstractNormal, Any}","page":"Home","title":"MarkovKernels.residual","text":"residual(N::AbstractNormal,x)\n\nGiven a realisation x, computes the whitened residual with respect to N.\n\n\n\n\n\n","category":"method"},{"location":"#Statistics.cov-Tuple{AbstractNormal}","page":"Home","title":"Statistics.cov","text":"cov(N::AbstractNormal)\n\nReturns the covariance matrix of the distribution N.\n\n\n\n\n\n","category":"method"},{"location":"#Statistics.mean-Tuple{AbstractNormal}","page":"Home","title":"Statistics.mean","text":"mean(N::AbstractNormal)\n\nReturns the mean vector of the distribution N.\n\n\n\n\n\n","category":"method"},{"location":"#Statistics.std-Tuple{AbstractNormal}","page":"Home","title":"Statistics.std","text":"std(N::AbstractNormal)\n\nReturns a vector of marginal standard deviations of the distribution N.\n\n\n\n\n\n","category":"method"},{"location":"#Statistics.var-Tuple{AbstractNormal}","page":"Home","title":"Statistics.var","text":"var(N::AbstractNormal)\n\nReturns a vector of marginal variances of the distribution N.\n\n\n\n\n\n","category":"method"},{"location":"#MarkovKernels.AbstractNormalKernel","page":"Home","title":"MarkovKernels.AbstractNormalKernel","text":"AbstractNormalKernel{T<:Number}\n\nAbstract type for representing Normal kernels taking values in T.\n\n\n\n\n\n","category":"type"},{"location":"#MarkovKernels.NormalKernel","page":"Home","title":"MarkovKernels.NormalKernel","text":"NormalKernel\n\nStandard parametrisation of Normal kernels.\n\n\n\n\n\n","category":"type"},{"location":"#MarkovKernels.NormalKernel-Tuple{AbstractMatrix, AbstractMatrix}","page":"Home","title":"MarkovKernels.NormalKernel","text":"NormalKernel(Φ::AbstractMatrix, Σ::AbstractMatrix)\n\nCreates a Normal kernel with linear conditional mean of slope Φ and covariance matrix Σ.\n\n\n\n\n\n","category":"method"},{"location":"#MarkovKernels.NormalKernel-Tuple{AbstractMatrix, AbstractVector, AbstractMatrix}","page":"Home","title":"MarkovKernels.NormalKernel","text":"NormalKernel(Φ::AbstractMatrix, b::AbstractVector, Σ::AbstractMatrix)\n\nCreates a Normal kernel with affine conditional mean of slope Φ, intercept b, and covariance matrix Σ.\n\n\n\n\n\n","category":"method"},{"location":"#MarkovKernels.condition-Tuple{AbstractNormalKernel, Any}","page":"Home","title":"MarkovKernels.condition","text":"condition(K::AbstractNormalKernel,x)\n\nReturns a Normal distribution corresponding to K evaluated at x.\n\n\n\n\n\n","category":"method"},{"location":"#Statistics.cov-Tuple{AbstractNormalKernel}","page":"Home","title":"Statistics.cov","text":"cov(K::AbstractNormalKernel)\n\nReturns the conditional covariance matrix of the Normal kernel K.\n\n\n\n\n\n","category":"method"},{"location":"#Statistics.mean-Tuple{AbstractNormalKernel}","page":"Home","title":"Statistics.mean","text":"mean(K::AbstractNormalKernel)\n\nReturns the conditional mean function of the Normal kernel K.\n\n\n\n\n\n","category":"method"}]
}
