module MarkovKernels

using LinearAlgebra, Statistics, Random, RecipesBase

import Base: *, +, eltype, length, size, log, ==, similar, filter, IteratorSize, HasLength

import LinearAlgebra: logdet, norm_sqr
import Statistics: mean, cov, var, std
import Random: rand, GLOBAL_RNG

abstract type AbstractDistribution end

abstract type AbstractMarkovKernel end

export AbstractDistribution, AbstractMarkovKernel

# defines observation likelihoods
include("likelihoods.jl")
export AbstractLikelihood, Likelihood, measurement_model, measurement

include("distributions/normal.jl")  # normal distributions
include("distributions/normal_generic.jl") # generic normal distributions
include("distributions/normal_plotting.jl") # plotting vectors of normal distributions
include("distributions/dirac.jl") # dirac distributions
export AbstractNormal,
    Normal,
    IsoNormal,
    dim,
    mean,
    cov,
    var,
    std,
    residual,
    logpdf,
    entropy,
    kldivergence,
    AbstractDirac,
    Dirac

# defines conditional mean for normal kernels
include("kernels/affinemap.jl")
export AbstractAffineMap, AffineMap, nin, nout, slope, intercept, compose

include("kernels/normalkernel.jl") # defines normal kernels
include("kernels/normalkernel_generic.jl") # generic normal kernels
include("kernels/dirackernel.jl") # defines dirac kernels
include("kernels/compose.jl")
export AbstractNormalKernel,
    NormalKernel, AffineNormalKernel, AffineIsoNormalKernel, condition, compose, marginalise, invert, DiracKernel

# general sampling functions for kernels and Markov processes
include("sampling.jl")

# helper functions
include("matrix_utilities.jl")

end
