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

# defines marginal normal distributions
include("normal/normal.jl")
export AbstractNormal,
    Normal, dim, mean, cov, var, std, residual, logpdf, entropy, kldivergence, Dirac

# plotting marginal normal distributions
include("normal/normal_plotting.jl")

# defines conditional mean for normal kernels
include("normal/conditionalmean.jl")
export AbstractConditionalMean,
    ConditionalMean, AbstractAffineMap, AffineMap, nin, nout, slope, intercept, compose

# defines normal kernels (conditional normal distributions)
include("normal/normalkernel.jl")
export AbstractNormalKernel,
    NormalKernel, condition, compose, marginalise, invert, DiracKernel

# general sampling functions for kernels and Markov processes
include("sampling.jl")

# helper functions
include("utilities.jl")

end
