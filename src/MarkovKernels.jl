module MarkovKernels

using LinearAlgebra, Statistics, Random

import Base: *, +, eltype, length, size, log, ==, similar

import LinearAlgebra: logdet, norm_sqr
import Statistics: mean, cov, var, std
import Random: rand, GLOBAL_RNG

abstract type AbstractDistribution end

abstract type AbstractMarkovKernel end

export AbstractDistribution, AbstractMarkovKernel

include("likelihoods.jl")
export AbstractLikelihood, Likelihood, FlatLikelihood

include("normal/normal.jl")
export AbstractNormal, Normal, dim, mean, cov, var, std, residual, logpdf, entropy, kldivergence

include("conditionalmoments/conditionalmean.jl")
export AbstractConditonalMean

include("conditionalmoments/affinemaps.jl")
export AbstractAffineMap, AffineMap, nin, nout, slope, intercept, stein, compose

include("normal/normalkernel.jl")
export NormalKernel, condition, compose, marginalise, invert

include("inference/filter.jl")
export filter

include("utilities.jl")


end
