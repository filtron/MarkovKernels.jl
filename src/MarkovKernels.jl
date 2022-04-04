module MarkovKernels

using LinearAlgebra, Statistics, Random

import Base: *, +, eltype, length, size, log

import LinearAlgebra: logdet, norm_sqr
import Statistics: mean, cov, var, std
import Random: rand, GLOBAL_RNG

abstract type AbstractDistribution end

abstract type AbstractMarkovKernel end

abstract type AbstractLikelihood end

export AbstractDistribution, AbstractMarkovKernel, AbstractLikelihood

include("normal.jl")

export AbstractNormalParametrisation, Usual, Information
export Normal, parametrisation
export dim, mean, cov, var, std, residual, logpdf, entropy, kldivergence


include("utilities.jl")


end
