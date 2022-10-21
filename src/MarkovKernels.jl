module MarkovKernels

using LinearAlgebra, Statistics, Random, RecipesBase

import Base: *, +, eltype, length, size, log, ==, similar, convert

import LinearAlgebra: logdet, norm_sqr
import Statistics: mean, cov, var, std
import Random: rand, GLOBAL_RNG

abstract type AbstractDistribution{T<:Number} end

eltype(::AbstractDistribution{T}) where {T} = T

abstract type AbstractMarkovKernel{T<:Number} end

eltype(::AbstractMarkovKernel{T}) where {T} = T

export AbstractDistribution, AbstractMarkovKernel

include("distributions/normal.jl")  # normal distributions
include("distributions/normal_plotting.jl") # plotting vectors of normal distributions
include("distributions/dirac.jl") # dirac distributions
export AbstractNormal,
    Normal,
    IsoNormal,
    dim,
    mean,
    cov,
    covp,
    var,
    std,
    residual,
    logpdf,
    entropy,
    kldivergence,
    AbstractDirac,
    Dirac

# defines affine conditional mean for normal kernels
include("kernels/affinemap.jl")
export AbstractAffineMap,
    AffineMap, LinearMap, AffineCorrector, slope, intercept, compose, nout

include("kernels/normalkernel.jl") # defines normal kernels
include("kernels/dirackernel.jl") # defines dirac kernels
export AbstractNormalKernel,
    NormalKernel,
    AffineNormalKernel,
    condition,
    AbstractDiracKernel,
    DiracKernel,
    AffineDiracKernel

include("kernels/compose.jl")
export compose

include("marginalise.jl")
export marginalise

include("invert.jl")
export invert

# defines observation likelihoods
include("likelihoods.jl")
export AbstractLogLike, LogLike, measurement_model, measurement, bayes_rule

# general sampling functions for kernels and Markov processes
include("sampling.jl")

# helper functions
include("matrix_utilities.jl")
end
