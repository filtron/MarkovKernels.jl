module MarkovKernels


using LinearAlgebra, Statistics, Random, RecipesBase

import Base: *, +, eltype, length, size, log, ==, similar, convert, show

import LinearAlgebra: logdet, norm_sqr, HermOrSym
import Statistics: mean, cov, var, std
import Random: rand, GLOBAL_RNG

include("affinemap.jl") # define affine maps to use as conditional means
export AbstractAffineMap,
    AffineMap, LinearMap, AffineCorrector, slope, intercept, compose, nout

include("covariance_parameter.jl")
export CovarianceParameter, FactorizationCompatible, lsqrt, stein, schur_reduce

abstract type AbstractDistribution{T<:Number} end

eltype(::AbstractDistribution{T}) where {T} = T

abstract type AbstractMarkovKernel{T<:Number} end

eltype(::AbstractMarkovKernel{T}) where {T} = T

abstract type AbstractLogLike end

export AbstractDistribution, AbstractMarkovKernel, AbstractLogLike

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

include("kernels/normalkernel.jl") # defines normal kernels
include("kernels/dirackernel.jl") # defines dirac kernels
export AbstractNormalKernel,
    NormalKernel,
    AffineNormalKernel,
    condition,
    AbstractDiracKernel,
    DiracKernel,
    AffineDiracKernel

include("likelihoods.jl") # defines observation likelihoods
export LogLike, measurement_model, measurement, bayes_rule

include("kernels/compose.jl")
include("marginalise.jl")
include("invert.jl")
export compose, marginalise, invert

include("matrix_utils.jl") # helper functions

end
