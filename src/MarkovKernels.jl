module MarkovKernels

using LinearAlgebra, ArrayInterfaceCore, Statistics, Random, RecipesBase

import Base: *, +, eltype, length, size, log, ==, similar, convert, show

import LinearAlgebra: logdet, norm_sqr, HermOrSym
import Statistics: mean, cov, var, std
import Random: rand, GLOBAL_RNG

include("affinemap.jl") # define affine maps to use as conditional means
export AbstractAffineMap, AffineMap, LinearMap, AffineCorrector, slope, intercept, compose

include("covariance_parameter.jl")
export CovarianceParameter, lsqrt, stein, schur_reduce

include("general.jl")
export AbstractDistribution, AbstractMarkovKernel, AbstractLogLike

include("distributions/normal.jl")  # normal distributions
include("distributions/normal_plotting.jl") # plotting vectors of normal distributions
include("distributions/dirac.jl") # dirac distributions
include("distributions/mixture.jl")
export AbstractNormal,
    Normal,
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
    Dirac,
    AbstractMixture,
    Mixture,
    weights,
    components

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

include("binary_operations/compose.jl")
include("binary_operations/marginalise.jl")
include("binary_operations/invert.jl")
export compose, marginalise, invert

include("matrix_utils.jl") # helper functions

end
