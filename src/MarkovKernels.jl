module MarkovKernels

using LinearAlgebra, ArrayInterface, Statistics, Random, RecipesBase

import Base:
    *,
    +,
    -,
    ∘,
    eltype,
    length,
    size,
    log,
    ==,
    similar,
    convert,
    show,
    log,
    copy,
    copy!,
    vcat

import LinearAlgebra:
    Factorization,
    AbstractMatrix,
    AbstractArray,
    Matrix,
    Array,
    HermOrSym,
    Cholesky,
    det,
    logdet,
    inv,
    pinv,
    diag,
    tr,
    norm_sqr

import Statistics: mean, cov, var, std
import Random: rand, rand!, GLOBAL_RNG

include("affinemap.jl") # define affine maps to use as conditional means
export AbstractAffineMap, AffineMap, LinearMap, AffineCorrector, slope, intercept, compose

include("PSDFactorizations/PSDFactorizations.jl")

include("covariance_parameter.jl")
export CovarianceParameter, lsqrt, stein, schur_reduce

include("general.jl")
export AbstractDistribution,
    AbstractMarkovKernel, AbstractLikelihood, typeof_sample, eltype_sample

include("distributions/normal.jl")
include("distributions/dirac.jl")
include("distributions/particle_system.jl")
include("distributions/plotting.jl")
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
    AbstractParticleSystem,
    ParticleSystem,
    logweights,
    weights,
    particles,
    nparticles

include("kernels/normalkernel.jl") # defines normal kernels
include("kernels/dirackernel.jl") # defines dirac kernels
export AbstractNormalKernel,
    NormalKernel,
    AffineNormalKernel,
    condition,
    AbstractDiracKernel,
    DiracKernel,
    AffineDiracKernel,
    IdentityKernel

include("likelihoods.jl") # defines observation likelihoods
export FlatLikelihood, Likelihood, measurement_model, measurement

include("binary_operations/compose.jl")
include("binary_operations/marginalize.jl")
include("binary_operations/invert.jl")
include("binary_operations/posterior.jl")
include("binary_operations/algebra.jl")
export compose,
    marginalize, invert, posterior_and_loglike, posterior, posterior_and_loglike!

# these will be removed
export bayes_rule_and_loglike, bayes_rule_and_loglike!, bayes_rule

include("matrix_utils.jl") # helper functions

end
