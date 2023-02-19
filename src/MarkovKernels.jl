module MarkovKernels

using LinearAlgebra, ArrayInterface, Statistics, Random, RecipesBase

import Base: *, +, eltype, length, size, log, ==, similar, convert, show, log, copy

import LinearAlgebra: logdet, norm_sqr, HermOrSym
import Statistics: mean, cov, var, std
import Random: rand, rand!, GLOBAL_RNG

include("affinemap.jl") # define affine maps to use as conditional means
export AbstractAffineMap, AffineMap, LinearMap, AffineCorrector, slope, intercept, compose

include("covariance_parameter.jl")
export CovarianceParameter, lsqrt, stein, schur_reduce

include("general.jl")
export AbstractDistribution, AbstractMarkovKernel, AbstractLikelihood

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
    AffineDiracKernel

include("likelihoods.jl") # defines observation likelihoods
export Likelihood, measurement_model, measurement

include("binary_operations/compose.jl")
include("binary_operations/marginalize.jl")
include("binary_operations/invert.jl")
include("binary_operations/bayes_rule.jl")
export compose, marginalize, invert, bayes_rule, bayes_rule!

include("matrix_utils.jl") # helper functions

end
