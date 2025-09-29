module MarkovKernels

using LinearAlgebra, Statistics, Random, AliasTables, RecipesBase

import Base:
    *,
    +,
    -,
    ∘,
    \,
    zero,
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
    norm_sqr,
    ldiv!,
    mul!

import Statistics: mean, cov, var, std
import Random: rand, rand!, GLOBAL_RNG

include("utils.jl") # helper functions
export AbstractNumOrVec

include("affinemaps/affinemaps.jl")
export AbstractAffineMap, AffineMap, LinearMap, AffineCorrector, slope, intercept, compose

include("PSDParametrizations/PSDParametrizations.jl")
export PSDTrait,
    IsPSD,
    IsNotPSD,
    psdcheck,
    convert_psd_eltype,
    SelfAdjoint,
    selfadjoint,
    rsqrt,
    lsqrt,
    stein,
    schur_reduce

include("distributions/distribution_generic.jl")
export AbstractDistribution, sample_type, sample_eltype

include("kernels/kernel_generic.jl")
export AbstractMarkovKernel, condition

include("likelihoods/likelihood_generic.jl")
export AbstractLikelihood, log

include("distributions/probability_vector.jl")
include("distributions/dirac.jl")
include("distributions/normal.jl")
include("distributions/plotting.jl")
export AbstractProbabilityVector,
    ProbabilityVector,
    probability_vector,
    AbstractNormal,
    Normal,
    dim,
    mean,
    cov,
    covparam,
    var,
    std,
    residual,
    logpdf,
    entropy,
    kldivergence,
    AbstractDirac,
    Dirac

include("kernels/normalkernel.jl")
include("kernels/dirackernel.jl")
include("kernels/stochasticmatrix.jl")
export Skedasticity,
    Homoskedastic,
    Heteroskedastic,
    skedasticity,
    AbstractNormalKernel,
    NormalKernel,
    HomoskedasticNormalKernel,
    AffineHomoskedasticNormalKernel,
    AffineHeteroskedasticNormalKernel,
    NonlinearNormalKernel,
    condition,
    AbstractDiracKernel,
    DiracKernel,
    AffineDiracKernel,
    IdentityKernel,
    AbstractStochasticMatrix,
    StochasticMatrix,
    probability_matrix

include("likelihoods/likelihood.jl")
include("likelihoods/categorical_likelihood.jl")
include("likelihoods/flatlikelihood.jl")
include("likelihoods/logquadratic.jl")
export Likelihood,
    measurement_model,
    measurement,
    CategoricalLikelihood,
    likelihood_vector,
    FlatLikelihood,
    LogQuadraticLikelihood

include("binary_operations/compose.jl")
include("binary_operations/forward_operator.jl")
include("binary_operations/invert.jl")
include("binary_operations/posterior.jl")
include("binary_operations/algebra.jl")
include("binary_operations/htransform.jl")
export compose,
    forward_operator, invert, posterior_and_loglike, posterior, htransform_and_likelihood

include("deprecated.jl")

end
