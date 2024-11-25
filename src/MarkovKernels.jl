module MarkovKernels

using LinearAlgebra, ArrayInterface, Statistics, Random, AliasTables, RecipesBase

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
    CovarianceParameter,
    SelfAdjoint,
    selfadjoint,
    rsqrt,
    lsqrt,
    stein,
    schur_reduce

include("generic.jl")
export AbstractDistribution,
    AbstractMarkovKernel, AbstractLikelihood, sample_type, sample_eltype

include("distributions/categorical.jl")
include("distributions/dirac.jl")
include("distributions/normal.jl")
include("distributions/plotting.jl")
export AbstractCategorical,
    Categorical,
    probability_vector,
    AbstractNormal,
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
export Likelihood,
    measurement_model, measurement, CategoricalLikelihood, likelihood_vector, FlatLikelihood

include("binary_operations/compose.jl")
include("binary_operations/marginalize.jl")
include("binary_operations/invert.jl")
include("binary_operations/posterior.jl")
include("binary_operations/algebra.jl")
include("binary_operations/htransform.jl")
export compose, marginalize, invert, posterior_and_loglike, posterior, htransform

end
