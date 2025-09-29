abstract type Skedasticity end
struct Homoskedastic <: Skedasticity end
struct Heteroskedastic <: Skedasticity end

# do stuff here
_skedasticity(::IsPSD) = Homoskedastic()
_skedasticity(::IsNotPSD) = Heteroskedastic()

skedasticity(Σ) = _skedasticity(psdcheck(Σ))

"""
    AbstractNormalKernel

Abstract type for representing Normal kernels.
"""
abstract type AbstractNormalKernel <: AbstractMarkovKernel end

"""
    NormalKernel

Standard mean vector / covariance matrix parametrisation of Normal kernels.
"""
struct NormalKernel{T<:Skedasticity,TM,TC} <: AbstractNormalKernel
    μ::TM
    Σ::TC
end

NormalKernel(μ, Σ, sked::Skedasticity) =
    NormalKernel{typeof(sked),typeof(μ),typeof(Σ)}(μ, Σ)

"""
    Normal(μ, Σ)

Creates a Normal kernel with conditional mean and covariance parameters μ and  Σ, respectively.
"""
NormalKernel(μ, Σ) = NormalKernel(μ, Σ, skedasticity(Σ))

function NormalKernel(μ::AbstractAffineMap, Σ, ::Homoskedastic)
    T = promote_type(eltype(μ), eltype(Σ))
    μ = convert(AbstractAffineMap{T}, μ)
    Σ = convert_psd_eltype(T, Σ)
    return NormalKernel{Homoskedastic,typeof(μ),typeof(Σ)}(μ, Σ)
end

const HomoskedasticNormalKernel{TM,TC} = NormalKernel{<:Homoskedastic,TM,TC} where {TM,TC} # constant conditional covariance
const AffineHomoskedasticNormalKernel{TM,TC} =
    NormalKernel{<:Homoskedastic,TM,TC} where {TM<:AbstractAffineMap,TC} # affine conditional mean, constant conditional covariance
const AffineHeteroskedasticNormalKernel{TM,TC} =
    NormalKernel{<:Heteroskedastic,TM,TC} where {TM<:AbstractAffineMap,TC} # affine conditional mean, non-constant covariance
const NonlinearNormalKernel{TM,TC} = NormalKernel{<:Heteroskedastic,TM,TC} where {TM,TC} # the general, nonlinear case

const AffineIsotropicNormalKernel{TM,TC} =
    NormalKernel{<:Homoskedastic,TM,TC} where {TM<:AbstractAffineMap,TC<:UniformScaling}

"""
    mean(K::AbstractNormalKernel)

Computes the conditonal mean function of the Normal kernel K.
That is, the output is callable.
"""
mean(K::NormalKernel) = K.μ

"""
    covparam(K::AbstractNormalKernel)

Returns the internal representation of the conditonal covariance matrix of the Normal kernel K.
For computing the actual conditional covariance matrix, use cov.
"""
covparam(K::NormalKernel) = K.Σ

"""
    cov(K::AbstractNormalKernel)

Computes the conditonal covariance function of the Normal kernel K.
That is, the output is callable.
"""
cov(K::NormalKernel) = covparam(K)
cov(K::HomoskedasticNormalKernel) = x -> covparam(K)

condition(K::AbstractNormalKernel, x) = Normal(mean(K)(x), covparam(K)(x))
condition(K::HomoskedasticNormalKernel, x) = Normal(mean(K)(x), covparam(K))

function Base.copy!(
    Kdst::TK,
    Ksrc::TK,
) where {TM,TK<:HomoskedasticNormalKernel{TM,<:Cholesky}}
    copy!(mean(Kdst), mean(Ksrc))
    if covparam(Kdst).uplo == covparam(Ksrc).uplo
        copy!(covparam(Kdst).factors, covparam(Ksrc).factors)
    else
        copy!(covparam(Kdst).factors, adjoint(covparam(Ksrc).factors))
    end
    return Kdst
end

function Base.similar(K::HomoskedasticNormalKernel{TM,<:Cholesky}) where {TM}
    C = covparam(K)
    return NormalKernel(similar(mean(K)), Cholesky(similar(C.factors), C.uplo, C.info))
end

"""
    rand([rng::AbstractRNG], K::AbstractNormalKernel, x::AbstractVector)

Samples a random vector conditionally on x with respect the the Normal kernel K
using the random number generator rng.
"""
rand(rng::AbstractRNG, K::AbstractNormalKernel, x::AbstractNumOrVec) =
    rand(rng, condition(K, x))
rand(K::AbstractNormalKernel, x) = rand(Random.default_rng(), K, x)

function Base.show(io::IO, N::NormalKernel)
    println(io, summary(N))
    print(io, "μ = ")
    show(io, N.μ)
    print(io, "\nΣ = ")
    show(io, N.Σ)
end
