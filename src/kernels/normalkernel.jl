"""
    AbstractNormalKernel

Abstract type for representing Normal kernels.
"""
abstract type AbstractNormalKernel <: AbstractMarkovKernel end

"""
    NormalKernel

Standard mean vector / covariance matrix parametrisation of Normal kernels.
"""
struct NormalKernel{U,V} <: AbstractNormalKernel
    μ::U
    Σ::V
end

struct HomoskedasticNormalKernel{A,B} <: AbstractNormalKernel
    μ::A
    Σ::B
end

function HomoskedasticNormalKernel(F::AbstractAffineMap, Σ)
    T = promote_type(eltype(F), eltype(Σ))
    F = convert(AbstractAffineMap{T}, F)
    Σ = selfadjoint(T, Σ)
    return HomoskedasticNormalKernel{typeof(F),typeof(Σ)}(F, Σ)
end

const AffineHomoskedasticNormalKernel{A,B} = NormalKernel{A,B} where {A<:AbstractAffineMap}

const AffineNormalKernel{T} =
    NormalKernel{<:AbstractAffineMap{T},<:CovarianceParameter{T}} where {T}

"""
    mean(K::AbstractNormalKernel)

Computes the conditonal mean function of the Normal kernel K.
That is, the output is callable.
"""
mean(K::NormalKernel) = K.μ
mean(K::HomoskedasticNormalKernel) = K.μ

"""
    covp(K::AbstractNormalKernel)

Returns the internal representation of the conditonal covariance matrix of the Normal kernel K.
For computing the actual conditional covariance matrix, use cov.
"""
covp(K::NormalKernel) = K.Σ
covp(K::HomoskedasticNormalKernel) = K.Σ

"""
    cov(K::AbstractNormalKernel)

Computes the conditonal covariance matrix function of the Normal kernel K.
That is, the output is callable.
"""
cov(K::NormalKernel) = K.Σ
cov(K::HomoskedasticNormalKernel) = x -> covp(K)
cov(K::AffineNormalKernel) = x -> covp(K)

function Base.copy!(
    Kdst::HomoskedasticNormalKernel{A,<:Cholesky},
    Ksrc::HomoskedasticNormalKernel{B,<:Cholesky},
) where {A,B}
    copy!(mean(Kdst), mean(Ksrc))
    if covp(Kdst).uplo == covp(Ksrc).uplo
        copy!(covp(Kdst).factors, covp(Ksrc).factors)
    else
        copy!(covp(Kdst).factors, adjoint(covp(Ksrc).factors))
    end
    return Kdst
end

function Base.copy!(
    Kdst::A,
    Ksrc::A,
) where {T,V<:Cholesky,A<:AffineNormalKernel{T,<:AbstractAffineMap{T},V}}
    copy!(mean(Kdst), mean(Ksrc))
    if covp(Kdst).uplo == covp(Ksrc).uplo
        copy!(covp(Kdst).factors, covp(Ksrc).factors)
    else
        copy!(covp(Kdst).factors, adjoint(covp(Ksrc).factors))
    end
    return Kdst
end

Base.similar(K::HomoskedasticNormalKernel{A,<:Cholesky}) where {A} =
    HomoskedasticNormalKernel(
        similar(mean(K)),
        Cholesky(similar(covp(K).factors), covp(K).uplo, covp(K).info),
    )
Base.similar(K::AffineNormalKernel{T,<:AbstractAffineMap{T},<:Cholesky}) where {T} =
    NormalKernel(
        similar(mean(K)),
        Cholesky(similar(covp(K).factors), covp(K).uplo, covp(K).info),
    )

"""
    NormalKernel(F::AbstractAffineMap{<:Real}, Σ::Symmetric{<:Real})

Creates a NormalKernel with conditional mean function F and a constant conditional covariance function parameterised by Σ.
"""
function NormalKernel(F::AbstractAffineMap{<:Real}, Σ::Symmetric{<:Real})
    T = promote_type(eltype(F), eltype(Σ))
    F = convert(AbstractAffineMap{T}, F)
    Σ = convert(AbstractMatrix{T}, Σ)
    return NormalKernel{typeof(F),typeof(Σ)}(F, Σ)
end

function NormalKernel(F::AbstractAffineMap{<:Complex}, Σ::Hermitian{<:Complex})
    T = promote_type(eltype(F), eltype(Σ))
    F = convert(AbstractAffineMap{T}, F)
    Σ = convert(AbstractMatrix{T}, Σ)
    return NormalKernel{typeof(F),typeof(Σ)}(F, Σ)
end

function NormalKernel(F::AbstractAffineMap, Σ::Factorization)
    T = promote_type(eltype(F), eltype(Σ))
    F = convert(AbstractAffineMap{T}, F)
    Σ = convert(Factorization{T}, Σ)
    return NormalKernel{typeof(F),typeof(Σ)}(F, Σ)
end

"""
    condition(K::AbstractNormalKernel, x)

Returns a Normal distribution corresponding to K evaluated at x.
"""
condition(K::AbstractNormalKernel, x) = Normal(mean(K)(x), cov(K)(x))
condition(K::HomoskedasticNormalKernel, x) = Normal(mean(K)(x), covp(K))
condition(K::AffineNormalKernel, x) = Normal(mean(K)(x), covp(K))

"""
    rand(RNG::AbstractRNG, K::AbstractNormalKernel, x::AbstractVector)

Computes a random vector conditionally on x with respect the the Normal kernel K
using the random number generator RNG.
"""
rand(RNG::AbstractRNG, K::AbstractNormalKernel, x::AbstractVector) =
    rand(RNG, condition(K, x))

"""
    rand(K::AbstractNormalKernel, x::AbstractVector)
Computes a random vector conditionally on x with respect the the Normal kernel K
using the random number generator Random.GLOBAL_RNG.
"""
rand(K::AbstractNormalKernel, x::AbstractVector) = rand(GLOBAL_RNG, K, x)

function Base.show(io::IO, K::AbstractNormalKernel)
    println(io, summary(K))
    println(io, "μ = ")
    show(io, mean(K))
    println(io, "\nΣ = ")
    show(io, covp(K))
end
