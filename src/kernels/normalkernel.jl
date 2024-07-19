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

"""
    NormalKernel(Φ::AbstractMatrix, Σ)

Creates a NormalKernel with a linear conditional mean function given by

    x ↦ Φ * x,

and conditional covariance function parameter Σ.
Σ is assumed to be callable and be of compatible eltype with Φ.
"""
NormalKernel(Φ::AbstractMatrix, Σ) = NormalKernel(LinearMap(Φ), Σ)

"""
    NormalKernel(Φ::AbstractMatrix, b::AbstractVector, Σ)

Creates a NormalKernel with an affine conditional mean function given by

    x ↦ b + Φ * x,

and conditional covariance function parameter Σ.
Σ is assumed to be callable and be of compatible eltype with Φ, b.
"""
NormalKernel(Φ::AbstractMatrix, b::AbstractVector, Σ) = NormalKernel(AffineMap(Φ, b), Σ)

"""
    NormalKernel(Φ::AbstractMatrix, b::AbstractVector, c::AbstractVector, Σ)

Creates a NormalKernel with an affine corrector conditional mean function given by

    x ↦ b + Φ * (x - c),

and conditional covariance function parameter Σ.
Σ is assumed to be callable and be of compatible eltype with Φ, b, c.
"""
NormalKernel(Φ::AbstractMatrix, b::AbstractVector, c::AbstractVector, Σ) =
    NormalKernel(AffineCorrector(Φ, b, c), Σ)

const AffineNormalKernel{T} =
    NormalKernel{<:AbstractAffineMap{T},<:CovarianceParameter{T}} where {T}

function Base.copy!(
    Kdst::A,
    Ksrc::A,
) where {T,V<:Cholesky,A<:AffineNormalKernel{T,<:AbstractAffineMap{T},V}}
    copy!(mean(Kdst), mean(Ksrc))
    covp(Kdst).uplo !== covp(Ksrc).uplo &&
        throw(ArgumentError("Both arguments need to have Cholesy factors with same uplo"))
    # should throw on different info as well?
    copy!(covp(Kdst).factors, covp(Ksrc).factors)
    return Kdst
end
# similar not implemented for Cholesky, argh...
function Base.similar(K::AffineNormalKernel{T,<:AbstractAffineMap{T},<:Cholesky}) where {T}
    C = covp(K)
    return NormalKernel(similar(mean(K)), Cholesky(similar(C.factors), C.uplo, C.info))
end

"""
    NormalKernel(F::AbstractAffineMap, Σ::CovarianceParameter)

Creates a NormalKernel with conditional mean function F and a constant conditional covariance function parameterised by Σ.
"""
function NormalKernel(F::AbstractAffineMap, Σ::CovarianceParameter)
    T = promote_type(eltype(F), eltype(Σ))
    F = convert(AbstractAffineMap{T}, F)
    Σ = convert(CovarianceParameter{T}, Σ)
    return NormalKernel{typeof(F),typeof(Σ)}(F, Σ)
end

function NormalKernel(F::AbstractAffineMap, Σ::Symmetric)
    T = promote_type(eltype(F), eltype(Σ))
    T <: Complex && throw(DomainError(Σ, "Complex valued covariance must be Hermitian"))
    F = convert(AbstractAffineMap{T}, F)
    Σ = convert(CovarianceParameter{T}, Σ)
    return NormalKernel{typeof(F),typeof(Σ)}(F, Σ)
end

"""
    NormalKernel(F::AbstractAffineMap, Σ::AbstractMatrix)

Creates a NormalKernel with conditional mean function F and a constant conditional covariance function Σ
if Σ is Symmetric / Hermitian. Throws domain error otherwise.
"""
function NormalKernel(F::AbstractAffineMap, Σ::AbstractMatrix)
    T = promote_type(eltype(F), eltype(Σ))
    if T <: Real
        issymmetric(Σ) && return NormalKernel(F, Symmetric(Σ))
        throw(DomainError(Σ, "Real valued covariance must be symmetric"))
    elseif T <: Complex
        ishermitian(Σ) && return NormalKernel(F, Hermitian(Σ))
        throw(DomainError(Σ, "Complex valued covariance must be Hermitian"))
    end
end

"""
    mean(K::AbstractNormalKernel)

Computes the conditonal mean function of the Normal kernel K.
That is, the output is callable.
"""
mean(K::NormalKernel) = K.μ

"""
    mean(K::AbstractNormalKernel)

Computes the conditonal covariance matrix function of the Normal kernel K.
That is, the output is callable.
"""
cov(K::NormalKernel) = K.Σ
cov(K::AffineNormalKernel) = x -> K.Σ

"""
    covp(K::AbstractNormalKernel)

Returns the internal representation of the conditonal covariance matrix of the Normal kernel K.
For computing the actual conditional covariance matrix, use cov.
"""
covp(K::NormalKernel) = K.Σ

"""
    condition(K::AbstractNormalKernel, x)

Returns a Normal distribution corresponding to K evaluated at x.
"""
condition(K::AbstractNormalKernel, x) = Normal(mean(K)(x), cov(K)(x))
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

function Base.show(io::IO, N::NormalKernel)
    println(io, summary(N))
    println(io, "μ = ")
    show(io, N.μ)
    println(io, "\nΣ = ")
    show(io, N.Σ)
end
