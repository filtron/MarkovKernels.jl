"""
    AbstractNormalKernel{T<:Number}

Abstract type for representing Normal kernels taking values in T.
"""
abstract type AbstractNormalKernel{T} <: AbstractMarkovKernel{T} end

"""
    NormalKernel

Standard mean vector / covariance matrix parametrisation of Normal kernels.
"""
struct NormalKernel{T,U,V} <: AbstractNormalKernel{T}
    μ::U
    Σ::V
end

NormalKernel{T}(μ, Σ) where {T} = NormalKernel{T,typeof(μ),typeof(Σ)}(μ, Σ)

"""
    NormalKernel(F::AbstractAffineMap, Σ)

Creates a NormalKernel with conditional mean function F and conditional covariance function parameter Σ.
Σ is assumed to be callable and be of compatible eltype with F.
"""
NormalKernel(F::AbstractAffineMap, Σ) = NormalKernel{eltype(F)}(F, Σ)

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

const AffineNormalKernel{T} = NormalKernel{T,<:AbstractAffineMap,<:CovarianceParameter}

function Base.copy!(Kdst::A, Ksrc::A) where {T,U,V<:Cholesky,A<:AffineNormalKernel{T,U,V}}
    copy!(mean(Kdst), mean(Ksrc))
    covp(Kdst).uplo !== covp(Ksrc).uplo &&
        throw(ArgumentError("Both arguments need to have Cholesy factors with same uplo"))
    # should throw on different info as well? 
    copy!(covp(Kdst).factors, covp(Ksrc).factors)
    return Kdst
end
# similar not implemented for Cholesky, argh...
function Base.similar(K::AffineNormalKernel{T,U,<:Cholesky}) where {T,U}
    C = covp(K)
    return NormalKernel(similar(mean(K)), Cholesky(similar(C.factors), C.uplo, C.info))
end

"""
    NormalKernel(F::AbstractAffineMap, Σ::CovarianceParameter)

Creates a NormalKernel with conditional mean function F and a constant conditional covariance function parameterised by Σ.
"""
function NormalKernel(F::AbstractAffineMap, Σ::CovarianceParameter)
    T = promote_type(eltype(F), eltype(Σ))
    return NormalKernel{T}(
        convert(AbstractAffineMap{T}, F),
        convert(CovarianceParameter{T}, Σ),
    )
end

function NormalKernel(F::AbstractAffineMap, Σ::Symmetric)
    T = promote_type(eltype(F), eltype(Σ))
    T <: Complex && throw(DomainError(Σ, "Complex valued covariance must be Hermitian"))
    return NormalKernel{T}(
        convert(AbstractAffineMap{T}, F),
        convert(CovarianceParameter{T}, Σ),
    )
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
    NormalKernel{T}(K::AffineNormalKernel{U}) where {T,U}

Computes a Normal kernel of eltype T from the Normal kernel K if T and U are compatible.
That is T and U must both be Real or both be Complex.
"""
function NormalKernel{T}(K::AffineNormalKernel{U}) where {T,U}
    T <: Real && U <: Real || T <: Complex && U <: Complex ?
    NormalKernel(convert(AbstractAffineMap{T}, K.μ), convert(CovarianceParameter{T}, K.Σ)) :
    error(
        "The constructor type $(T) and the argument type $(U) must both be real or both be complex",
    )
end

AbstractMarkovKernel{T}(K::AbstractNormalKernel) where {T} = AbstractNormalKernel{T}(K)
AbstractNormalKernel{T}(K::AbstractNormalKernel{T}) where {T} = K
AbstractNormalKernel{T}(K::NormalKernel) where {T} = NormalKernel{T}(K)

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

function Base.show(io::IO, N::NormalKernel{T,U,V}) where {T,U,V}
    print(io, "NormalKernel{$T,$U,$V}(μ, Σ)")
    print(io, "\n μ = ")
    show(io, N.μ)
    print(io, "\n Σ = ")
    show(io, N.Σ)
end
