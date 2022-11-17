"""
    AbstractNormal{T<:Number}

Abstract type for representing Normal distributed random vectors taking values in T.
"""
abstract type AbstractNormal{T} <: AbstractDistribution{T} end

"""
    Normal{T,U,V}

Standard mean vector / covariance matrix parametrisation of the normal distribution with element type T.
"""
struct Normal{T,U,V} <: AbstractNormal{T}
    μ::U
    Σ::V
    Normal{T}(μ, Σ) where {T} = new{T,typeof(μ),typeof(Σ)}(μ, Σ)
end

"""
    Normal(μ::AbstractVector, Σ::CovarianceParameter)

Creates a Normal distribution with mean vector μ and covariance matrix parametrised by Σ.
"""
function Normal(μ::AbstractVector, Σ::CovarianceParameter)
    T = promote_type(eltype(μ), eltype(Σ))
    return Normal{T}(convert(AbstractVector{T}, μ), convert(CovarianceParameter{T}, Σ))
end

function Normal(μ::AbstractVector, Σ::Symmetric)
    T = promote_type(eltype(μ), eltype(Σ))
    T <: Complex && throw(DomainError(Σ, "Complex valued covariance must be Hermitian"))
    return Normal{T}(convert(AbstractVector{T}, μ), convert(CovarianceParameter{T}, Σ))
end

"""
    Normal(μ::AbstractVector, Σ::AbstractMatrix)

Creates a Normal distribution with mean vector μ and covariance matrix Σ
if Σ is Symmetric / Hermitian. Throws domain error otherwise.
"""
function Normal(μ::AbstractVector, Σ::AbstractMatrix)
    T = promote_type(eltype(μ), eltype(Σ))
    if T <: Real
        issymmetric(Σ) && return Normal(μ, Symmetric(Σ))
        throw(DomainError(Σ, "Real valued covariance must be symmetric"))
    elseif T <: Complex
        ishermitian(Σ) && return Normal(μ, Hermitian(Σ))
        throw(DomainError(Σ, "Complex valued covariance must be Hermitian"))
    end
end

"""
    Normal{T}(N::Normal{U,V,W})

Computes a Normal distribution of eltype T from the Normal distribution N if T and U are compatible.
That is T and U must both be Real or both be Complex.
"""
function Normal{T}(N::Normal{U,V,W}) where {T,U,V<:AbstractVector,W<:CovarianceParameter}
    T <: Real && U <: Real || T <: Complex && U <: Complex ?
    Normal(convert(AbstractVector{T}, N.μ), convert(CovarianceParameter{T}, N.Σ)) :
    error(
        "The constructor type $(T) and the argument type $(U) must both be real or both be complex",
    )
end

AbstractDistribution{T}(N::AbstractNormal) where {T} = AbstractNormal{T}(N)
AbstractNormal{T}(N::AbstractNormal{T}) where {T} = N
AbstractNormal{T}(N::Normal) where {T} = Normal{T}(N)

"""
    dim(N::AbstractNormal)

Returns the dimension of the Normal distribution N.
"""
dim(N::Normal) = length(N.μ)

"""
    mean(N::AbstractNormal)

Computes the mean vector of the Normal distribution N.
"""
mean(N::Normal) = N.μ

"""
    cov(N::AbstractNormal)

Computes the covariance matrix of the Normal distribution N.
"""
cov(N::Normal) = AbstractMatrix(N.Σ)
cov(N::Normal{T,U,V}) where {T,U,V<:AbstractMatrix} = N.Σ

"""
    covp(N::AbstractNormal)

Returns the internal representation of the covariance matrix of the Normal distribution N.
For computing the actual covariance matrix, use cov.
"""
covp(N::Normal) = N.Σ

"""
    var(N::AbstractNormal)
Computes the vector of marginal variances of the Normal distribution N.
"""
var(N::AbstractNormal) = real(diag(covp(N)))
var(N::Normal{T,U,V}) where {T,U,V<:Cholesky} = vec(sum(abs2, covp(N).L, dims = 2)) # this is broken for StaticArrays

"""
    std(N::AbstractNormal)
Computes the vector of marginal standard deviations of the Normal distribution N.
"""
std(N::AbstractNormal) = sqrt.(var(N)) # this is broken for StaticArrays

"""
    residual(N::AbstractNormal, x::AbstractVector)

Computes the whitened residual associated with the Normal distribution N and observed vector x.
"""
residual(N::AbstractNormal, x::AbstractVector) = lsqrt(covp(N)) \ (x - mean(N))

_nscale(T::Type{<:Real}) = T(0.5)
_nscale(T::Type{<:Complex}) = one(real(T))

_piconst(T::Type{<:Real}) = T(2π)
_piconst(T::Type{<:Complex}) = real(T)(π)

"""
    logpdf(N::AbstractNormal,x)

Computes the logarithm of the probability density function of the Normal distribution N evaluated at x.
"""
logpdf(N::AbstractNormal{T}, x) where {T} =
    -_nscale(T) * (dim(N) * log(_piconst(T)) + logdet(covp(N)) + norm_sqr(residual(N, x)))

"""
    entropy(N::AbstractNormal)

Computes the entropy of the Normal distribution N.
"""
entropy(N::AbstractNormal{T}) where {T} =
    _nscale(T) * (dim(N) * (log(_piconst(T)) + one(real(T))) + logdet(covp(N)))

"""
    kldivergence(N1::AbstractNormal, N2::AbstractNormal)

Computes the Kullback-Leibler divergence between the Normal distributions N1 and N2.
"""
function kldivergence(N1::AbstractNormal{T}, N2::AbstractNormal{T}) where {T<:Number}
    root_ratio = lsqrt(covp(N2)) \ lsqrt(covp(N1))
    _nscale(T) * (
        norm_sqr(root_ratio) + norm_sqr(residual(N2, mean(N1))) - dim(N1) -
        real(T)(2) * real(logdet(root_ratio))
    )
end

"""
    rand(RNG::AbstractRNG, N::AbstractNormal)

Computes a random vector distributed according to the Normal distribution N
using the random number generator RNG.
"""
rand(RNG::AbstractRNG, N::AbstractNormal) =
    mean(N) + lsqrt(covp(N)) * randn(RNG, eltype(N), dim(N))

"""
    rand(N::AbstractNormal)

Computes a random vector distributed according to the Normal distribution N
using the random number generator Random.GLOBAL_RNG.
"""
rand(N::AbstractNormal) = rand(GLOBAL_RNG, N)

function Base.show(io::IO, N::Normal{T,U,V}) where {T,U,V}
    print(io, "Normal{$T,$U,$V}(μ, Σ)")
    print(io, "\n μ = ")
    show(io, (N.μ))
    print(io, "\n Σ = ")
    show(io, N.Σ)
end
