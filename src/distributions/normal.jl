"""
    AbstractNormal{T<:Number}

Abstract type for representing normal distributed random vectors taking values in T.
"""
abstract type AbstractNormal{T} <: AbstractDistribution{T} end

==(N1::T, N2::T) where {T<:AbstractNormal} =
    all(f -> getfield(N1, f) == getfield(N2, f), 1:nfields(N1))

"""
    Normal{T,U,V}

Standard parametrisation of the normal distribution with element type T.
"""
struct Normal{T,U,V} <: AbstractNormal{T}
    μ::U
    Σ::V
    Normal{T}(μ, Σ) where {T} = new{T,typeof(μ),typeof(Σ)}(μ, Σ)
end

function Normal(μ::AbstractVector, Σ::CovarianceParameter)
    T = promote_type(eltype(μ), eltype(Σ))
    return Normal{T}(convert(AbstractVector{T}, μ), convert(CovarianceParameter{T}, Σ))
end

function Normal(μ::AbstractVector, Σ::Symmetric)
    T = promote_type(eltype(μ), eltype(Σ))
    T <: Complex && throw(DomainError(Σ, "Complex valued covariance must be Hermitian"))
    return Normal{T}(convert(AbstractVector{T}, μ), convert(CovarianceParameter{T}, Σ))
end

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

function Normal{T}(N::Normal{U,V,W}) where {T,U,V<:AbstractVector,W<:CovarianceParameter}
    T <: Real && U <: Real || T <: Complex && U <: Complex ?
    Normal(convert(AbstractVector{T}, N.μ), convert(CovarianceParameter{T}, N.Σ)) :
    error(
        "The constructor type $(T) and the argument type $(U) must both be real or both be complex",
    )
end

const IsoNormal{T,U} = Normal{T,U,<:UniformScaling}
IsoNormal(μ::AbstractVector, λ::Real) = Normal(μ, λ * I)

AbstractDistribution{T}(N::AbstractNormal) where {T} = AbstractNormal{T}(N)
AbstractNormal{T}(N::AbstractNormal{T}) where {T} = N
AbstractNormal{T}(N::Normal) where {T} = Normal{T}(N)

"""
    dim(N::AbstractNormal)

Returns the dimension of the normal distribution N.
"""
dim(N::Normal) = length(N.μ)

"""
    covp(N::Normal)

Returns the internal representation of the covariance matrix of the Normal distribution N.
For computing the actual covariance matrix, use cov.
"""
covp(N::Normal) = N.Σ

mean(N::Normal) = N.μ

cov(N::Normal) = AbstractMatrix(N.Σ)
cov(N::Normal{T,U,V}) where {T,U,V<:AbstractMatrix} = N.Σ
cov(N::IsoNormal) = covp(N)(dim(N))

var(N::AbstractNormal) = real(diag(covp(N)))
var(N::IsoNormal) = real(diag(N.Σ(dim(N))))
var(N::Normal{T,U,V}) where {T,U,V<:Cholesky} = vec(sum(abs2, covp(N).L, dims = 2))

std(N::AbstractNormal) = sqrt.(var(N))

"""
    residual(N::AbstractNormal,x)

Returns the whitened residual associated with N and observed vector x.
"""
residual(N::AbstractNormal, x) = lsqrt(covp(N)) \ (x - mean(N))

_nscale(T::Type{<:Real}) = T(0.5)
_nscale(T::Type{<:Complex}) = one(real(T))

_piconst(T::Type{<:Real}) = T(2π)
_piconst(T::Type{<:Complex}) = real(T)(π)

"""
    logpdf(N::AbstractNormal,x)

Returns the logarithm of the probability density function of N evaluated at x.
"""
logpdf(N::AbstractNormal{T}, x) where {T} =
    -_nscale(T) * (dim(N) * log(_piconst(T)) + logdet(covp(N)) + norm_sqr(residual(N, x)))
logpdf(N::IsoNormal{T}, x) where {T} =
    -_nscale(T) * (dim(N) * (log(_piconst(T)) + log(abs(N.Σ.λ))) + norm_sqr(residual(N, x)))

"""
    entropy(N::AbstractNormal)

Returns the entropy of N.
"""
entropy(N::AbstractNormal{T}) where {T} =
    _nscale(T) * (dim(N) * (log(_piconst(T)) + one(real(T))) + logdet(covp(N)))
entropy(N::IsoNormal{T}) where {T} =
    _nscale(T) * (dim(N) * (log(_piconst(T)) + one(real(T))) + dim(N) * log(abs(covp(N).λ)))

"""
    kldivergence(N1::AbstractNormal,N2::AbstractNormal)

Returns the Kullback-Leibler divergence between N1 and N2.
"""
function kldivergence(N1::AbstractNormal{T}, N2::AbstractNormal{T}) where {T<:Number}
    root_ratio = lsqrt(covp(N2)) \ lsqrt(covp(N1))
    _nscale(T) * (
        norm_sqr(root_ratio) + norm_sqr(residual(N2, mean(N1))) - dim(N1) -
        real(T)(2) * real(logdet(root_ratio))
    )
end
function kldivergence(N1::IsoNormal{T}, N2::IsoNormal{T}) where {T<:Number}
    ratio = abs(covp(N2).λ \ covp(N1).λ)
    _nscale(T) *
    (dim(N1) * ratio + norm_sqr(residual(N2, mean(N1))) - dim(N1) - dim(N1) * log(ratio))
end

rand(RNG::AbstractRNG, N::AbstractNormal) =
    mean(N) + lsqrt(covp(N)) * randn(RNG, eltype(N), dim(N))
rand(N::AbstractNormal) = rand(GLOBAL_RNG, N)
