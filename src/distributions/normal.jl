"""
    AbstractNormal{T<:Number}

Abstract type for representing normal distributed random vectors taking values in T.
"""
abstract type AbstractNormal{T<:Number} <: AbstractDistribution end

eltype(::AbstractNormal{T}) where {T} = T

AbstractNormal{T}(N::AbstractNormal{T}) where {T} = N
convert(::Type{T}, N::T) where {T<:AbstractNormal} = N
convert(::Type{T}, N::AbstractNormal) where {T<:AbstractNormal} = T(N)::T

==(N1::T, N2::T) where {T<:AbstractNormal} = all(f -> getfield(N1, f) == getfield(N2, f), 1:nfields(N1))

"""
    Normal{T,U,V}

Standard parametrisation of the normal distribution with element type T.
"""
struct Normal{T,U,V} <: AbstractNormal{T}
    μ::U
    Σ::V
    Normal{T}(μ, Σ) where {T} = new{T,typeof(μ),typeof(Σ)}(μ, Σ)
end

for c in (:AbstractMatrix, :Factorization)
    @eval function Normal(μ::AbstractVector, Σ::$c)
        T = promote_type(eltype(μ), eltype(Σ))
        return Normal{T}(convert(AbstractVector{T}, μ), symmetrise(convert($c{T}, Σ)))
    end
    @eval Normal{T}(K::Normal{U,V,W}) where {T,U,V<:AbstractVector,W<:$c} =
        Normal(convert(AbstractVector{T}, K.μ), convert($c{T}, K.Σ))
end

for c in (:Diagonal, :UniformScaling)
    @eval function Normal(μ::AbstractVector, Σ::$c)
        T = promote_type(eltype(μ), eltype(Σ))
        return Normal{T}(convert(AbstractVector{T}, μ), symmetrise(convert($c{real(T)}, Σ)))
    end
    @eval Normal{T}(K::Normal{U,V,W}) where {T,U,V<:AbstractVector,W<:$c} =
        T <: Real && U <: Real || T <: Complex && U <: Complex ?
        Normal(convert(AbstractVector{T}, K.μ), convert($c{real(T)}, K.Σ)) :
        error("T and U must both be complex or both be real")
end

AbstractNormal{T}(K::Normal) where {T} = Normal{T}(K)

const IsoNormal{T,U} = Normal{T,U,<:UniformScaling}
IsoNormal(μ::AbstractVector, λ::Real) = Normal(μ, λ * I)

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
    -_nscale(T) * (dim(N) * (log(_piconst(T)) + log(N.Σ.λ)) + norm_sqr(residual(N, x)))

"""
    entropy(N::AbstractNormal)

Returns the entropy of N.
"""
entropy(N::AbstractNormal{T}) where {T} =
    _nscale(T) * (dim(N) * (log(_piconst(T)) + one(real(T))) + logdet(covp(N)))
entropy(N::IsoNormal{T}) where {T} =
    _nscale(T) * (dim(N) * (log(_piconst(T)) + one(real(T))) + dim(N) * log(covp(N).λ))

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
    ratio = covp(N2).λ \ covp(N1).λ
    _nscale(T) *
    (dim(N1) * ratio + norm_sqr(residual(N2, mean(N1))) - dim(N1) - dim(N1) * log(ratio))
end

rand(RNG::AbstractRNG, N::AbstractNormal) =
    mean(N) + lsqrt(covp(N)) * randn(RNG, eltype(N), dim(N))
rand(N::AbstractNormal) = rand(GLOBAL_RNG, N)
