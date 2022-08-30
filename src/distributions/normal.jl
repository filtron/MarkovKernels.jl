"""
    AbstractNormal{T<:Number}

Abstract type for representing normal distributed random vectors taking values in T.
"""
abstract type AbstractNormal{T<:Number} <: AbstractDistribution end

eltype(::AbstractNormal{T}) where {T} = T

"""
    Normal{T,U,V}

Standard parametrisation of the normal distribution with element type T.
"""
struct Normal{T,U,V} <: AbstractNormal{T}
    μ::U
    Σ::V
    function Normal(μ::AbstractVector, Σ::AbstractMatrix)
        T = promote_type(eltype(μ), eltype(Σ))
        new{T,typeof(μ),typeof(Σ)}(μ, Σ)
    end
end

similar(N::Normal) = Normal(similar(N.μ), similar(N.Σ))
==(N1::Normal, N2::Normal) = N1.μ == N2.μ && N1.Σ == N2.Σ

dim(N::Normal) = length(N.μ)
mean(N::Normal) = N.μ
cov(N::Normal) = Hermitian(Matrix(N.Σ))
var(N::Normal) = real(diag(N.Σ))
std(N::Normal) = sqrt.(var(N))

residual(N::Normal, x) = cholesky(N.Σ).L \ (x .- N.μ)

_nscale(T::Type{<:Real}) = T(0.5)
_nscale(T::Type{<:Complex}) = one(real(T))

_piconst(T::Type{<:Real}) = T(2π)
_piconst(T::Type{<:Complex}) = real(T)(π)

logpdf(N::Normal{T}, x) where {T} =
    -_nscale(T) * (logdet(_piconst(T) * N.Σ) + norm_sqr(residual(N, x)))
entropy(N::Normal{T}) where {T} =
    _nscale(T) * (dim(N) * (log(_piconst(T)) + 1) + logdet(N.Σ))

function kldivergence(N1::Normal{T}, N2::Normal{T}) where {T<:Number}
    root_ratio = lsqrt(N2.Σ) \ lsqrt(N1.Σ)
    _nscale(T) * (
        norm_sqr(root_ratio) + norm_sqr(residual(N2, N1.μ)) - dim(N1) -
        real(T)(2) * real(logdet(root_ratio))
    )
end

rand(RNG::AbstractRNG, N::Normal{T,U,V}) where {T,U,V} =
    N.μ + lsqrt(N.Σ) * randn(RNG, eltype(N), dim(N))
rand(N::Normal) = rand(GLOBAL_RNG, N)
