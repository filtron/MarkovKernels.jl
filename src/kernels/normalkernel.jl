"""
    AbstractNormalKernel{T<:Number}

Abstract type for representing Normal kernels taking values in T.
"""
abstract type AbstractNormalKernel{T<:Number} <: AbstractMarkovKernel end

eltype(::AbstractNormalKernel{T}) where {T} = T

"""
    NormalKernel

Standard parametrisation of Normal kernels.
"""
struct NormalKernel{T,U,V} <: AbstractNormalKernel{T}
    μ::U
    Σ::V
    function NormalKernel(μ, Σ)
        new{eltype(μ),typeof(μ),typeof(Σ)}(μ, Σ)
    end
end

const AffineNormalKernel{T} =
    NormalKernel{T,<:AbstractAffineMap,<:Union{UniformScaling,AbstractMatrix}}

const AffineIsoNormalKernel{T} = NormalKernel{T,<:AbstractAffineMap,<:UniformScaling}

"""
    NormalKernel(Φ::AbstractMatrix, Σ)

Creates a Normal kernel with linear conditional mean of slope Φ and covariance parameter Σ.
"""
NormalKernel(Φ::AbstractMatrix, Σ) = NormalKernel(AffineMap(Φ), Σ)

"""
    NormalKernel(Φ::AbstractMatrix, b::AbstractVector, Σ::AbstractMatrix)

Creates a Normal kernel with affine conditional mean of slope Φ, intercept b, and covariance matrix Σ.
"""
NormalKernel(Φ::AbstractMatrix, b::AbstractVector, Σ::AbstractMatrix) =
    NormalKernel(AffineMap(Φ, b), Σ)
NormalKernel(
    Φ::AbstractMatrix,
    b::AbstractVector,
    pred::AbstractVector,
    Σ::AbstractMatrix,
) = NormalKernel(AffineMap(Φ, b, pred), Σ)

mean(K::NormalKernel) = K.μ

cov(K::NormalKernel) = K.Σ

condition(K::NormalKernel, x) = Normal(mean(K)(x), cov(K))

compose(K2::AffineNormalKernel{T}, K1::AffineNormalKernel{T}) where {T} =
    NormalKernel(compose(mean(K2), mean(K1)), stein(cov(K1), mean(K2), cov(K2)))

*(K2::AbstractNormalKernel, K1::AbstractNormalKernel) = compose(K2, K1)

marginalise(N::Normal{T}, K::AffineNormalKernel{T}) where {T} =
    Normal(mean(K)(mean(N)), stein(cov(N), mean(K), cov(K)))

function invert(N::Normal{T}, K::AffineNormalKernel{T}) where {T}
    pred = mean(K)(mean(N))

    S, G, Σ = schur_red(N.Σ, slope(K.μ), K.Σ)

    Nout = Normal(pred, S)

    corrector = AffineMap(G, mean(N), pred)
    Kout = NormalKernel(corrector, Σ)

    return Nout, Kout
end

rand(RNG::AbstractRNG, K::NormalKernel, x::AbstractVector) = rand(RNG, condition(K, x))
rand(K::NormalKernel, x::AbstractVector) = rand(GLOBAL_RNG, K, x)
