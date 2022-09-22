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
        Σ = symmetrise(Σ)
        new{eltype(μ),typeof(μ),typeof(Σ)}(μ, Σ)
    end
end

const AffineNormalKernel{T} =
    NormalKernel{T,<:AbstractAffineMap,<:Union{UniformScaling,AbstractMatrix}}

NormalKernel(Φ::AbstractMatrix, Σ) = NormalKernel(AffineMap(Φ), Σ)
NormalKernel(Φ::AbstractMatrix, b::AbstractVector, Σ) = NormalKernel(AffineMap(Φ, b), Σ)
NormalKernel(Φ::AbstractMatrix, b::AbstractVector, pred::AbstractVector, Σ) =
    NormalKernel(AffineMap(Φ, b, pred), Σ)

"""
    covp(K::NormalKernel)

Returns the internal representation of the conditonal covariance matrix of the Normal distribution N.
For computing the actual covariance matrix, use cov.
"""
covp(K::NormalKernel) = K.Σ

mean(K::NormalKernel) = K.μ
cov(K::NormalKernel) = x -> K.Σ # needs to return callable

"""
    condition(K::AbstractNormalKernel, x)

Returns a Normal distribution corresponding to K evaluated at x.
"""
condition(K::AbstractNormalKernel, x) = Normal(mean(K)(x), cov(K)(x))
condition(K::AffineNormalKernel, x) = Normal(mean(K)(x), covp(K))

"""
    compose(K2::AffineNormalKernel, K1::AffineNormalKernel)

Returns K3, the composition of K2 ∘ K1 i.e,

K3(y,x) = ∫ K2(y,z) K1(z,x) dz
"""
compose(K2::AffineNormalKernel{T}, K1::AffineNormalKernel{T}) where {T} =
    NormalKernel(compose(mean(K2), mean(K1)), stein(covp(K1), mean(K2), covp(K2)))

"""
    marginalise(N::AbstractNormal, K::AffineNormalKernel)

Returns M, K marginalised with respect to N i.e,

M(y) = ∫ K(y,x)N(x) dx
"""
marginalise(N::AbstractNormal{T}, K::AffineNormalKernel{T}) where {T} = # fallback
    Normal(mean(K)(mean(N)), stein(cov(N), mean(K), covp(K)))
marginalise(N::Normal{T}, K::AffineNormalKernel{T}) where {T} =
    Normal(mean(K)(mean(N)), stein(covp(N), mean(K), covp(K)))

"""
    invert(N::AbstractNorma, K::AffineNormalKernel)

Returns the inverted factorisation of the joint distirbution P(y,x) = N(x)*K(y, x) i.e

P(y,x) = Nout(y)*Kout(x,y)
"""
function invert(N::Normal{T}, K::AffineNormalKernel{T}) where {T} #fallback
    pred = mean(K)(mean(N))
    S, G, Σ = schur_red(cov(N), slope(mean(K)), cov(K))

    Nout = Normal(pred, S)
    Kout = NormalKernel(AffineMap(G, mean(N), pred), Σ)

    return Nout, Kout
end
function invert(N::Normal{T}, K::AffineNormalKernel{T}) where {T}
    pred = mean(K)(mean(N))
    S, G, Σ = schur_red(covp(N), slope(mean(K)), covp(K))

    Nout = Normal(pred, S)
    Kout = NormalKernel(AffineMap(G, mean(N), pred), Σ)

    return Nout, Kout
end

rand(RNG::AbstractRNG, K::NormalKernel, x::AbstractVector) = rand(RNG, condition(K, x))
rand(K::NormalKernel, x::AbstractVector) = rand(GLOBAL_RNG, K, x)
