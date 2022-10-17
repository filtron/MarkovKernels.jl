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
    NormalKernel{T}(μ, Σ) where {T} = new{T,typeof(μ),typeof(Σ)}(μ, Σ) 
end

NormalKernel(F::AbstractAffineMap, Σ) = NormalKernel{eltype(F)}(F, Σ) 
NormalKernel(Φ::AbstractMatrix, Σ) = NormalKernel(LinearMap(Φ), Σ)
NormalKernel(Φ::AbstractMatrix, b::AbstractVector, Σ) = NormalKernel(AffineMap(Φ, b), Σ)
NormalKernel(Φ::AbstractMatrix, b::AbstractVector, c::AbstractVector, Σ) =
    NormalKernel(AffineCorrector(Φ, b, c), Σ)

const AffineNormalKernel{T} =
NormalKernel{T,<:AbstractAffineMap,<:Union{UniformScaling,Factorization,AbstractMatrix}}

for c in (:AbstractMatrix, :UniformScaling, :Factorization)
    @eval function NormalKernel(F::AbstractAffineMap, Σ::$c) 
        T = promote_type(eltype(F), eltype(Σ)) 
        F = convert(AbstractAffineMap{T}, F) 
        Σ = convert($c{T}, Σ) 
        return NormalKernel{T}(F, symmetrise(Σ))
    end
end

"""
    covp(K::NormalKernel)

Returns the internal representation of the conditonal covariance matrix of the Normal kernel K.
For computing the actual conditional covariance matrix, use cov.
"""
covp(K::NormalKernel) = K.Σ

mean(K::NormalKernel) = K.μ
cov(K::NormalKernel) = K.Σ
cov(K::AffineNormalKernel) = x -> K.Σ

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
marginalise(N::AbstractNormal{T}, K::AffineNormalKernel{T}) where {T} =
    Normal(mean(K)(mean(N)), stein(covp(N), mean(K), covp(K)))

"""
    invert(N::AbstractNorma, K::AffineNormalKernel)

Returns the inverted factorisation of the joint distirbution P(y,x) = N(x)*K(y, x) i.e

P(y,x) = Nout(y)*Kout(x,y)
"""
function invert(N::AbstractNormal{T}, K::AffineNormalKernel{T}) where {T}
    pred = mean(K)(mean(N))
    S, G, Σ = schur_red(covp(N), mean(K), covp(K))

    Nout = Normal(pred, S)
    Kout = NormalKernel(AffineCorrector(G, mean(N), pred), Σ)

    return Nout, Kout
end

rand(RNG::AbstractRNG, K::AbstractNormalKernel, x::AbstractVector) =
    rand(RNG, condition(K, x))
rand(K::AbstractNormalKernel, x::AbstractVector) = rand(GLOBAL_RNG, K, x)
