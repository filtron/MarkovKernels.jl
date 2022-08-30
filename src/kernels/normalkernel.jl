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
struct NormalKernel{T,U<:AbstractConditionalMean,V<:AbstractMatrix} <:
       AbstractNormalKernel{T}
    μ::U
    Σ::V
    function NormalKernel(μ, Σ)
        new{eltype(μ),typeof(μ),typeof(Σ)}(μ, Σ)
    end
end

"""
    NormalKernel(Φ::AbstractMatrix, Σ::AbstractMatrix)

Creates a Normal kernel with linear conditional mean of slope Φ and covariance matrix Σ.
"""
NormalKernel(Φ::AbstractMatrix, Σ::AbstractMatrix) = NormalKernel(AffineMap(Φ), Σ)

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

# Dirac kernel
struct DiracKernel{T,U<:AbstractConditionalMean} <: AbstractNormalKernel{T}
    μ::U
    DiracKernel(μ) = new{eltype(μ),typeof(μ)}(μ)
end

DiracKernel(Φ::AbstractMatrix) = DiracKernel(AffineMap(Φ))
DiracKernel(Φ::AbstractMatrix, b::AbstractVector) = DiracKernel(AffineMap(Φ, b))

mean(K::NormalKernel) = K.μ
mean(K::DiracKernel) = K.μ

cov(K::NormalKernel{T,U,V}) where {T,U,V<:AbstractMatrix} = K.Σ
cov(K::DiracKernel{T}) where {T} = zeros(T, nout(mean(K)), nout(mean(K)))

condition(
    K::NormalKernel{T,U,V},
    x,
) where {T,U<:AbstractConditionalMean,V<:AbstractMatrix} = Normal(mean(K)(x), cov(K))
condition(K::DiracKernel, x) = Dirac(mean(K)(x))

compose(
    K2::NormalKernel{T,U,V},
    K1::NormalKernel{T,U,V},
) where {T,U<:AbstractAffineMap,V<:AbstractMatrix} =
    NormalKernel(compose(mean(K2), mean(K1)), stein(cov(K1), mean(K2), cov(K2)))
compose(K2::DiracKernel{T,U}, K1::DiracKernel{T,U}) where {T,U<:AbstractAffineMap} =
    DiracKernel(compose(mean(K2), mean(K1)))
compose(
    K2::DiracKernel{T,U},
    K1::NormalKernel{T,U,V},
) where {T,U<:AbstractAffineMap,V<:AbstractMatrix} =
    NormalKernel(compose(mean(K2), mean(K1)), stein(cov(K1), mean(K2), cov(K2)))
compose(
    K2::NormalKernel{T,U,V},
    K1::DiracKernel{T,U},
) where {T,U<:AbstractAffineMap,V<:AbstractMatrix} =
    NormalKernel(compose(mean(K2), mean(K1)), cov(K2))

*(K2::AbstractNormalKernel, K1::AbstractNormalKernel) = compose(K2, K1)

marginalise(
    N::Normal{T,U,V},
    K::NormalKernel{T,S,V},
) where {T,U,S<:AbstractAffineMap,V<:AbstractMatrix} =
    Normal(mean(K)(mean(N)), stein(cov(N), mean(K), cov(K)))
marginalise(
    N::Normal{T,U,V},
    K::DiracKernel{T,S},
) where {T,U,S<:AbstractAffineMap,V<:AbstractMatrix} =
    Normal(mean(K)(mean(N)), stein(cov(N), mean(K), cov(K))) # not the smartest way ...

function invert(
    N::Normal{T,U,V},
    K::NormalKernel{T,M,W},
) where {T,U,V<:AbstractMatrix,M<:AbstractAffineMap,W<:AbstractMatrix}
    pred = mean(K)(mean(N))

    Π = cov(N)
    C = slope(mean(K))
    R = cov(K)

    S, G, Σ = schur_red(Π, C, R)

    Nout = Normal(pred, S)

    corrector = AffineMap(G, mean(N), pred)
    Kout = NormalKernel(corrector, Σ)

    return Nout, Kout
end

rand(RNG::AbstractRNG, K::NormalKernel, x::AbstractVector) = rand(RNG, condition(K, x))
rand(K::NormalKernel, x::AbstractVector) = rand(GLOBAL_RNG, K, x)
rand(RNG::AbstractRNG, K::DiracKernel, x::AbstractVector) = mean(condition(K, x))
rand(K::DiracKernel, x::AbstractVector) = rand(GLOBAL_RNG, K, x)
