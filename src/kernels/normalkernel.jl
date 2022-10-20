"""
    AbstractNormalKernel{T<:Number}

Abstract type for representing Normal kernels taking values in T.
"""
abstract type AbstractNormalKernel{T<:Number} <: AbstractMarkovKernel end

eltype(::AbstractNormalKernel{T}) where {T} = T

AbstractNormalKernel{T}(K::AbstractNormalKernel{T}) where {T} = K
convert(::Type{T}, K::T) where {T<:AbstractNormalKernel} = K
convert(::Type{T}, K::AbstractNormalKernel) where {T<:AbstractNormalKernel} = T(K)::T

==(K1::T, K2::T) where {T<:AbstractNormalKernel} =
    all(f -> getfield(K1, f) == getfield(K2, f), 1:nfields(K1))

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

for c in (:AbstractMatrix, :Factorization)
    @eval function NormalKernel(F::AbstractAffineMap, Σ::$c)
        T = promote_type(eltype(F), eltype(Σ))
        return NormalKernel{T}(
            convert(AbstractAffineMap{T}, F),
            symmetrise(convert($c{T}, Σ)),
        )
    end
    @eval NormalKernel{T}(K::NormalKernel{U,V,W}) where {T,U,V<:AbstractAffineMap,W<:$c} =
        T <: Real && U <: Real || T <: Complex && U <: Complex ?
        NormalKernel(convert(AbstractAffineMap{T}, K.μ), convert($c{T}, K.Σ)) :
        error("T and U must both be complex or both be real")
end

for c in (:Diagonal, :UniformScaling)
    @eval function NormalKernel(F::AbstractAffineMap, Σ::$c)
        T = promote_type(eltype(F), eltype(Σ))
        return NormalKernel{T}(
            convert(AbstractAffineMap{T}, F),
            symmetrise(convert($c{real(T)}, Σ)),
        )
    end
    @eval NormalKernel{T}(K::NormalKernel{U,V,W}) where {T,U,V<:AbstractAffineMap,W<:$c} =
        T <: Real && U <: Real || T <: Complex && U <: Complex ?
        NormalKernel(convert(AbstractAffineMap{T}, K.μ), convert($c{real(T)}, K.Σ)) :
        error("T and U must both be complex or both be real")
end

AbstractNormalKernel{T}(K::NormalKernel) where {T} = NormalKernel{T}(K)

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
