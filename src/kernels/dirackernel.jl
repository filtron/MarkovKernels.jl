abstract type AbstractDiracKernel{T<:Number} <: AbstractMarkovKernel end

eltype(::AbstractDiracKernel{T}) where {T} = T

AbstractDiracKernel{T}(K::AbstractDiracKernel{T}) where {T} = K
convert(::Type{T}, K::T) where {T<:AbstractDiracKernel} = K
convert(::Type{T}, K::AbstractDiracKernel) where {T<:AbstractDiracKernel} = T(K)::T

==(K1::T, K2::T) where {T<:AbstractDiracKernel} =
    all(f -> getfield(K1, f) == getfield(K2, f), 1:nfields(K1))


"""
    DiracKernel

Type for representing Dirac kernels K(y,x) = δ(y - μ(x))
"""
struct DiracKernel{T,U} <: AbstractDiracKernel{T}
    μ::U
    DiracKernel{T}(μ) where {T} = new{T,typeof(μ)}(μ)
end

DiracKernel(F::AbstractAffineMap) = DiracKernel{eltype(F)}(F)
DiracKernel(Φ::AbstractMatrix) = DiracKernel(LinearMap(Φ))
DiracKernel(Φ::AbstractMatrix, b::AbstractVector) = DiracKernel(AffineMap(Φ, b))
DiracKernel(Φ::AbstractMatrix, b::AbstractVector, c::AbstractVector) =
    DiracKernel(AffineCorrector(Φ, b, c))

const AffineDiracKernel{T} =
    DiracKernel{T,<:AbstractAffineMap}

DiracKernel{T}(K::DiracKernel{U,V}) where {T,U,V<:AbstractAffineMap} =
    T <: Real && U <: Real || T <: Complex && U <: Complex ?
    DiracKernel(convert(AbstractAffineMap{T}, K.μ)) :
    error("T and U must both be complex or both be real")

AbstractDiracKernel{T}(K::DiracKernel) where {T} = DiracKernel{T}(K)

mean(K::DiracKernel) = K.μ

cov(K::DiracKernel{T}) where {T} = x->  Diagonal(zeros(T, nout(mean(K))))

condition(K::DiracKernel, x) = Dirac(mean(K)(x))

compose(K2::DiracKernel{T,U}, K1::DiracKernel{T,U}) where {T,U<:AbstractAffineMap} =
    DiracKernel(compose(mean(K2), mean(K1)))

marginalise(
    N::Normal{T,U,V},
    K::DiracKernel{T,S},
) where {T,U,S<:AbstractAffineMap,V<:AbstractMatrix} =
    Normal(mean(K)(mean(N)), stein(covp(N), mean(K)))

function invert(
    N::Normal{T,U,V},
    K::DiracKernel{T,M},
) where {T,U,V<:AbstractMatrix,M<:AbstractAffineMap}
    pred = mean(K)(mean(N))

    Π = cov(N)
    C = slope(mean(K))

    S, G, Σ = schur_red(Π, C)

    Nout = Normal(pred, S)

    corrector = AffineCorrector(G, mean(N), pred)
    Kout = NormalKernel(corrector, Σ)

    return Nout, Kout
end

rand(RNG::AbstractRNG, K::DiracKernel, x::AbstractVector) = mean(condition(K, x))
rand(K::DiracKernel, x::AbstractVector) = rand(GLOBAL_RNG, K, x)
