abstract type AbstractDiracKernel{T<:Number} <: AbstractMarkovKernel end

eltype(::AbstractDiracKernel{T}) where {T} = T

struct DiracKernel{T,U} <: AbstractDiracKernel{T}
    μ::U
    DiracKernel(μ) = new{eltype(μ),typeof(μ)}(μ)
end

DiracKernel(Φ::AbstractMatrix) = DiracKernel(LinearMap(Φ))
DiracKernel(Φ::AbstractMatrix, b::AbstractVector) = DiracKernel(AffineMap(Φ, b))

mean(K::DiracKernel) = K.μ
cov(K::DiracKernel{T}) where {T} = zeros(T, nout(mean(K)), nout(mean(K)))
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
