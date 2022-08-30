abstract type AbstractDiracKernel{T<:Number} <: AbstractMarkovKernel end

eltype(::AbstractDiracKernel{T}) where {T} = T

struct DiracKernel{T,U<:AbstractConditionalMean} <: AbstractDiracKernel{T}
    μ::U
    DiracKernel(μ) = new{eltype(μ),typeof(μ)}(μ)
end

DiracKernel(Φ::AbstractMatrix) = DiracKernel(AffineMap(Φ))
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
    Normal(mean(K)(mean(N)), stein(cov(N), mean(K), cov(K))) # not the smartest way ...


rand(RNG::AbstractRNG, K::DiracKernel, x::AbstractVector) = mean(condition(K, x))
rand(K::DiracKernel, x::AbstractVector) = rand(GLOBAL_RNG, K, x)