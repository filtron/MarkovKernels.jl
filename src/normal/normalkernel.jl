

abstract type AbstractNormalKernel{T<:Number}  <: AbstractMarkovKernel end

eltype(K::AbstractNormalKernel{T}) where T = T

(K::AbstractNormalKernel)(x) = condition(K,x)

# NormalKernel for Homoscedastic noise
struct NormalKernel{T,U<:AbstractConditionalMean,V<:AbstractMatrix} <: AbstractNormalKernel{T}
    μ::U
    Σ::V
    function NormalKernel(μ,Σ)
        new{eltype(μ),typeof(μ),typeof(Σ)}(μ,Σ)
    end
end

NormalKernel(Φ::AbstractMatrix,Σ::AbstractMatrix) = NormalKernel( AffineMap(Φ), Hermitian(Σ) )
NormalKernel(Φ::AbstractMatrix,b::AbstractVector,Σ::AbstractMatrix) = NormalKernel( AffineMap(Φ,b),Σ )

mean(K::NormalKernel) = K.μ
cov(K::NormalKernel{T,U,V}) where {T,U,V<:AbstractMatrix} = K.Σ


condition(K::NormalKernel{T,U,V},x) where {T,U<:AbstractConditionalMean,V<:AbstractMatrix} = Normal(K.μ(x), K.Σ)

compose(K2::NormalKernel{T,U,V},K1::NormalKernel{T,U,V}) where {T,U<:AbstractAffineMap,V<:AbstractMatrix} = NormalKernel( compose(K2.μ,K1.μ), stein(K1.Σ,K2.μ,K2.Σ) )
*(K2::NormalKernel,K1::NormalKernel)  =  compose(K2,K1)

marginalise(N::Normal{T,U,V},K::NormalKernel{T,S,V}) where {T,U,S<:AbstractAffineMap,V<:AbstractMatrix} = Normal( mean(K)(mean(N)), stein(N.Σ,K.μ,K.Σ) )

function invert(N::Normal{T,U,V},K::NormalKernel{T,M,V})  where {T,U,V<:AbstractMatrix,M<:AbstractAffineMap}

    pred = mean(K)(mean(N))

    Π = cov(N)
    C = slope(mean(K))
    R = cov(K)

    S, G, Σ = schur_red(Π,C,R)

    Nout = Normal(pred, S)

    corrector = AffineMap(G,mean(N),pred)
    Kout = NormalKernel( corrector, Σ )

    return Nout, Kout
end