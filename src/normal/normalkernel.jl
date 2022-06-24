

abstract type AbstractNormalKernel{T<:Number}  <: AbstractMarkovKernel end

# NormalKernel for Homoscedastic noise
struct NormalKernel{T,U<:AbstractConditionalMean,V<:AbstractMatrix} <: AbstractNormalKernel{T}
    μ::U
    Σ::V
    function NormalKernel(μ,Σ)
        #should convert μ and Σ to common eltype here
        new{eltype(μ),typeof(μ),typeof(Σ)}(μ,Σ)
    end
end

NormalKernel(Φ::AbstractMatrix,Σ::AbstractMatrix) = NormalKernel( LinearMap(Φ), Hermitian(Σ) )
NormalKernel(Φ::AbstractMatrix,b::AbstractVector,Σ::AbstractMatrix) = NormalKernel( AffineMap(Φ,b),Σ )

mean(K::NormalKernel) = K.μ
cov(K::NormalKernel) = K.Σ # callable conditional covariance
cov(K::NormalKernel{T,U,V}) where {T,U,V<:AbstractMatrix} = x -> K.Σ # constant coditionmal covariance

"""
condition(K::NormalKernel,x)

"""
condition(K::NormalKernel{T,U,V},x) where {T,U<:AbstractAffineMap,V<:AbstractMatrix} = Normal(K.μ(x), K.Σ)

"""
compose(K2::NormalKernel,K1::NormalKernel)

Chapman--Kolmogorov:

K3(y∣x) = ∫K2(y∣z)K1(z∣x)dz

alternative:  K2*K1
"""
compose(K2::NormalKernel{T,U,V},K1::NormalKernel{T,U,V}) where {T,U<:AbstractAffineMap,V<:AbstractMatrix} = NormalKernel( compose(K2.μ,K1.μ), stein(K1.Σ,K2.μ,K2.Σ) )
*(K2::NormalKernel,K1::NormalKernel)  =  compose(K2,K1)

"""
marginalise(N::AbstractNormal,K::AbstractNormalKernel)

P(y) = ∫K(y∣x)N(x)dx

"""
marginalise(N::Normal{T,U,V},K::NormalKernel{T,U,V}) where {T,U<:AbstractVector,V<:AbstractMatrix} = Normal( mean(K)(mean(N)), stein(N.Σ,K.μ,K.Σ) )
*(K::AbstractNormalKernel,N::AbstractNormal) = marginalise(N,K)

"""
invert(N::AbstractNormal,K::AbstractNormalKernel)

inverts the factorisation  N(x)K(y∣x) to

( ∫K(y∣x)N(x)dx, N(x)K(y∣x) / ∫N(x)K(y∣x)dx )

"""
function invert(N::Normal{T,U,V},K::NormalKernel{T,M,V})  where {T,U,V<:AbstractMatrix,M<:AbstractAffineMap}

    pred = mean(K)(mean(N))
    S = stein(N.Σ,K.μ,K.Σ)
    Nout = Normal(pred, S)

    G = N.Σ*slope(K.μ)' / S
    corrector = AffineCorrector(G,N.μ,pred)
    L = (I - K*slope(K.μ))
    Π = Hermitian(L*N.Σ*L' + G*K.Σ*G')
    Kout = NormalKernel( corrector, Π )

    return Nout, Kout
end