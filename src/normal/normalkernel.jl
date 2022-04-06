

abstract type AbstractNormalKernel{T<:Number}  <: AbstractMarkovKernel end

struct NormalKernel{T,U,V} <: AbstractNormalKernel{T}
    μ::U
    Σ::V
    function(μ,Σ)
        #outdim, indim = size(μ)
        new{eltype(μ),typeof(μ),typeof(Σ)}(μ,Σ,par)
    end
end

NormalKernel(μ,Σ) = Normalkernel(μ,Σ,Usual())

nout(K::NormalKernel) = nout(K.μ)
nin(K::NormalKernel) = nin(K.μ)

mean(K::NormalKernel) = K.μ
cov(K::NormalKernel) = K.Σ # callable conditional covariance
cov(K::NormalKernel{T,U,V}) where {T,U,V<:AbstractMatrix} = x -> K.Σ # constant coditionmal covariance

"""
condition(K::NormalKernel,x)

"""
condition(K::NormalKernel{T,U,V},x) where {T,U<:AbstractAffineMap,V<:AbstractMatrix} = Normal(K.μ(x), K.Σ)
(K::AbstractNormalKernel)(x) =  condition(K,x)

"""
compose(K2::NormalKernel,K1::NormalKernel)

Chapman--Kolmogorov:

K3(y∣x) = ∫K2(y∣z)K1(z∣x)dz

alternative:  K2*K1
"""
compose(K2::NormalKernel{T1,U1,V1},K1::NormalKernel{T2,U2,V2}) where {T1,U1<:AbstractAffineMap,V1<:AbstractMatrix,T2,U2<:AbstractAffineMap,V2<:AbstractMatrix} = NormalKernel( compose(K2.μ,K1.μ), slope(K2.μ)*K1.Σ*slope(K2.μ)' +  K2.Σ)
#*(K2::NormalKernel,K1::NormalKernel)  =  compose(K2,K1)

"""
marginalise(N::AbstractNormal,K::AbstractNormalKernel)

P(y) = ∫K(y∣x)N(x)dx

"""
marginalise(N::Normal{T1,U1,V1},K::NormalKernel{T2,U2,V2}) where {T1,U1<:AbstractVector,V1<:AbstractMatrix,T2,U2<:AbstractAffineMap,V2<:AbstractMatrix} = Normal( mean(K)(mean(N)), slope(K.μ)*N.Σ*slope(K.μ)' + K.Σ)
*(K::AbstractNormalKernel,N::AbstractNormal) = marginalise(N,K)

"""
reverse(N::AbstractNormal,K::AbstractNormalKernel)

reverses the factorisation  N(x)K(y∣x) to

( ∫K(y∣x)N(x)dx, N(x)K(y∣x) / ∫N(x)K(y∣x)dx )

"""
function reverse(N::Normal{T1,U1,V1},K::NormalKernel{T2,U2,V2})  where {T1,U1<:AbstractVector,V1<:AbstractMatrix,T2,U2<:AbstractAffineMap,V2<:AbstractMatrix}

    pred = mean(K)(mean(N))
    S = Hermitian(slope(K.μ)*N.Σ*slope(K.μ)' + K.Σ)


    G = N.Σ*slope(K.μ)' / S
    L = (I - K*slope(K.μ))
    Π = Hermitian(L*N.Σ*L' + G*K.Σ*G')

    Nout = Normal(pred, S)
    Kout = NormalKernel( AffineCorrector(G,N.μ,pred), Π )

    return Nout, Kout
end