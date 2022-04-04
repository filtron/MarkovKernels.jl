
# define normal kernel type
abstract type AbstractNormalKernel{T<:Number,P<:AbstractNormalParametrisation}  <: AbstractMarkovKernel end

struct NormalKernel{T,U,V,P} <: AbstractNormalKernel{T,P}
    μ::U
    Σ::V
end

size(K::NormalKernel) = size(K.μ)
size(K::NormalKernel,i) = 1 <= i <= 2 ? size(K)[i] : 1

outdim(K::NormalKernel) = size(K)[1]
indim(K::NormalKernel) = size(K)[2]

mean(K::NormalKernel) = K.μ
cov(K::NormalKernel) = K.Σ

"""
condition(K::NormalKernel,x)

"""
condition(K::NormalKernel,x) = Normal( K.μ(x), K.Σ(x))
(K::NormalKernel)(x) =  condition(K,x)

"""
compose(K2::NormalKernel,K1::NormalKernel)

Chapman--Kolmogorov:

K3(y∣x) = ∫K2(y∣z)K1(z∣x)dz

alternative:  K2*K1

"""
# implement Chapman-Kolmogorov (FIX THIS)
#compose(K2::NormalKernel,K1::NormalKernel) = NormalKernel( K2.μ, K2.Σ )
#*(K2::NormalKernel,K1::NormalKernel)  =  compose(K2,K1)

"""
marginalise(N::AbstractNormal,K::AbstractNormalKernel)

P(y) = ∫K(y∣x)N(x)dx

"""
#marginalise(N::AbstractNormal,K::AbstractNormalKernel) = Normal()
#*(K::NormalKernel,N::Normal) = marginalise(N,K)


"""
reverse(N::AbstractNormal,K::AbstractNormalKernel)

reverses the factorisation  N(x)K(y∣x) to

( ∫K(y∣x)N(x)dx, N(x)K(y∣x) / ∫N(x)K(y∣x)dx )

"""
#=
function reverse(Nk::NormalKernel,N::Normal)

    μy = mean(Nk)(mean(N))

    Σ, K, S = schur_reduce(Π::AbstractMatrix,C::AbstractMatrix,R::AbstractMatrix)

    Nout = Normal(μy,Hermitian(S))
    Nkout = NormalKernel(  AffineMap(K,mean(N) - K*μy ), Σ )

end
=#