


# define normal types
abstract type AbstractNormal{T<:Number}  <: AbstractDistribution end

eltype(n::AbstractNormal{T}) where {T} = T

struct Normal{T,U,V} <: AbstractNormal{T}
    μ::U
    Σ::V
    function Normal(μ::AbstractVector,Σ::AbstractMatrix)
        T = promote_type( eltype(μ), eltype(Σ) )
        new{T,typeof(μ),typeof(Σ)}(μ,Σ)
    end
end

dim(N::Normal) = length(N.μ)

mean(N::Normal) = N.μ
cov(N::Normal) = Hermitian(Matrix(N.Σ))
var(N::Normal) = real(diag(N.Σ))
std(N::Normal) = sqrt.(var(N))

residual(N::Normal,x) = cholesky( Hermitian(N.Σ) ).L \ (x - N.μ)

logpdf(N::Normal{T,U,V},x) where {T<:Real,U,V} = -dim(N)/2*log(2*π) - 1/2*logdet( Hermitian(N.Σ) ) -  norm_sqr( residual(N,x) )/2
logpdf(N::Normal{T,U,V},x) where {T<:Complex,U,V} = -dim(N)*log(π) - logdet( Hermitian(N.Σ) ) - norm_sqr( residual(N,x) )

entropy(N::Normal{T,U,V}) where {T<:Real,U,V} = dim(N)/2.0*( log(2.0*π) + 1) + logdet( Hermitian(N.Σ) )/2.0
entropy(N::Normal{T,U,V,}) where {T<:Complex,U,V} = dim(N)*( log(π) + 1) + logdet( Hermitian(N.Σ) )

function kldivergence(N1::Normal{T,U,V},N2::Normal{T,U,V}) where {T<:Real,U,V}
    root_ratio = lsqrt( Hermitian(N2.Σ) ) \ lsqrt( Hermitian(N1.Σ) )
    return 1/2*( norm_sqr(root_ratio) + norm_sqr(residual(N2,N1.μ)) - dim(N1) - 2.0*logdet(root_ratio) )
end

function kldivergence(N1::Normal{T,U,V},N2::Normal{T,U,V}) where {T<:Complex,U,V}
    root_ratio = lsqrt( Hermitian(N2.Σ) ) \ lsqrt( Hermitian(N1.Σ) )
    return norm_sqr(root_ratio) + norm_sqr(residual(N2,N1.μ)) - dim(N1) - 2.0*logabsdet(root_ratio)[1]
end

rand(RNG::AbstractRNG, N::Normal{T,U,V}) where {T,U,V} = N.μ + lsqrt( Hermitian(N.Σ) )*randn(RNG,eltype(N),dim(N))
rand(N::Normal) = rand(GLOBAL_RNG,N)

