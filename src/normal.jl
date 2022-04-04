
# define normal parametrisations
abstract type AbstractNormalParametrisation end
struct Usual <: AbstractNormalParametrisation  end
struct Information <: AbstractNormalParametrisation end

# define normal types
abstract type AbstractNormal{T<:Number,P<:AbstractNormalParametrisation}  <: AbstractDistribution end

struct Normal{T,U,V,P} <: AbstractNormal{T,P}
    μ::U
    Σ::V
    par::P
    function Normal(μ::AbstractVector,Σ::AbstractMatrix,par::AbstractNormalParametrisation)
        T = promote_type( eltype(μ), eltype(Σ) )
        μ = convert(AbstractVector{T}, μ)
        Σ = convert(AbstractMatrix{T}, Σ)
        new{T,typeof(μ),typeof(Σ),typeof(par)}(μ,Hermitian(Σ))
    end
end


Normal(μ::AbstractVector,Σ::AbstractMatrix) = Normal(μ,Σ,Usual())

eltype(n::AbstractNormal{T,P}) where {T,P} = T
parametrisation(n::AbstractNormal{T,P}) where {T,P} = P
dim(N::Normal) = length(N.μ)

mean(N::Normal{T,U,V,P}) where {T,U,V,P<:Usual} = N.μ
cov(N::Normal{T,U,V,P}) where {T,U,V,P<:Usual} = Matrix(N.Σ)
var(N::Normal{T,U,V,P}) where {T,U,V,P<:Usual} = real(diag(N.Σ))
std(N::Normal{T,U,V,P}) where {T,U,V,P<:Usual} = sqrt.(var(N))

mean(N::Normal{T,U,V,P}) where {T,U,V,P<:Information} = N.Σ \ N.μ
cov(N::Normal{T,U,V,P}) where {T,U,V,P<:Information} = Matrix(inv(N.Σ))
var(N::Normal{T,U,V,P}) where {T,U,V,P<:Information} = real(diag(inv(N.Σ)))
std(N::Normal{T,U,V,P}) where {T,U,V,P<:Information} = sqrt.(var(N))

residual(N::Normal{T,U,V,P},x) where {T,U,V,P<:Usual} = cholesky(N.Σ).L \ (x - N.μ)
logpdf(N::Normal{T,U,V,P},x) where {T<:Real,U,V,P<:Usual} = -dim(N)/2*log(2*π) - 1/2*logdet(N.Σ) -  norm_sqr( residual(N,x) )/2
logpdf(N::Normal{T,U,V,P},x) where {T<:Complex,U,V,P<:Usual} = -dim(N)*log(π) - real(logdet(N.Σ)) - norm_sqr( residual(N,x) )
entropy(N::Normal{T,U,V,P}) where {T<:Real,U,V,P<:Usual} = dim(N)/2.0*( log(2.0*π) + 1) + logdet(N.Σ)/2.0
entropy(N::Normal{T,U,V,P}) where {T<:Complex,U,V,P<:Usual} = dim(N)*( log(π) + 1) + real(logdet(N.Σ))

function kldivergence(N1::Normal{T,U,V,P},N2::Normal{T,U,V,P}) where {T<:Real,U,V,P}
    root_ratio = lsqrt(N2.Σ) \ lsqrt(N1.Σ)
    return 1/2*( norm_sqr(root_ratio) + norm_sqr(residual(N2,N1.μ)) - dim(N1) - 2.0*logdet(root_ratio) )
end

function kldivergence(N1::Normal{T,U,V,P},N2::Normal{T,U,V,P}) where {T<:Complex,U,V,P}
    root_ratio = lsqrt(N2.Σ) \ lsqrt(N1.Σ)
    return (norm_sqr(root_ratio) + norm_sqr(residual(N2,N1.μ)) - dim(N1) - 2.0*logabsdet(root_ratio)[1]) #logdet broken for complex Hermitian matrices (returns complex number  type)
end

rand(RNG::AbstractRNG, N::Normal{T,U,V,P}) where {T,U,V,P<:Usual} = n.μ + lsqrt(N.Σ)*randn(RNG,eltype(N),dim(N))
rand(N::Normal) = rand(GLOBAL_RNG,N)

