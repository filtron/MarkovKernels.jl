# define normal types
abstract type AbstractNormal{T<:Number} <: AbstractDistribution end

eltype(N::AbstractNormal{T}) where {T} = T

# hm.. should make Information form / Precision parametrisations here as well?
struct Normal{T,U,V} <: AbstractNormal{T}
    μ::U
    Σ::V
    function Normal(μ::AbstractVector, Σ::AbstractMatrix)
        T = promote_type(eltype(μ), eltype(Σ))
        new{T,typeof(μ),typeof(Σ)}(
             μ,
             Σ,
        )
    end
end

similar(N::Normal) = Normal(similar(N.μ), similar(N.Σ))
==(N1::Normal, N2::Normal) = N1.μ == N2.μ && N1.Σ == N2.Σ

"""
    dim(N::AbstractNormal)

Returns the dimension of the random vector represented by the normal distribution N.
"""
dim(N::Normal) = length(N.μ)

mean(N::Normal) = N.μ
cov(N::Normal) = Hermitian(Matrix(N.Σ))
var(N::Normal) = real(diag(N.Σ))
std(N::Normal) = sqrt.(var(N))

residual(N::Normal, x) = cholesky(N.Σ).L \ (x .- N.μ)

logpdf(N::Normal{T,U,V}, x) where {T<:Real,U,V} =
    -dim(N) / 2 * log(2 * π) - 1 / 2 * logdet(N.Σ) - norm_sqr(residual(N, x)) / 2
logpdf(N::Normal{T,U,V}, x) where {T<:Complex,U,V} =
    -dim(N) * log(π) - logdet(N.Σ) - norm_sqr(residual(N, x))

entropy(N::Normal{T,U,V}) where {T<:Real,U,V} =
    dim(N) / 2.0 * (log(2.0 * π) + 1) + logdet(N.Σ) / 2.0
entropy(N::Normal{T,U,V}) where {T<:Complex,U,V} = dim(N) * (log(π) + 1) + logdet(N.Σ)

function kldivergence(N1::Normal{T,U,V}, N2::Normal{T,U,V}) where {T<:Real,U,V}
    root_ratio = lsqrt(N2.Σ) \ lsqrt(N1.Σ)
    return 1 / 2 * (
        norm_sqr(root_ratio) + norm_sqr(residual(N2, N1.μ)) - dim(N1) -
        2.0 * logdet(root_ratio)
    )
end

function kldivergence(N1::Normal{T,U,V}, N2::Normal{T,U,V}) where {T<:Complex,U,V}
    root_ratio = lsqrt(N2.Σ) \ lsqrt(N1.Σ)
    return norm_sqr(root_ratio) + norm_sqr(residual(N2, N1.μ)) - dim(N1) -
           2.0 * real(logdet(root_ratio))
end

rand(RNG::AbstractRNG, N::Normal{T,U,V}) where {T,U,V} =
    N.μ + lsqrt(N.Σ) * randn(RNG, eltype(N), dim(N))
rand(N::Normal) = rand(GLOBAL_RNG, N)

# Dirac
struct Dirac{T,U} <: AbstractNormal{T}
    μ::U
    Dirac(μ::AbstractVector) = new{eltype(μ),typeof(μ)}(μ)
end

similar(D::Dirac) = Dirac(similar(D.μ))

dim(D::Dirac) = length(D.μ)
mean(D::Dirac) = D.μ
cov(D::Dirac) = zeros(eltype(D), dim(D), dim(D))
var(D::Dirac) = zeros(T, dim(D))
std(D::Dirac) = zeros(T, dim(D))

rand(RNG::AbstractRNG, D::Dirac) = mean(D)
rand(D::Dirac) = rand(GLOBAL_RNG, D)
