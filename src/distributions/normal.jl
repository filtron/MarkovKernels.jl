"""
    AbstractNormal{T<:Number}

Abstract type for representing Normal distributed random vectors taking values in T.
"""
abstract type AbstractNormal{T} <: AbstractDistribution{T} end

"""
    Normal{T,U,V}

Standard mean vector / covariance matrix parametrisation of the normal distribution with element type T.
"""
struct Normal{T,U,V} <: AbstractNormal{T}
    μ::U
    Σ::V
end

function Normal{T}(μ::AbstractVector, Σ, ::IsPSD) where {T}
    μ = convert(AbstractVector{T}, μ)
    Σ = convert_psd_eltype(T, Σ)
    return Normal{T,typeof(μ),typeof(Σ)}(μ, Σ)
end

function Normal{T}(μ::Number, Σ, ::IsPSD) where {T}
    μ = convert(T, μ)
    Σ = convert_psd_eltype(T, Σ)
    return Normal{T,typeof(μ),typeof(Σ)}(μ, Σ)
end

"""
    Normal(μ, Σ)

Creates a Normal distribution with mean μ and covariance Σ.
"""
function Normal(μ, Σ)
    T = promote_type(eltype(μ), eltype(Σ))
    return Normal{T}(μ, Σ, psdcheck(Σ))
end

const UvNormal{T,V} = Union{Normal{V,V,V},Normal{T,T,V}} where {V<:Real,T<:Complex{V}}

typeof_sample(N::Normal) = typeof(mean(N))

function Base.copy!(Ndst::A, Nsrc::A) where {T,U,V<:Cholesky,A<:Normal{T,U,V}}
    copy!(mean(Ndst), mean(Nsrc))
    if covp(Ndst).uplo == covp(Nsrc).uplo
        copy!(covp(Ndst).factors, covp(Nsrc).factors)
    else
        copy!(covp(Ndst).factors, adjoint(covp(Nsrc).factors))
    end
    return Ndst
end

function Base.similar(N::Normal{T,U,V}) where {T,U,V<:Cholesky}
    C = covp(N)
    return Normal(similar(mean(N)), Cholesky(similar(C.factors), C.uplo, C.info))
end

function Base.isapprox(
    N1::Normal{T1,V1,<:Cholesky},
    N2::Normal{T2,V2,<:Cholesky},
    kwargs...,
) where {T1,V1,T2,V2}
    return isapprox(mean(N1), mean(N2); kwargs...) &&
           isapprox(rsqrt(covp(N1)), rsqrt(covp(N1)); kwargs...)
end

"""
    dim(N::AbstractNormal)

Returns the dimension of the Normal distribution N.
"""
dim(N::Normal) = length(N.μ)

"""
    mean(N::AbstractNormal)

Computes the mean vector of the Normal distribution N.
"""
mean(N::Normal) = N.μ
"""
    covp(N::AbstractNormal)

Returns the internal representation of the covariance matrix of the Normal distribution N.
For computing the actual covariance matrix, use cov.
"""
covp(N::Normal) = N.Σ

"""
    cov(N::AbstractNormal)

Computes the covariance matrix of the Normal distribution N.
"""
cov(N::Normal) = AbstractMatrix(covp(N))
cov(N::Normal{T,U,V}) where {T,U,V<:AbstractMatrix} = covp(N)
cov(N::UvNormal) = covp(N)

"""
    var(N::AbstractNormal)
Computes the vector of marginal variances of the Normal distribution N.
"""
var(N::AbstractNormal) = real(diag(covp(N)))
var(N::Normal{T,U,V}) where {T,U,V<:Cholesky} = map(norm_sqr, eachrow(lsqrt(covp(N))))
var(N::UvNormal) = cov(N)

"""
    std(N::AbstractNormal)
Computes the vector of marginal standard deviations of the Normal distribution N.
"""
std(N::AbstractNormal) = sqrt.(var(N))

Normal{T}(N::Normal{A,<:AbstractVector}) where {T,A} =
    Normal(convert(AbstractVector{T}, mean(N)), convert_psd_eltype(T, covp(N)))
Normal{T}(N::Normal{A,<:Number}) where {T,A} =
    Normal(convert(T, mean(N)), convert_psd_eltype(T, covp(N)))
AbstractDistribution{T}(N::AbstractNormal) where {T} = AbstractNormal{T}(N)
AbstractNormal{T}(N::AbstractNormal{T}) where {T} = N
AbstractNormal{T}(N::Normal) where {T} = Normal{T}(N)

"""
    residual(N::AbstractNormal, x::AbstractVector)

Computes the whitened residual associated with the Normal distribution N and observed vector x.
"""
residual(N::AbstractNormal, x) = lsqrt(covp(N)) \ (x - mean(N))

_nscale(T::Type{<:Real}) = T(0.5)
_nscale(T::Type{<:Complex}) = one(real(T))

_piconst(T::Type{<:Real}) = T(2π)
_piconst(T::Type{<:Complex}) = real(T)(π)

"""
    logpdf(N::AbstractNormal,x)

Computes the logarithm of the probability density function of the Normal distribution N evaluated at x.
"""
logpdf(N::AbstractNormal{T}, x) where {T} =
    -_nscale(T) *
    (dim(N) * log(_piconst(T)) + real(logdet(covp(N))) + norm_sqr(residual(N, x)))

"""
    entropy(N::AbstractNormal)

Computes the entropy of the Normal distribution N.
"""
entropy(N::AbstractNormal{T}) where {T} =
    _nscale(T) * (dim(N) * (log(_piconst(T)) + one(real(T))) + real(logdet(covp(N))))

"""
    kldivergence(N1::AbstractNormal, N2::AbstractNormal)

Computes the Kullback-Leibler divergence between the Normal distributions N1 and N2.
"""
function kldivergence(N1::AbstractNormal{T}, N2::AbstractNormal{T}) where {T<:Number}
    root_ratio = lsqrt(covp(N2)) \ lsqrt(covp(N1))
    _nscale(T) * (
        norm_sqr(root_ratio) + norm_sqr(residual(N2, mean(N1))) - dim(N1) -
        real(T)(2) * real(logdet(root_ratio))
    )
end

"""
    rand(RNG::AbstractRNG, N::AbstractNormal)

Computes a random vector distributed according to the Normal distribution N
using the random number generator RNG.
"""
rand(rng::AbstractRNG, N::AbstractNormal) =
    mean(N) + lsqrt(covp(N)) * randn(rng, eltype(N), dim(N))

rand(rng::AbstractRNG, N::UvNormal) = mean(N) + lsqrt(covp(N)) * randn(rng, eltype(N))

function Base.show(io::IO, N::Normal)
    println(io, summary(N))
    print(io, "μ = ")
    show(io, (N.μ))
    print(io, "\nΣ = ")
    show(io, N.Σ)
end

function Base.show(io::IO, N::UvNormal)
    println(io, summary(N))
    print(io, "μ = ")
    show(io, (N.μ))
    print(io, "\nσ² = ")
    show(io, N.Σ)
end
