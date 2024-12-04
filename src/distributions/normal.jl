"""
    AbstractNormal{ST}

Abstract type for representing Normal distributed random vectors taking values in ST.
"""
abstract type AbstractNormal{ST} <: AbstractDistribution{ST} end

"""
    Normal{ST,U,V}

Standard mean vector / covariance matrix parametrization of the Normal distribution with sample type ST.
"""
struct Normal{ST,U,V} <: AbstractNormal{ST}
    μ::U
    Σ::V
end

Normal{ST}(μ::AbstractVector, Σ, ::IsPSD) where {ST<:AbstractVector} =
    Normal{ST,typeof(μ),typeof(Σ)}(μ, Σ)
Normal{ST}(μ::Number, Σ, ::IsPSD) where {ST<:Number} = Normal{ST,typeof(μ),typeof(Σ)}(μ, Σ)

"""
    Normal(μ, Σ)

Creates a Normal distribution with mean μ and covariance Σ.
"""
function Normal(μ::AbstractVector, Σ)
    T = promote_type(eltype(μ), eltype(Σ))
    T = float(T)
    ST = Base.promote_op(convert, Type{AbstractVector{T}}, typeof(μ))
    return Normal{ST}(μ, Σ, psdcheck(Σ))
end

function Normal(μ::Number, Σ)
    T = promote_type(eltype(μ), eltype(Σ))
    ST = float(T)
    return Normal{ST}(μ, Σ, psdcheck(Σ))
end

# this needs to change to allow for heterogneous eltype in fields / sample_type
const UvNormal{T,V} = Union{Normal{V,V,V},Normal{T,T,V}} where {V<:Real,T<:Complex{V}}

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
var(N::Normal{T,U,V}) where {T,U,V<:Cholesky} = map(norm_sqr, eachcol(rsqrt(covp(N))))
var(N::UvNormal) = cov(N)

"""
    std(N::AbstractNormal)
Computes the vector of marginal standard deviations of the Normal distribution N.
"""
std(N::AbstractNormal) = sqrt.(var(N))

function sample_type(N::AbstractNormal)
    T = promote_type(eltype(mean(N)), eltype(covp(N)))
    T = float(T)
    ST = Base.promote_op(convert, Type{AbstractVector{T}}, typeof(mean(N)))
    return ST
end
sample_type(N::UvNormal) = float(promote_type(typeof(mean(N)), typeof(covp(N))))

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

_logpiconst(T::Type{<:Real}) = log(T(2π))
_logpiconst(T::Type{<:Complex}) = log(real(T)(π))

function logpdf(N::AbstractNormal, x)
    T = sample_eltype(N)
    return -_nscale(T) *
           (dim(N) * _logpiconst(T) + real(logdet(covp(N))) + norm_sqr(residual(N, x)))
end

"""
    entropy(N::AbstractNormal)

Computes the entropy of the Normal distribution N.
"""
function entropy(N::AbstractNormal)
    T = sample_eltype(N)
    _nscale(T) * (dim(N) * (_logpiconst(T) + one(real(T))) + real(logdet(covp(N))))
end

"""
    kldivergence(N1::AbstractNormal, N2::AbstractNormal)

Computes the Kullback-Leibler divergence between the Normal distributions N1 and N2.
"""
function kldivergence(N1::AbstractNormal, N2::AbstractNormal)
    T = promote_type(sample_eltype(N1), sample_eltype(N2))
    root_ratio = lsqrt(covp(N2)) \ lsqrt(covp(N1))
    _nscale(T) * (
        norm_sqr(root_ratio) + norm_sqr(residual(N2, mean(N1))) - dim(N1) -
        real(T)(2) * real(logdet(root_ratio))
    )
end

function rand(rng::AbstractRNG, N::AbstractNormal)
    T = sample_eltype(N)
    x = mean(N) + lsqrt(covp(N)) * randn(rng, T, dim(N))
    return sample_type(N)(x)
end

rand(rng::AbstractRNG, N::UvNormal) = mean(N) + lsqrt(covp(N)) * randn(rng, sample_type(N))

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
