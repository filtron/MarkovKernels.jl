"""
    AbstractNormal{ST}

Abstract type for representing Normal distributed random vectors taking values in ST.
"""
abstract type AbstractNormal{ST} <: AbstractDistribution{ST} end

const AbstractMultivariateNormal{ST} = AbstractNormal{ST} where {ST<:AbstractVector}
const AbstractUnivariateNormal{ST} = AbstractNormal{ST} where {ST<:Number}

mean_and_covparam(d::AbstractNormal) = Tuple(getfield(d, f) for f in fieldnames(typeof(d)))

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

Normal(μ::Number, Σ::UniformScaling) = Normal(μ, Σ.λ)

const CholeskyNormal{ST,MT,CT} = Normal{ST,MT,CT} where {ST,MT,CT<:Cholesky}

# this needs to change to allow for heterogneous eltype in fields / sample_type
const UnivariateNormal{T,V} =
    Union{Normal{V,V,V},Normal{T,T,V}} where {V<:Real,T<:Complex{V}}
const IsotropicNormal{ST,MT,VT} = Normal{ST,MT,VT} where {VT<:UniformScaling}

"""
    mean(d::AbstractNormal)

Computes the mean vector of the Normal distribution d.
"""
mean(d::Normal) = mean_and_covparam(d)[1]
"""
    covparam(N::AbstractNormal)

Returns the internal representation of the covariance matrix of the Normal distribution d.
For computing the actual covariance matrix, use cov.
"""
covparam(d::Normal) = mean_and_covparam(d)[2]

"""
    dim(d::AbstractNormal)

Returns the dimension of the Normal distribution d.
"""
dim(d::AbstractNormal) = length(mean(d))

"""
    cov(d::AbstractNormal)

Computes the covariance matrix of the Normal distribution d.
"""
cov(d::AbstractMultivariateNormal) = AbstractMatrix(covparam(d))
cov(d::AbstractUnivariateNormal) = covparam(d)
cov(d::Normal{T,U,V}) where {T,U,V<:AbstractMatrix} = covparam(d)
cov(d::IsotropicNormal) = covparam(d)[1:dim(d), 1:dim(d)]

"""
    var(d::AbstractNormal)
Computes the vector of marginal variances of the Normal distribution d.
"""
var(d::AbstractMultivariateNormal) = real(diag(cov(d)))
var(d::AbstractUnivariateNormal) = cov(d)
var(d::Normal{T,U,V}) where {T,U,V<:Cholesky} = map(norm_sqr, eachcol(rsqrt(covparam(d))))
var(d::IsotropicNormal) = fill(covparam(d).λ, dim(d))

"""
    std(d::AbstractNormal)
Computes the vector of marginal standard deviations of the Normal distribution d.
"""
std(d::AbstractNormal) = sqrt.(var(d))

"""
    residual(d::AbstractNormal, x::AbstractVector)

Computes the whitened residual associated with the Normal distribution d and observed vector x.
"""
residual(d::AbstractNormal, x) = lsqrt(covparam(d)) \ (x - mean(d))

_nscale(T::Type{<:Real}) = T(0.5)
_nscale(T::Type{<:Complex}) = one(real(T))

_logpiconst(T::Type{<:Real}) = log(T(2π))
_logpiconst(T::Type{<:Complex}) = log(real(T)(π))

function logpdf(d::AbstractNormal, x)
    T = sample_eltype(d)
    return -_nscale(T) *
           (dim(d) * _logpiconst(T) + real(logdet(covparam(d))) + norm_sqr(residual(d, x)))
end

function logpdf(d::IsotropicNormal, x)
    T = sample_eltype(d)
    ld = dim(d) * log(covparam(d).λ)
    return -_nscale(T) * (dim(d) * _logpiconst(T) + ld + norm_sqr(residual(d, x)))
end

"""
    entropy(d::AbstractNormal)

Computes the entropy of the Normal distribution d.
"""
function entropy(d::AbstractNormal)
    T = sample_eltype(d)
    _nscale(T) * (dim(d) * (_logpiconst(T) + one(real(T))) + real(logdet(covparam(d))))
end

function entropy(d::IsotropicNormal)
    T = sample_eltype(d)
    ld = dim(d) * log(covparam(d).λ)
    _nscale(T) * (dim(d) * (_logpiconst(T) + one(real(T))) + ld)
end

"""
    kldivergence(d1::AbstractNormal, d2::AbstractNormal)

Computes the Kullback-Leibler divergence between the Normal distributions d1 and d2.
"""
function kldivergence(d1::AbstractNormal, d2::AbstractNormal)
    T = promote_type(sample_eltype(d1), sample_eltype(d2))
    root_ratio = lsqrt(covparam(d2)) \ lsqrt(covparam(d1))
    _nscale(T) * (
        norm_sqr(root_ratio) + norm_sqr(residual(d2, mean(d1))) - dim(d1) -
        real(T)(2) * real(logdet(root_ratio))
    )
end

function kldivergence(d1::IsotropicNormal, d2::IsotropicNormal)
    T = promote_type(sample_eltype(d1), sample_eltype(d2))
    root_ratio = lsqrt(covparam(d2)) \ lsqrt(covparam(d1))
    root_ratio_norm_sqr = root_ratio.λ^2 * dim(d1)
    root_ratio_ld = dim(d1) * log(root_ratio.λ)
    _nscale(T) * (
        root_ratio_norm_sqr + norm_sqr(residual(d2, mean(d1))) - dim(d1) -
        real(T)(2) * root_ratio_ld
    )
end

function rand(rng::AbstractRNG, d::AbstractMultivariateNormal)
    T = eltype(sample_type(d))
    x = mean(d) + lsqrt(covparam(d)) * randn(rng, T, dim(d))
    return sample_type(d)(x)
end

rand(rng::AbstractRNG, d::AbstractUnivariateNormal) =
    mean(d) + lsqrt(covparam(d)) * randn(rng, sample_type(d))

function Base.copy!(ddst::CholeskyNormal, dsrc::CholeskyNormal)
    μdst, Σdst = mean_and_covparam(ddst)
    μsrc, Σsrc = mean_and_covparam(dsrc)
    copy!(μdst, μsrc)
    copy!(rsqrt(Σdst), rsqrt(Σsrc))
    return ddst
end

#= 
function Base.similar(d::CholeskyNormal)
    C = covparam(d)
    return Normal(similar(mean(d)), Cholesky(similar(C.factors), C.uplo, C.info))
end
=#

similar(d::Normal) = similar(d, eltype(sample_type(d)), dim(d))
similar(d::Normal, m) = similar(d, eltype(sample_type(d)), m)
similar(d::Normal, ::Type{T}) where {T} = similar(d, T, dim(d))

function similar(d::Normal, ::Type{T}, m) where {T}
    μ, Σ = mean_and_covparam(d)
    μ = similar(μ, T, m)
    Σ = psdsimilar(Σ, T, m)
    return Normal(μ, Σ)
end

isequal(d1::CholeskyNormal, d2::CholeskyNormal) =
    mean(d1) == mean(d2) && rsqrt(covparam(d1)) == rsqrt(covparam(d2))
==(d1::CholeskyNormal, d2::CholeskyNormal) = isequal(d1, d2)

function Base.isapprox(d1::CholeskyNormal, d2::CholeskyNormal, kwargs...)
    return isapprox(mean(d1), mean(d2); kwargs...) &&
           isapprox(rsqrt(covparam(d1)), rsqrt(covparam(d2)); kwargs...)
end

function Base.show(io::IO, d::AbstractMultivariateNormal)
    println(io, summary(d))
    print(io, "μ = ")
    show(io, mean(d))
    print(io, "\nΣ = ")
    show(io, covparam(d))
end

function Base.show(io::IO, d::AbstractUnivariateNormal)
    println(io, summary(d))
    print(io, "μ = ")
    show(io, mean(d))
    print(io, "\nσ² = ")
    show(io, covparam(d))
end
