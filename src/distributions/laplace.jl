"""
    AbstractLaplace{ST}

Abstract type for representing Laplace distributed random vectors taking values in ST.
"""
abstract type AbstractLaplace{ST} <: AbstractDistribution{ST} end

"""
    Laplace{ST,U,V}

location parameter / covariance matrix parametrization of the Laplace distribution with sample type ST.
"""
struct Laplace{ST,U,V} <: AbstractLaplace{ST}
    μ::U
    Σ::V
end

Laplace{ST}(μ::AbstractVector, Σ, ::IsPSD) where {ST<:AbstractVector} =
    Laplace{ST,typeof(μ),typeof(Σ)}(μ, Σ)
Laplace{ST}(μ::Number, Σ, ::IsPSD) where {ST<:Number} =
    Laplace{ST,typeof(μ),typeof(Σ)}(μ, Σ)

"""
    Laplace(μ, Σ)

Creates a Laplace distribution with location parameter μ and covariance Σ.
"""
function Laplace(μ::AbstractVector, Σ)
    T = promote_type(eltype(μ), eltype(Σ))
    T = float(T)
    ST = Base.promote_op(convert, Type{AbstractVector{T}}, typeof(μ))
    return Laplace{ST}(μ, Σ, psdcheck(Σ))
end

function Laplace(μ::Number, Σ)
    T = promote_type(eltype(μ), eltype(Σ))
    ST = float(T)
    return Laplace{ST}(μ, Σ, psdcheck(Σ))
end

Laplace(μ::Number, Σ::UniformScaling) = Laplace(μ, Σ.λ)

const UnivariateLaplace{T,V} = Laplace{V,V,V}
const IsotropicLaplace{ST,MT,VT} = Laplace{ST,MT,VT} where {VT<:UniformScaling}

function Base.copy!(Ldst::A, Lsrc::A) where {T,U,V<:Cholesky,A<:Laplace{T,U,V}}
    copy!(location(Ldst), location(Lsrc))
    if covp(Ldst).uplo == covp(Lsrc).uplo
        copy!(covp(Ldst).factors, covp(Lsrc).factors)
    else
        copy!(covp(Ldst).factors, adjoint(covp(Lsrc).factors))
    end
    return Ldst
end

function Base.similar(L::Laplace{T,U,V}) where {T,U,V<:Cholesky}
    C = covp(L)
    return Laplace(similar(location(L)), Cholesky(similar(C.factors), C.uplo, C.info))
end

function Base.isapprox(
    L1::Laplace{T1,V1,<:Cholesky},
    L2::Laplace{T2,V2,<:Cholesky},
    kwargs...,
) where {T1,V1,T2,V2}
    return isapprox(location(L1), location(L2); kwargs...) &&
           isapprox(rsqrt(covp(L1)), rsqrt(covp(L1)); kwargs...)
end

dim(L::Laplace) = length(L.μ)

location(L::Laplace) = L.μ
mean(L::Laplace) = L.μ

covp(L::Laplace) = L.Σ

cov(L::Laplace) = AbstractMatrix(covp(L))
cov(L::Laplace{T,U,V}) where {T,U,V<:AbstractMatrix} = covp(L)
cov(L::UnivariateLaplace) = covp(L)
cov(L::IsotropicLaplace) = covp(L)[1:dim(L), 1:dim(L)]

var(L::AbstractLaplace) = real(diag(cov(L)))
var(L::Laplace{T,U,V}) where {T,U,V<:Cholesky} = map(norm_sqr, eachcol(rsqrt(covp(L))))
var(L::UnivariateLaplace) = cov(L)
var(L::IsotropicLaplace) = typeof(location(L))(fill(covp(L).λ, dim(L)))

std(L::AbstractLaplace) = sqrt.(var(L))

residual(L::AbstractLaplace, x) = lsqrt(covp(L)) \ (x - location(L))

logpdf(L::AbstractLaplace, x) =
    - (dim(L)*log(2) + logdet(covp(L)))/2 - sqrt(2) * sum(abs, residual(L, x))
logpdf(L::IsotropicLaplace, x) =
    - dim(L)/2 * (log(2) + log(covp(L).λ)) - sqrt(2) * residual(L, x)

entropy(L::UnivariateLaplace) = log(2*var(L)) / 2 + 1
entropy(L::AbstractLaplace) = (log(2) * dim(L) + logdet(covp(L))) / 2 + dim(L)

function rand(rng::AbstractRNG, L::AbstractLaplace)
    scale = sample_eltype(L)(sqrt(1/2))
    v =
        scale * randexp(rng, sample_eltype(L), dim(L)) .*
        ifelse.(rand(rng, Bool, dim(L)), -1, 1)     # unit covariance
    x = location(L) + lsqrt(covp(L)) * v
    return x
end

function rand(rng::AbstractRNG, L::UnivariateLaplace)
    scale = sample_eltype(L)(sqrt(1/2))
    v = scale * randexp(rng, sample_eltype(L)) .* ifelse.(rand(rng, Bool), -1, 1)
    x = location(L) + lsqrt(covp(L)) * v
    return x
end

function Base.show(io::IO, L::Laplace)
    println(io, summary(L))
    print(io, "μ = ")
    show(io, (L.μ))
    print(io, "\nΣ = ")
    show(io, L.Σ)
end

function Base.show(io::IO, L::UnivariateLaplace)
    println(io, summary(L))
    print(io, "μ = ")
    show(io, (L.μ))
    print(io, "\nσ² = ")
    show(io, L.Σ)
end
