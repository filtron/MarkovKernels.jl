"""
    NormalKernel

Standard mean vector / covariance matrix parametrisation of Normal kernels.
"""
struct NormalKernel{U,V} <: AbstractNormalKernel
    μ::U
    Σ::V
end

const AffineNormalKernel{T} =
    NormalKernel{<:AbstractAffineMap{T},<:CovarianceParameter{T}} where {T}

"""
    NormalKernel(F::AbstractAffineMap{<:Real}, Σ::Symmetric{<:Real})

Creates a NormalKernel with conditional mean function F and a constant conditional covariance function parameterised by Σ.
"""
function NormalKernel(F::AbstractAffineMap{<:Real}, Σ::Symmetric{<:Real})
    T = promote_type(eltype(F), eltype(Σ))
    F = convert(AbstractAffineMap{T}, F)
    Σ = convert(AbstractMatrix{T}, Σ)
    return NormalKernel{typeof(F),typeof(Σ)}(F, Σ)
end

function NormalKernel(F::AbstractAffineMap{<:Complex}, Σ::Hermitian{<:Complex})
    T = promote_type(eltype(F), eltype(Σ))
    F = convert(AbstractAffineMap{T}, F)
    Σ = convert(AbstractMatrix{T}, Σ)
    return NormalKernel{typeof(F),typeof(Σ)}(F, Σ)
end

function NormalKernel(F::AbstractAffineMap, Σ::Factorization)
    T = promote_type(eltype(F), eltype(Σ))
    F = convert(AbstractAffineMap{T}, F)
    Σ = convert(Factorization{T}, Σ)
    return NormalKernel{typeof(F),typeof(Σ)}(F, Σ)
end

mean(K::NormalKernel) = K.μ
covp(K::NormalKernel) = K.Σ
cov(K::NormalKernel) = K.Σ
cov(K::AffineNormalKernel) = x -> covp(K)
condition(K::AffineNormalKernel, x) = Normal(mean(K)(x), covp(K))

function Base.copy!(
    Kdst::A,
    Ksrc::A,
) where {T,V<:Cholesky,A<:AffineNormalKernel{T,<:AbstractAffineMap{T},V}}
    copy!(mean(Kdst), mean(Ksrc))
    if covp(Kdst).uplo == covp(Ksrc).uplo
        copy!(covp(Kdst).factors, covp(Ksrc).factors)
    else
        copy!(covp(Kdst).factors, adjoint(covp(Ksrc).factors))
    end
    return Kdst
end

function Base.similar(K::AffineNormalKernel{T,<:AbstractAffineMap{T},<:Cholesky}) where {T}
    NormalKernel(
        similar(mean(K)),
        Cholesky(similar(covp(K).factors), covp(K).uplo, covp(K).info),
    )
end
