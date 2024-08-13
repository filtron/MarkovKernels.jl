struct HomoskedasticNormalKernel{A,B} <: AbstractNormalKernel
    μ::A
    Σ::B
end

function HomoskedasticNormalKernel(F::AbstractAffineMap, Σ)
    T = promote_type(eltype(F), eltype(Σ))
    F = convert(AbstractAffineMap{T}, F)
    Σ = selfadjoint(T, Σ)
    return HomoskedasticNormalKernel{typeof(F),typeof(Σ)}(F, Σ)
end

mean(K::HomoskedasticNormalKernel) = K.μ
covp(K::HomoskedasticNormalKernel) = K.Σ
cov(K::HomoskedasticNormalKernel) = x -> covp(K)
condition(K::HomoskedasticNormalKernel, x) = Normal(mean(K)(x), covp(K))

const AffineHomoskedasticNormalKernel{A,B} =
    HomoskedasticNormalKernel{A,B} where {A<:AbstractAffineMap}

function Base.copy!(
    Kdst::HomoskedasticNormalKernel{A,<:Cholesky},
    Ksrc::HomoskedasticNormalKernel{B,<:Cholesky},
) where {A,B}
    copy!(mean(Kdst), mean(Ksrc))
    if covp(Kdst).uplo == covp(Ksrc).uplo
        copy!(covp(Kdst).factors, covp(Ksrc).factors)
    else
        copy!(covp(Kdst).factors, adjoint(covp(Ksrc).factors))
    end
    return Kdst
end

function Base.similar(K::HomoskedasticNormalKernel{A,<:Cholesky}) where {A}
    HomoskedasticNormalKernel(
        similar(mean(K)),
        Cholesky(similar(covp(K).factors), covp(K).uplo, covp(K).info),
    )
end
