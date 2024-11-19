_symmetrise(T, Σ) = Σ
_symmetrise(::Type{T}, Σ::AbstractMatrix{T}) where {T<:Real} = Symmetric(Σ)
_symmetrise(::Type{T}, Σ::AbstractMatrix{T}) where {T<:Complex} = Hermitian(Σ)

_make_covp(A::AbstractMatrix{T}, ::Type{LinearAlgebra.HermOrSym}) where {T} =
    T <: Complex ? Hermitian(A) : Symmetric(A)
_make_covp(A::AbstractMatrix, ::Type{Cholesky}) = cholesky(A)

function _logpdf(T, μ1, Σ1, x1)
    n = length(μ1)
    Σ1 = _symmetrise(T, Σ1)
    if T <: Real
        logpdf = -T(0.5) * logdet(T(2π) * Σ1) - T(0.5) * dot(x1 - μ1, inv(Σ1), x1 - μ1)
    elseif T <: Complex
        logpdf = -real(T)(n) * log(real(T)(π)) - logdet(Σ1) - dot(x1 - μ1, inv(Σ1), x1 - μ1)
    end

    return logpdf
end

function _entropy(T, μ1, Σ1)
    n = length(μ1)
    Σ1 = _symmetrise(T, Σ1)
    if T <: Real
        entropy = T(0.5) * logdet(T(2π) * exp(T(1)) * Σ1)
    elseif T <: Complex
        entropy = real(T)(n) * log(real(T)(π)) + logdet(Σ1) + real(T)(n)
    end
end

function _kld(T, μ1, Σ1, μ2, Σ2)
    n = length(μ1)
    Σ1 = _symmetrise(T, Σ1)
    Σ2 = _symmetrise(T, Σ2)

    if T <: Real
        kld =
            T(0.5) *
            (tr(Σ2 \ Σ1) - T(n) + dot(μ2 - μ1, inv(Σ2), μ2 - μ1) + logdet(Σ2) - logdet(Σ1))
    elseif T <: Complex
        kld =
            real(tr(Σ2 \ Σ1)) - real(T)(n) +
            real(dot(μ2 - μ1, inv(Σ2), μ2 - μ1)) +
            logdet(Σ2) - logdet(Σ1)
    end

    return kld
end
