const RealSymmetric{T,S} = Symmetric{T,S} where {T<:Real,S}
const ComplexHermitian{T,S} = Hermitian{T,S} where {T<:Complex,S}
const SelfAdjoint{T,S} = Union{RealSymmetric{T,S},ComplexHermitian{T,S}} where {T,S}

ispsdparametrization(::SelfAdjoint) = IsPSDParametrization()

psdparametrization(::Type{T}, A::AbstractMatrix) where {T<:Complex} =
    Hermitian(convert(AbstractMatrix{T}, A))
psdparametrization(::Type{T}, A::AbstractMatrix) where {T<:Real} =
    Symmetric(convert(AbstractMatrix{T}, A))

"""
    selfadjoint(x::Number)

Computes the self-adjoint part of the input  (real part).
"""
selfadjoint(x::Number) = real(x)

"""
    selfadjoint(A::AbstractMatrix{<:Real})

Wraps the input in Symmetric.
"""
selfadjoint(A::AbstractMatrix{<:Real}) = Symmetric(A)

"""
    selfadjoint(A::AbstractMatrix{<:Complex})

Wraps the input in Hermitian.
"""
selfadjoint(A::AbstractMatrix{<:Complex}) = Hermitian(A)

"""
    rsqrt(A::SelfAdjoint)

Computes the rigtht square-root of A.
"""
rsqrt(A::SelfAdjoint) = cholesky(A).U

"""
    lsqrt(A::SelfAdjoint)

Computes the left square-root of A.
"""
lsqrt(A::SelfAdjoint) = adjoint(rsqrt(A))

"""
    stein(Σ::SelfAdjoint, Φ::AbstractMatrix)

Computes the output of the stein  operator

    Σ ↦ Φ * Σ * Φ'.
"""
stein(Σ::SelfAdjoint, Φ::AbstractMatrix) = selfadjoint(Φ * Σ * adjoint(Φ))

"""
    stein(Σ::SelfAdjoint, Φ::AbstractMatrix, Q::SelfAdjoint)

Computes the output of the stein  operator

    Σ ↦ Φ * Σ * Φ' + Q.

Both Σ and Q need to be of the same CovarianceParameter type, e.g. both SymOrHerm or both Cholesky.
The type of the CovarianceParameter is preserved at the output.
"""
stein(Σ::SelfAdjoint, Φ::AbstractMatrix, Q::SelfAdjoint) =
    selfadjoint(Φ * Σ * adjoint(Φ) + Q)

"""
    stein(Σ::SelfAdjoint, Φ::AbstractMatrix, Q::SelfAdjoint)

Computes the output of the stein  operator

    Σ ↦ Φ * Σ * Φ' + Q.

Both Σ and Q need to be of the same CovarianceParameter type, e.g. both SymOrHerm or both Cholesky.
The type of the CovarianceParameter is preserved at the output.
"""
stein(Σ::SelfAdjoint, Φ::AbstractMatrix, Q::Real) = selfadjoint(Φ * Σ * adjoint(Φ) + Q)

"""
    schur_reduce(Π::SelfAdjoint, C::AbstractMatrix)

Returns the tuple (S, K, Σ) associated with the following (block) Schur reduction:

    [C*Π*C' C*Π; Π*C' Π] = [0 0; 0 Σ] + [I; K]*(C*Π*C')*[I; K]'

In terms of Kalman filtering, Π is the predictive covariance, C the measurement matrix, and R the measurement covariance,
then S is the marginal measurement covariance, K is the Kalman gain, and Σ is the filtering covariance.
"""
function schur_reduce(Π::SelfAdjoint, C::AbstractMatrix)
    K = Π * adjoint(C)
    S = selfadjoint(C * K)
    K = K / S
    L = (I - K * C)
    Σ = selfadjoint(L * Π * adjoint(L))
    return S, K, Σ
end

"""
    schur_reduce(Π::SelfAdjoint, C::AbstractMatrix, R::SelfAdjoint)

Returns the tuple (S, K, Σ) associated with the following (block) Schur reduction:

    [C*Π*C'+R C*Π; Π*C' Π] = [0 0; 0 Σ] + [I; K]*(C*Π*C' + R)*[I; K]'

In terms of Kalman filtering, Π is the predictive covariance, C the measurement matrix, and R the measurement covariance,
then S is the marginal measurement covariance, K is the Kalman gain, and Σ is the filtering covariance.
"""
function schur_reduce(Π::SelfAdjoint, C::AbstractMatrix, R::SelfAdjoint)
    K = Π * adjoint(C)
    S = selfadjoint(C * K + R)
    K = K / S
    L = (I - K * C)
    Σ = selfadjoint(L * Π * adjoint(L) + K * R * adjoint(K))
    return S, K, Σ
end

"""
    schur_reduce(Π::SelfAdjoint, C::AbstractMatrix, R::Real)

Returns the tuple (S, K, Σ) associated with the following (block) Schur reduction:

    [C*Π*C'+R C*Π; Π*C' Π] = [0 0; 0 Σ] + [I; K]*(C*Π*C' + R)*[I; K]'

In terms of Kalman filtering, Π is the predictive covariance, C the measurement matrix, and R the measurement covariance,
then S is the marginal measurement covariance, K is the Kalman gain, and Σ is the filtering covariance.
"""
function schur_reduce(Π::SelfAdjoint, C::AbstractMatrix, R::Real)
    K = Π * adjoint(C)
    S = selfadjoint(C * K + R)
    K = K / S
    L = (I - K * C)
    Σ = selfadjoint(L * Π * adjoint(L) + K * R * adjoint(K))
    return S, K, Σ
end
