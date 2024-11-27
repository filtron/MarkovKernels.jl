const RealSymmetric{T,S} = Symmetric{T,S} where {T<:Real,S}
const ComplexHermitian{T,S} = Hermitian{T,S} where {T<:Complex,S}
const RealDiagonal{T,S} = Diagonal{T,S} where {T<:Real,S}
const SelfAdjoint{T,S} =
    Union{RealSymmetric{T,S},ComplexHermitian{T,S},RealDiagonal{T,S}} where {T,S}

psdcheck(::SelfAdjoint) = IsPSD()

convert_psd_eltype(::Type{T}, A::ComplexHermitian) where {T<:Complex} =
    convert(AbstractMatrix{T}, A)
convert_psd_eltype(::Type{T}, A::RealSymmetric) where {T<:Real} =
    convert(AbstractMatrix{T}, A)

"""
    selfadjoint!(A)

Computes the self-adjoint part of A, in-place, and wraps it in an appropriate self-adjoitn wrapper type (i.e. Symemtric / Hermitian).
"""
selfadjoint!(x::Number) = real(x)
selfadjoint!(A::AbstractMatrix{<:Real}) = Symmetric(hermitianpart!(A))
selfadjoint!(A::AbstractMatrix{<:Complex}) = Hermitian(hermitianpart!(A))

# this should probably be selfadjoint!(D::RealDiagonal) = D
function selfadjoint!(D::Diagonal)
    d = D.diag
    for m in eachindex(d)
        d[m] = real(d[m])
    end
    return D
end

"""
    selfadjoint(A)

Computes the self-adjoint part of A and wraps it in an appropriate self-adjoitn wrapper type (i.e. Symemtric / Hermitian).
"""
selfadjoint(A) = selfadjoint!(copy(A))
selfadjoint(D::Diagonal) = real.(D)

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
stein(Σ::SelfAdjoint, Φ::AbstractMatrix) = selfadjoint!(Φ * Σ * adjoint(Φ))

"""
    stein(Σ::SelfAdjoint, Φ::AbstractMatrix, Q::SelfAdjoint)

Computes the output of the stein  operator

    Σ ↦ Φ * Σ * Φ' + Q.

Both Σ and Q need to be of the same CovarianceParameter type, e.g. both SymOrHerm or both Cholesky.
The type of the CovarianceParameter is preserved at the output.
"""
stein(Σ::SelfAdjoint, Φ::AbstractMatrix, Q::SelfAdjoint) =
    selfadjoint!(Φ * Σ * adjoint(Φ) + Q)

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
    Σ = selfadjoint!(L * Π * adjoint(L))
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
    S = selfadjoint!(C * K + R)
    K = K / S
    L = (I - K * C)
    Σ = selfadjoint!(L * Π * adjoint(L) + K * R * adjoint(K))
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
    S = selfadjoint!(C * K + R)
    K = K / S
    L = (I - K * C)
    Σ = selfadjoint!(L * Π * adjoint(L) + K * R * adjoint(K))
    return S, K, Σ
end
