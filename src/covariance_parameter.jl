const CovarianceParameter{T} = Union{HermOrSym{T},Factorization{T}}

CovarianceParameter{T}(Σ::Factorization) where {T} = convert(Factorization{T}, Σ)
CovarianceParameter{T}(Σ::HermOrSym) where {T} = convert(AbstractMatrix{T}, Σ)

convert(::Type{CovarianceParameter{T}}, Σ::CovarianceParameter) where {T} =
    CovarianceParameter{T}(Σ)

"""
    lsqrt(A::CovarianceParameter)
Computes a square 'matrix' L such that A = L*L'.
L need not be a Cholesky factor.
"""
lsqrt(C::Cholesky) = C.L

"""
    lsqrt(A::AbstractMatrix)

Equivalent to cholesky(A).L
"""
lsqrt(A::AbstractMatrix) = cholesky(A).L

"""
    stein(Σ::CovarianceParameter, Φ::AbstractMatrix)

Computes the output of the stein  operator

    Σ ↦ Φ * Σ * Φ'.

The type of CovarianceParameter is preserved at the output.
"""
stein(Σ::HermOrSym, Φ::AbstractMatrix) = symmetrise(Φ * Σ * Φ')
stein(Σ::Cholesky, Φ::AbstractMatrix) = Cholesky(rsqrt2cholU(lsqrt(Σ)' * Φ'))

"""
    stein(Σ::CovarianceParameter, Φ::AbstractMatrix, Q::CovarianceParameter)

Computes the output of the stein  operator

    Σ ↦ Φ * Σ * Φ' + Q.

Both Σ and Q need to be of the same CovarianceParameter type, e.g. both SymOrHerm or both Cholesky.
The type of the CovarianceParameter is preserved at the output.
"""
stein(Σ::HermOrSym, Φ::AbstractMatrix, Q::HermOrSym) = symmetrise(Φ * Σ * Φ' + Q)
stein(Σ::Cholesky, Φ::AbstractMatrix, Q::Cholesky) = Cholesky(rsqrt2cholU([lsqrt(Σ)' * Φ'; lsqrt(Q)']))

stein(Σ, A::AbstractAffineMap) = stein(Σ, slope(A))
stein(Σ, A::AbstractAffineMap, Q) = stein(Σ, slope(A), Q)

"""
    schur_reduce(Π::CovarianceParameter, C::AbstractMatrix)

Returns the tuple (S, K, Σ) associated with the following (block) Schur reduction:

    [C*Π*C' C*Π; Π*C' Π] = [0 0; 0 Σ] + [I; K]*(C*Π*C')*[I; K]'

In terms of Kalman filtering, Π is the predictive covariance, C the measurement matrix, and R the measurement covariance,
then S is the marginal measurement covariance, K is the Kalman gain, and Σ is the filtering covariance.
"""
function schur_reduce(Π::HermOrSym, C::AbstractMatrix)
    K = Π * C'
    S = symmetrise(C * K)
    K = K / S
    L = (I - K * C)
    Σ = symmetrise(L * Π * L')
    return S, K, Σ
end


"""
    schur_reduce(Π::CovarianceParameter, C::AbstractMatrix, R::CovarianceParameter)

Returns the tuple (S, K, Σ) associated with the following (block) Schur reduction:

[C*Π*C' + R C*Π; Π*C' Π] = [0 0; 0 Σ] + [I; K]*(C*Π*C' + R)*[I; K]'

In terms of Kalman filtering, Π is the predictive covariance, C the measurement matrix, and R the measurement covariance,
then S is the marginal measurement covariance, K is the Kalman gain, and Σ is the filtering covariance.
"""
function schur_reduce(Π::HermOrSym, C::AbstractMatrix, R::HermOrSym)
    K = Π * C'
    S = symmetrise(C * K + R)
    K = K / S
    L = (I - K * C)
    Σ = symmetrise(L * Π * L' + K * R * K')
    return S, K, Σ
end

function schur_reduce(Π::Cholesky, C::AbstractMatrix)
    ny, nx = size(C)
    pre_array = [zeros(ny, nx + ny); lsqrt(Π)'*C' lsqrt(Π)']
    post_array = rsqrt2cholU(pre_array)
    S, K, Σ = _schur_red_chol_make_output(ny, nx, post_array)
    return S, K, Σ
end

function schur_reduce(Π::Cholesky, C::AbstractMatrix, R::Cholesky)
    ny, nx = size(C)
    pre_array = [lsqrt(R)' zeros(ny, nx); lsqrt(Π)'*C' lsqrt(Π)']
    post_array = rsqrt2cholU(pre_array)
    S, K, Σ = _schur_red_chol_make_output(ny, nx, post_array)
    return S, K, Σ
end

schur_reduce(Π, C::AbstractAffineMap) = schur_reduce(Π, slope(C))
schur_reduce(Π, C::AbstractAffineMap, R) = schur_reduce(Π, slope(C), R)

function _schur_red_chol_make_output(ny, nx, post_array)
    S = Cholesky(UpperTriangular(post_array[1:ny, 1:ny]))
    Σ = Cholesky(UpperTriangular(post_array[ny+1:ny+nx, ny+1:ny+nx]))
    Kt = post_array[1:ny, ny+1:ny+nx]
    K = Kt' / lsqrt(S)
    return S, K, Σ
end
