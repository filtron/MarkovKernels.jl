
# fix logdet for Hermitian matrices
function LinearAlgebra.logdet(H::Hermitian)
    mag, sign = logabsdet(H)
    resign = real(sign)
    return mag + log(resign)
end

"""
    rsqrt2cholU(pre_array::AbstractMatrix)

Computes the upper triangular cholesky factor of the matrix pre_array'*pre_array
"""
function rsqrt2cholU(pre_array)
    right = qr(pre_array).R
    right_pos = conj.(sign.(Diagonal(right))) * right
    return UpperTriangular(right_pos)
end

"""
    lsqrt(A) 
returns a 'matrix' L such that A = L*L'. 
L need not be a Cholesky factor.   
"""
lsqrt(A::AbstractMatrix) = cholesky(A).L
lsqrt(J::UniformScaling) = sqrt(J)
lsqrt(C::Cholesky) = C.L

# project matrix onto symmetric matrix
symmetrise(Σ::AbstractMatrix{T}) where {T} = T <: Real ? Symmetric(Σ) : Hermitian(Σ)
symmetrise(Σ::Diagonal) = Σ
symmetrise(Σ::UniformScaling) = Σ
symmetrise(C::Cholesky) = C

const CholeskyCompatible = Union{Diagonal,UniformScaling}

"""
    stein(Σ, Φ, Q)

Computes the output of the stein  operator
Σ ↦ Φ * Σ * Φ' + Q

    stein(Σ, Φ)

Same as stein(Σ, Φ, 0.0I)
"""
stein(Σ, Φ::AbstractMatrix) = symmetrise(Φ * Σ * Φ')
stein(Σ, Φ::AbstractMatrix, Q) = _stein(Σ, Φ, Q)

stein(Σ::Cholesky, Φ::AbstractMatrix) = Cholesky(rsqrt2cholU(lsqrt(Σ)' * Φ'))
stein(Σ::Cholesky, Φ::AbstractMatrix, Q) = _stein_chol(Σ, Φ, Q)
stein(Σ::CholeskyCompatible, Φ::AbstractMatrix, Q::Cholesky) = _stein_chol(Σ, Φ, Q)

stein(Σ, A::AbstractAffineMap) = stein(Σ, slope(A))
stein(Σ, A::AbstractAffineMap, Q) = stein(Σ, slope(A), Q)

_stein(Σ, Φ, Q) = symmetrise(Φ * Σ * Φ' + Q)
_stein(Σ, Φ, Q::Cholesky) = _stein(Σ, Φ, symmetrise(AbstractMatrix(Q)))
_stein_chol(Σ, Φ, Q) = Cholesky(rsqrt2cholU(_stein_pre_array(Σ, Φ, Q)))
_stein_pre_array(Σ, Φ, Q) = [lsqrt(Σ)' * Φ'; lsqrt(Q)']
# below needed unless we jumpt to compat >= 1.8 ? 
_stein_pre_array(Σ, Φ, Q::Diagonal) = [lsqrt(Σ)' * Φ'; diagm(sqrt.(Q.diag))]
_stein_pre_array(Σ::Diagonal, Φ, Q) = [diagm(sqrt.(Σ.diag)) * Φ'; lsqrt(Q)']
_stein_pre_array(Σ::Diagonal, Φ, Q::Diagonal) =
    [diagm(sqrt.(Σ.diag)) * Φ'; diagm(sqrt.(Q.diag))]

"""
    schur_red(Π, C, R)

Returns the tuple (S, K, Σ) associated with the following (block) Schur reduction:

[S C*Π; Π*C' Π] = [0 0; 0 Σ] + [I; K]*S*[I K']

where S = C*Π*C' + R.

In terms of Kalman filtering, if Π is the predictive covariance, C the measurement matrix, and R the measurement covariance,
then S is the marginal measurement covariance, K is the Kalman gain, and Σ is the filtering covariance.

    schur_red(Π, C)

Same as schur_red(Π, C, 0.0I)
"""
function schur_red(Π, C, R)
    K = Π * C'
    S = symmetrise(C * K + R)
    K = K / S

    # Joseph form
    L = (I - K * C)
    Σ = symmetrise(L * Π * L' + K * R * K')

    return S, K, Σ
end
schur_red(Π, C) = schur_red(Π, C, 0.0 * I)

schur_red(Π, C, R::Cholesky) = schur_red(Π, C, symmetrise(AbstractMatrix(R)))
schur_red(Π::Cholesky, C, R) = _schur_red_chol(Π, C, R)
schur_red(Π::Cholesky, C, R::Cholesky) = _schur_red_chol(Π, C, R)

function _schur_red_chol(Π, C, R)
    ny, nx = size(C)
    pre_array = _schur_pre_array(Π, C, R)
    post_array = rsqrt2cholU(pre_array)
    S = Cholesky(UpperTriangular(post_array[1:ny, 1:ny]))
    Σ = Cholesky(UpperTriangular(post_array[ny+1:ny+nx, ny+1:ny+nx]))
    Kt = post_array[1:ny, ny+1:ny+nx]
    K = Kt' / lsqrt(S)
    return S, K, Σ
end

function _schur_pre_array(Π, C, R)
    ny, nx = size(C)
    pre_array = [lsqrt(R)' zeros(ny, nx); lsqrt(Π)'*C' lsqrt(Π)']
    return pre_array
end
_schur_pre_array(Π::Cholesky, C, R::Diagonal) = _schur_pre_array(Π, C, diagm(R.diag)) # not efficient?
