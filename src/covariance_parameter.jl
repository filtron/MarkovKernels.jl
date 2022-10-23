const CovarianceParameter{T} = Union{HermOrSym{T},UniformScaling{T},Factorization{T}}

for P in (:UniformScaling, :Factorization)
    @eval CovarianceParameter{T}(Σ::$P) where {T} = convert($P{T}, Σ)
end
CovarianceParameter{T}(Σ::HermOrSym) where {T} = convert(AbstractMatrix{T}, Σ)

convert(::Type{CovarianceParameter{T}}, Σ::CovarianceParameter) where {T} =
    CovarianceParameter{T}(Σ)

"""
    lsqrt(A) 
returns a square 'matrix' L such that A = L*L'. 
L need not be a Cholesky factor.   
"""
lsqrt(A::AbstractMatrix) = cholesky(A).L
lsqrt(J::UniformScaling) = sqrt(J)
lsqrt(C::Cholesky) = C.L

const FactorizationCompatible{T,V} = Union{HermOrSym{T,Diagonal{T,V}},UniformScaling{T}}

"""
    stein(Σ, Φ, Q)

Computes the output of the stein  operator
Σ ↦ Φ * Σ * Φ' + Q

    stein(Σ, Φ)

Mathematically, the same as stein(Σ, Φ, R) for R = 0.
"""
stein(Σ, Φ::AbstractMatrix) = symmetrise(Φ * Σ * Φ')
stein(Σ, Φ::AbstractMatrix, Q) = _stein(Σ, Φ, Q)

stein(Σ::Cholesky, Φ::AbstractMatrix) = Cholesky(rsqrt2cholU(lsqrt(Σ)' * Φ'))
stein(Σ::Cholesky, Φ::AbstractMatrix, Q) = _stein_chol(Σ, Φ, Q)
stein(Σ::FactorizationCompatible, Φ::AbstractMatrix, Q::Cholesky) = _stein_chol(Σ, Φ, Q)

stein(Σ, A::AbstractAffineMap) = stein(Σ, slope(A))
stein(Σ, A::AbstractAffineMap, Q) = stein(Σ, slope(A), Q)

_stein(Σ, Φ, Q) = symmetrise(Φ * Σ * Φ' + Q)
_stein(Σ, Φ, Q::Cholesky) = _stein(Σ, Φ, symmetrise(AbstractMatrix(Q)))
_stein_chol(Σ, Φ, Q) = Cholesky(rsqrt2cholU([lsqrt(Σ)' * Φ'; lsqrt(Q)']))

"""
    schur_reduce(Π, C, R)

Returns the tuple (S, K, Σ) associated with the following (block) Schur reduction:

[C*Π*C' + R C*Π; Π*C' Π] = [0 0; 0 Σ] + [I; K]*(C*Π*C' + R)*[I; K]'

In terms of Kalman filtering, Π is the predictive covariance, C the measurement matrix, and R the measurement covariance,
then S is the marginal measurement covariance, K is the Kalman gain, and Σ is the filtering covariance.

    schur_reduce(Π, C)

Mathematically, the same as schur_red(Π, C, R) for R = 0
"""
schur_reduce(Π, C::AbstractMatrix) = _schur_red(Π, C)
schur_reduce(Π, C::AbstractMatrix, R) = _schur_red(Π, C, R)

schur_reduce(Π::Cholesky, C::AbstractMatrix) = _schur_red_chol(Π, C)
schur_reduce(Π::Cholesky, C::AbstractMatrix, R) = _schur_red_chol(Π, C, R)
schur_reduce(Π::FactorizationCompatible, C::AbstractMatrix, R::Cholesky) =
    _schur_red_chol(Π, C, R)

schur_reduce(Π, C::AbstractAffineMap) = schur_reduce(Π, slope(C))
schur_reduce(Π, C::AbstractAffineMap, R) = schur_reduce(Π, slope(C), R)

function _schur_red(Π, C)
    K = Π * C'
    S = symmetrise(C * K)
    K = K / S
    L = (I - K * C)
    Σ = symmetrise(L * Π * L')
    return S, K, Σ
end

function _schur_red(Π, C, R)
    K = Π * C'
    S = symmetrise(C * K + R)
    K = K / S
    L = (I - K * C)
    Σ = symmetrise(L * Π * L' + K * R * K')
    return S, K, Σ
end

_schur_red(Π, C, R::Cholesky) = _schur_red(Π, C, symmetrise(AbstractMatrix(R)))

function _schur_red_chol(Π, C)
    ny, nx = size(C)
    pre_array = [zeros(ny, nx + ny); lsqrt(Π)'*C' lsqrt(Π)']
    post_array = rsqrt2cholU(pre_array)
    S, K, Σ = _schur_red_chol_make_output(ny, nx, post_array)
    return S, K, Σ
end

function _schur_red_chol(Π, C, R)
    ny, nx = size(C)
    pre_array = [lsqrt(R)' zeros(ny, nx); lsqrt(Π)'*C' lsqrt(Π)']
    post_array = rsqrt2cholU(pre_array)
    S, K, Σ = _schur_red_chol_make_output(ny, nx, post_array)
    return S, K, Σ
end

function _schur_red_chol_make_output(ny, nx, post_array)
    S = Cholesky(UpperTriangular(post_array[1:ny, 1:ny]))
    Σ = Cholesky(UpperTriangular(post_array[ny+1:ny+nx, ny+1:ny+nx]))
    Kt = post_array[1:ny, ny+1:ny+nx]
    K = Kt' / lsqrt(S)
    return S, K, Σ
end
