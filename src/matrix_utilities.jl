
# fix logdet for Hermitian matrices
function LinearAlgebra.logdet(H::Hermitian)
    mag, sign = logabsdet(H)
    resign = real(sign)
    return mag + log(resign)
end

"""
    qr2chol(pre_array::AbstractMatrix)

Computes the upper triangular cholesky factor of the matrix pre_array'*pre_array
"""
function rsqrt2cholU(pre_array)
    right = qr(pre_array).R
    right_pos = conj.(sign.(Diagonal(right))) * right
    return UpperTriangular(right_pos)
end


# matrix square-roots
lsqrt(m::AbstractMatrix) = cholesky(m).L
lsqrt(m::UniformScaling) = sqrt(m)
lsqrt(C::Cholesky) = C.L

# project matrix onto symmetric matrix
symmetrise(Σ::AbstractMatrix{T}) where {T} = T <: Real ? Symmetric(Σ) : Hermitian(Σ)
symmetrise(Σ::UniformScaling) = Σ
symmetrise(Σ::Diagonal) = Σ
symmetrise(C::Cholesky) = C

"""
    stein(Σ, Φ, Q)

Computes the output of the stein  operator
Σ ↦ Φ*Σ*Φ' + Q

    stein(Σ, Φ)

Same as stein(Σ, Φ, 0.0I)
"""

# cov_types: HermOrSym, Diagonal, UniformScaling, Cholesky
# priority rules: 
# (HermOrSym; HermOrSym, Diagonal, UniformScaling, Cholesky) = HermOrSym 
# (Diagonal; HermOrSym, Diagonal, UniformScaling, Cholesky) = Cholesky ? 
# (UniformScaling; HermOrSym, Diagonal, UniformScaling, Cholesky) = Cholesky? 
# (Cholesky; HermOrSym, Diagonal, UniformScaling, Cholesky) = Cholesky
stein(Σ, Φ::AbstractMatrix) = symmetrise(Φ * Σ * Φ')
stein(Σ::Cholesky, Φ::AbstractMatrix) = Cholesky(rsqrt2cholU(Σ.U * Φ'))

stein(Σ, Φ::AbstractMatrix, Q) = symmetrise(Φ * Σ * Φ' + Q)
stein(Σ::Cholesky, Φ::AbstractMatrix, Q) = Cholesky(rsqrt2cholU([Σ.U * Φ'; lsqrt(Q)']))
stein(Σ::Cholesky, Φ::AbstractMatrix, Q::Diagonal) =
    Cholesky(rsqrt2cholU([Σ.U * Φ'; diagm(sqrt.(Q.diag))]))
stein(Σ::Cholesky, Φ::AbstractMatrix, Q::Cholesky) = Cholesky(rsqrt2cholU([Σ.U * Φ'; Q.U]))
stein(Σ, Φ::AbstractMatrix, Q::Cholesky) = stein(Σ, Φ, symmetrise(AbstractMatrix(Q)))

stein(Σ, A::AbstractAffineMap) = stein(Σ, slope(A))
stein(Σ, A::AbstractAffineMap, Q) = stein(Σ, slope(A), Q)

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
    pre_array = _make_pre_array(Π, C, R)
    post_array = rsqrt2cholU(pre_array)
    S = Cholesky(UpperTriangular(post_array[1:ny, 1:ny]))
    Σ = Cholesky(UpperTriangular(post_array[ny+1:ny+nx, ny+1:ny+nx]))
    Kt = post_array[1:ny, ny+1:ny+nx]
    K = Kt' / lsqrt(S)    
    return S, K, Σ
end

function _make_pre_array(Π, C, R) 
    ny, nx = size(C)
    pre_array = [lsqrt(R)' zeros(ny, nx); lsqrt(Π)'*C' lsqrt(Π)']
    return pre_array 
end 
_make_pre_array(Π::Cholesky, C, R::Diagonal) = _make_pre_array(Π, C, diagm(R.diag))