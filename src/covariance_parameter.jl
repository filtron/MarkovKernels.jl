const CovarianceParameter{T} = Union{HermOrSym{T},Factorization{T}}

CovarianceParameter{T}(Σ::Factorization) where {T} = convert(Factorization{T}, Σ)
CovarianceParameter{T}(Σ::HermOrSym) where {T} = convert(AbstractMatrix{T}, Σ)

convert(::Type{CovarianceParameter{T}}, Σ::CovarianceParameter) where {T} =
    CovarianceParameter{T}(Σ)

"""
    lsqrt(A::CovarianceParameter)

Computes a square matrix L such that A = L*L'.
L need not be a Cholesky factor.
"""
lsqrt(C::Cholesky) = C.L
lsqrt(A::HermOrSym) = cholesky(A).L

"""
    stein(Σ::CovarianceParameter, Φ::AbstractMatrix)

Computes the output of the stein  operator

    Σ ↦ Φ * Σ * Φ'.

The type of CovarianceParameter is preserved at the output.
"""
stein(Σ::HermOrSym, Φ::AbstractMatrix) = symmetrise(Φ * Σ * Φ')
stein(Σ::Cholesky, Φ::AbstractMatrix) = _make_post_array(lsqrt(Σ)' * Φ') |> _upper_cholesky

"""
    stein(Σ::CovarianceParameter, Φ::AbstractMatrix, Q::CovarianceParameter)

Computes the output of the stein  operator

    Σ ↦ Φ * Σ * Φ' + Q.

Both Σ and Q need to be of the same CovarianceParameter type, e.g. both SymOrHerm or both Cholesky.
The type of the CovarianceParameter is preserved at the output.
"""
stein(Σ::HermOrSym, Φ::AbstractMatrix, Q::HermOrSym) = symmetrise(Φ * Σ * Φ' + Q)
stein(Σ::Cholesky, Φ::AbstractMatrix, Q::Cholesky) =
    _make_post_array(vcat(lsqrt(Σ)' * Φ', lsqrt(Q)')) |> _upper_cholesky

stein(Σ, A::AbstractAffineMap) = stein(Σ, slope(A))
stein(Σ, A::AbstractAffineMap, Q) = stein(Σ, slope(A), Q)

"""
    schur_reduce(Π::CovarianceParameter, C::AbstractMatrix)

Returns the tuple (S, K, Σ) associated with the following (block) Schur reduction:

    [C*Π*C' C*Π; Π*C' Π] = [0 0; 0 Σ] + [I; K]*(C*Π*C')*[I; K]'

In terms of Kalman filtering, Π is the predictive covariance, C the measurement matrix, and R the measurement covariance,
then S is the marginal measurement covariance, K is the Kalman gain, and Σ is the filtering covariance.
The type of the CovarianceParameter is preserved at the output.
"""
function schur_reduce(Π::HermOrSym, C::AbstractMatrix)
    K = Π * C'
    S = symmetrise(C * K)
    K = K / S
    L = (I - K * C)
    Σ = symmetrise(L * Π * L')
    return S, K, Σ
end

function schur_reduce(Π::Cholesky, C::AbstractMatrix)
    Rfac = ArrayInterfaceCore.zeromatrix(diag(C))
    zero_array = hcat(Rfac, zero(C))
    post_array = vcat(_make_post_array(hcat(lsqrt(Π)' * C', lsqrt(Π)')), zero_array)
    S, K, Σ = _make_schur_output_cholesky(post_array, Π.factors, C, Rfac)

    return S, K, Σ
end

"""
    schur_reduce(Π::CovarianceParameter, C::AbstractMatrix, R::CovarianceParameter)

Returns the tuple (S, K, Σ) associated with the following (block) Schur reduction:

[C*Π*C' + R C*Π; Π*C' Π] = [0 0; 0 Σ] + [I; K]*(C*Π*C' + R)*[I; K]'

In terms of Kalman filtering, Π is the predictive covariance, C the measurement matrix, and R the measurement covariance,
then S is the marginal measurement covariance, K is the Kalman gain, and Σ is the filtering covariance.
Both Π and R need to be of the same CovarianceParameter type, e.g. both SymOrHerm or both Cholesky.
The type of the CovarianceParameter is preserved at the output.
"""
function schur_reduce(Π::HermOrSym, C::AbstractMatrix, R::HermOrSym)
    K = Π * C'
    S = symmetrise(C * K + R)
    K = K / S
    L = (I - K * C)
    Σ = symmetrise(L * Π * L' + K * R * K')
    return S, K, Σ
end

function schur_reduce(Π::Cholesky, C::AbstractMatrix, R::Cholesky)
    pre_array = vcat(hcat(lsqrt(R)', zero(C)), hcat(lsqrt(Π)' * C', lsqrt(Π)'))
    post_array = _make_post_array(pre_array)
    S, K, Σ = _make_schur_output_cholesky(post_array, Π.factors, C, R.factors)
    return S, K, Σ
end

schur_reduce(Π, C::AbstractAffineMap) = schur_reduce(Π, slope(C))
schur_reduce(Π, C::AbstractAffineMap, R) = schur_reduce(Π, slope(C), R)

symmetrise(Σ::AbstractMatrix{T}) where {T} = T <: Real ? Symmetric(Σ) : Hermitian(Σ)

function _make_post_array(pre_array)
    U = qr(pre_array).R
    return conj.(sign.(Diagonal(U))) * U
end
_upper_cholesky(U) = U |> UpperTriangular |> Cholesky

function _make_schur_output_cholesky(post_array, Πfac, C, Rfac)
    ny, nx = size(C)
    S = @inbounds _convert2similar(Rfac, post_array[1:ny, 1:ny]) |> _upper_cholesky
    Σ = @inbounds _convert2similar(Πfac, post_array[ny+1:ny+nx, ny+1:ny+nx]) |>
              _upper_cholesky
    Kt = @inbounds _convert2similar(C, post_array[1:ny, ny+1:ny+nx])
    K = Kt' / lsqrt(S)
    return S, K, Σ
end

# convert here is a bit hacky but only way I could figure out 
# that makes output matrices match  input matrices for eg StaticArrays.  
_convert2similar(Ain::AbstractMatrix, Aout::AbstractMatrix) = convert(typeof(Ain), Aout)
_convert2similar(::Diagonal{T}, Aout::AbstractMatrix) where {T} =
    convert(AbstractMatrix{T}, Aout)
