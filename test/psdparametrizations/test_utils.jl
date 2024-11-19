_to_matrix(x::Number) = x
_to_matrix(F::Factorization) = AbstractMatrix(F)
_to_matrix(A::AbstractMatrix) = A
_to_matrix(v::AbstractVector) = v

_stein(Σ, C) = C * Σ * adjoint(C)
_stein(Σ, C, R) = _stein(Σ, C) + R

function _schur_reduce(Π, C)
    S = _stein(Π, C)
    K = Π * adjoint(C) / S
    Σ = Π - K * S * adjoint(K)
    return S, K, Σ
end

function _schur_reduce(Π, C, R)
    S = _stein(Π, C, R)
    K = Π * adjoint(C) / S
    Σ = Π - K * S * adjoint(K)
    return S, K, Σ
end
