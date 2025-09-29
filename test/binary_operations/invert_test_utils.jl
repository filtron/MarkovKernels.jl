_mat_or_num(V) = Matrix(V)
_mat_or_num(V::Number) = V

_cov(K::NormalKernel) = _mat_or_num(covparam(K))
_cov(::DiracKernel) = false * I

_symmetrise(T, Σ) = Σ
_symmetrise(::Type{T}, Σ::AbstractMatrix{T}) where {T<:Real} = Symmetric(Σ)
_symmetrise(::Type{T}, Σ::AbstractMatrix{T}) where {T<:Complex} = Hermitian(Σ)

function _schur(Σ, C)
    S = C * Σ * C'
    S = _symmetrise(eltype(S), S)
    G = Σ * C' / S
    Π = Σ - G * S * G'
    Π = _symmetrise(eltype(Π), Π)
    return S, G, Π
end

function _schur(Σ, C, R)
    S = C * Σ * C' + R
    S = _symmetrise(eltype(S), S)
    G = Σ * C' / S
    Π = Σ - G * S * G'
    Π = _symmetrise(eltype(Π), Π)
    return S, G, Π
end
