
_make_vector(v::AbstractVector, ::Type{Matrix}) = Vector(v)
#_make_vector(v::AbstractVector, ::Type{SMatrix}) = SVector{length(v)}(v)

_make_matrix(A::AbstractMatrix, ::Type{Matrix}) = Matrix(A)
#_make_matrix(A::AbstractMatrix, ::Type{SMatrix}) = SMatrix{size(A)...}(A)

_make_covp(A::AbstractMatrix{T}, ::Type{LinearAlgebra.HermOrSym}) where {T} =
    T <: Complex ? Hermitian(A) : Symmetric(A)
_make_covp(A::AbstractMatrix, ::Type{Cholesky}) = cholesky(A)
_make_covp(A::AbstractMatrix, ::Type{SelfAdjoint}) = selfadjoint(A)

function _ofsametype(Ain::AbstractVector, Aout::AbstractVector)
    typeof(Aout) <: typeof(Ain)
end

function _ofsametype(Ain::AbstractMatrix, Aout::AbstractMatrix)
    typeof(Aout) <: typeof(Ain)
end

function _ofsametype(Ain::AbstractMatrix, Aout::LinearAlgebra.HermOrSym)
    typeof(parent(Aout)) <: typeof(Ain)
end

_ofsametype(Ain::AbstractMatrix, Aout::Cholesky) = _ofsametype(Ain, Aout.factors)

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
