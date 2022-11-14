
_make_vector(v::AbstractVector, ::Type{Matrix}) = Vector(v)
_make_vector(v::AbstractVector, ::Type{SMatrix}) = SVector{length(v)}(v)

_make_matrix(A::AbstractMatrix, ::Type{Matrix}) = Matrix(A)
_make_matrix(A::AbstractMatrix, ::Type{SMatrix}) = SMatrix{size(A)...}(A)

_make_covp(A::AbstractMatrix{T}, ::Type{HermOrSym}) where {T} =
    T <: Complex ? Hermitian(A) : Symmetric(A)
_make_covp(A::AbstractMatrix, ::Type{Cholesky}) = cholesky(A)

function _ofsametype(Ain::AbstractVector, Aout::AbstractVector)
    typeof(Aout) <: typeof(Ain)
end

function _ofsametype(Ain::AbstractMatrix, Aout::AbstractMatrix)
    typeof(Aout) <: typeof(Ain)
end

function _ofsametype(Ain::AbstractMatrix, Aout::HermOrSym)
    typeof(parent(Aout)) <: typeof(Ain)
end

function _ofsametype(Ain::Diagonal, Aout::HermOrSym)
    typeof(parent(Aout)) <: typeof(diagm(parent(Ain)))
end

function _ofsametype(Ain::AbstractMatrix, Aout::Cholesky)
    typeof(Aout.factors) <: typeof(Ain)
end

_symmetrise(T, Σ) = Σ
_symmetrise(T, Σ::AbstractMatrix) = T <: Real ? Symmetric(Σ) : Hermitian(Σ)

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
