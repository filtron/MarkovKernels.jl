
_make_vector(v::AbstractVector, ::Type{Matrix}) = Vector(v)
_make_vector(v::AbstractVector, ::Type{SMatrix}) = SVector{length(v)}(v)
_make_vector(v::AbstractVector, ::Type{CuMatrix}) = CuVector(v)

_make_matrix(A::AbstractMatrix, ::Type{Matrix}) = Matrix(A)
_make_matrix(A::AbstractMatrix, ::Type{SMatrix}) = SMatrix{size(A)...}(A)
_make_matrix(A::AbstractMatrix, ::Type{CuMatrix}) = CuMatrix(A)

_make_covp(A::AbstractMatrix{T}, ::Type{HermOrSym}) where {T} =
    T <: Complex ? Hermitian(A) : Symmetric(A)
_make_covp(A::AbstractGPUMatrix{T}, ::Type{HermOrSym}) where {T} = (A + A') / T(2) # dont wrap GPUMatrix
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

_ofsametype(Ain::AbstractMatrix, Aout::Cholesky) = _ofsametype(Ain, Aout.factors)

_subarray_type(SA::SubArray{T,2,U,S,false}) where {T,U,S} = U

function _ofsametype(Ain::AbstractMatrix, Aout::SubArray)
    _subarray_type(Aout) <: typeof(Ain)
end

_symmetrize(T, Σ) = Σ
_symmetrize(T, Σ::AbstractMatrix) = T <: Real ? Symmetric(Σ) : Hermitian(Σ)
_symmetrize(T, Σ::AbstractGPUArray) = (Σ + Σ') / T(2)

function _schur(Σ, C)
    S = C * Σ * C'
    S = _symmetrize(eltype(S), S)
    G = Σ * C' / S
    Π = Σ - G * S * G'
    Π = _symmetrize(eltype(Π), Π)
    return S, G, Π
end

function _schur(Σ, C, R)
    S = C * Σ * C' + R
    S = _symmetrize(eltype(S), S)
    G = Σ * C' / S
    Π = Σ - G * S * G'
    Π = _symmetrize(eltype(Π), Π)
    return S, G, Π
end

_logdet(A) = logdet(A)
_logdet(C::Cholesky{T,A}) where {T,A<:AbstractGPUMatrix} = T(2) * sum(log, C.factors)
_logdet(A::AbstractGPUMatrix) = logdet(cholesky(A))
