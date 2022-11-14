
function _make_vector(v::AbstractVector, s)
    n = length(v)
    if s === :Matrix
        return Vector(v)
    elseif s === :SMatrix
        return SVector{n}(v)
    end
end

function _make_matrix(A::AbstractMatrix, s)
    n, m = size(A)
    if s === :Matrix
        return Matrix(A)
    elseif s === :SMatrix
        return SMatrix{n,m}(A)
    end
end

function _wrap_matrix(A::AbstractMatrix, s)
    if s === :AbstractMatrix
        return A
    elseif s === :Diagonal
        return Diagonal(A)
    end
end

function _make_covp(A::AbstractMatrix{T}, s) where {T}
    if s === :HermOrSym
        return T <: Complex ? Hermitian(A) : Symmetric(A)
    elseif s === :Cholesky
        return cholesky(A)
    end
end
