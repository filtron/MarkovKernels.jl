"""
    fix_sign!(A::AbstractMatrix)

Applies an in-place orthogonal transform to the upper triangular matrix A, such that
the diagonal entries are real and positive.  
"""
function fix_sign!(A::AbstractMatrix)
    T = eltype(A)
    LinearAlgebra.require_one_based_indexing(A)
    for row in axes(A, 1)
        dval = A[row, row]
        sign_flip = iszero(dval) ? one(T) : conj(sign(dval)) # LAPACK actually makes sure diag is real valued so conj is technically unnecessary...
        for col in axes(A, 2)[row:end]
            A[row, col] = sign_flip * A[row, col]
        end
    end
    return A
end

"""
    positive_qrwoq!(A::AbstractMatrix)

Computes the R factor in the QR decomposition of A, in-place,
ensuring that the diagonal entries of R are positive.
The returned object is a view of A. 
"""
function positive_qrwoq!(A::AbstractMatrix)
    LinearAlgebra.require_one_based_indexing(A)
    m, n = size(A)
    qr!(A)
    Av = view(A, 1:min(m, n), 1:n)
    triu!(Av)
    Av = fix_sign!(Av)
    return Av
end

"""
    utrichol(U)

Equivalent to Cholesky(UpperTriangular(U)). 
"""
utrichol(U) = Cholesky(UpperTriangular(U))
