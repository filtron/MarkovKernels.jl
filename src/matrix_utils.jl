# Fix logdet for HermOrSym 
function LinearAlgebra.logdet(H::Hermitian)
    mag, sign = logabsdet(H)
    resign = real(sign)
    return mag + log(resign)
end
LinearAlgebra.logdet(H::HermOrSym{T,<:Diagonal}) where {T} = real(logdet(parent(H)))

symmetrise(Σ::AbstractMatrix{T}) where {T} = T <: Real ? Symmetric(Σ) : Hermitian(Σ)

"""
    rsqrt2cholU(pre_array::AbstractMatrix)

Computes the upper triangular cholesky factor of the matrix pre_array'*pre_array
"""
function rsqrt2cholU(pre_array)
    right = qr(pre_array).R
    right_pos = conj.(sign.(Diagonal(right))) * right
    return UpperTriangular(right_pos)
end
