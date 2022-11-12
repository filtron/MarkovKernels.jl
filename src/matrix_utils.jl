# Fix logdet for HermOrSym
function LinearAlgebra.logdet(H::Hermitian)
    mag, sign = logabsdet(H)
    resign = real(sign)
    return mag + log(resign)
end
LinearAlgebra.logdet(H::HermOrSym{T,<:Diagonal}) where {T} = real(logdet(parent(H)))
