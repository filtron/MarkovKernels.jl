# Fix some LinearAlgebra.jl functions 

function LinearAlgebra.logdet(H::Hermitian)
    mag, sign = logabsdet(H)
    resign = real(sign)
    return mag + log(resign)
end
LinearAlgebra.logdet(H::HermOrSym{T,<:Diagonal}) where {T} = real(logdet(parent(H)))

LinearAlgebra.inv(H::HermOrSym{T,<:Diagonal}) where {T} =
    T <: Complex ? Hermitian(inv(parent(H))) : Symmetric(inv(parent(H)))
