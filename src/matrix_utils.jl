# Fix some LinearAlgebra.jl functions 

if VERSION ≤ v"1.10.0"
    function LinearAlgebra.logdet(H::Hermitian)
        mag, sign = logabsdet(H)
        resign = real(sign)
        return mag + log(resign)
    end
    LinearAlgebra.logdet(H::HermOrSym{T,<:Diagonal}) where {T} = logdet(real.(parent(H)))
end

LinearAlgebra.inv(H::HermOrSym{T,<:Diagonal}) where {T} =
    T <: Complex ? Hermitian(inv(parent(H))) : Symmetric(inv(parent(H)))
