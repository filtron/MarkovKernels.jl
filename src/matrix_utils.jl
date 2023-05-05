# Fix some LinearAlgebra.jl functions 

function LinearAlgebra.logdet(H::Hermitian)
    mag, sign = logabsdet(H)
    resign = real(sign)
    return mag + log(resign)
end

LinearAlgebra.logdet(H::HermOrSym{T,<:Diagonal}) where {T} = logdet(real.(parent(H)))
LinearAlgebra.logdet(C::Cholesky{T,A}) where {T,A<:AbstractGPUMatrix} =
    real(T)(2) * sum(log, real.(diag(C.U)))
LinearAlgebra.logdet(A::AbstractGPUMatrix) = logdet(cholesky(A))

LinearAlgebra.inv(H::HermOrSym{T,<:Diagonal}) where {T} =
    T <: Complex ? Hermitian(inv(parent(H))) : Symmetric(inv(parent(H)))
