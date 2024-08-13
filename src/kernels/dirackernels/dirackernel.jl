"""
    DiracKernel

Type for representing Dirac kernels K(y,x) = δ(y - μ(x)).
"""
struct DiracKernel{A} <: AbstractDiracKernel
    μ::A
end

"""
    AffineDiracKernel

Alias for DiracKernel{<:AbstractAffineMap}
"""
const AffineDiracKernel{T} = DiracKernel{<:AbstractAffineMap{T}} where {T}

mean(K::DiracKernel) = K.μ
