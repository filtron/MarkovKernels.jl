"""
    compose(K2::AbstractMarkovKernel, K1::AbstractMarkovKernel)

Computes K3, the composition of K2 ∘ K1 i.e.,

K3(y,x) = ∫ K2(y,z) K1(z,x) dz.

See also [`∘`](@ref)
"""
compose(K2::AffineNormalKernel{T}, K1::AffineNormalKernel{T}) where {T} =
    NormalKernel(compose(mean(K2), mean(K1)), stein(covp(K1), mean(K2), covp(K2)))

compose(K2::AffineNormalKernel{T}, K1::AffineDiracKernel{T}) where {T} =
    NormalKernel(compose(mean(K2), mean(K1)), covp(K2))

compose(K2::NormalKernel{<:AbstractAffineMap{T}}, K1::AffineDiracKernel{T}) where {T} =
    NormalKernel(compose(mean(K2), mean(K1)), covp(K2) ∘ mean(K1))

compose(K2::AffineDiracKernel{T}, K1::AffineDiracKernel{T}) where {T} =
    DiracKernel(compose(mean(K2), mean(K1)))

compose(K2::AffineDiracKernel{T}, K1::AffineNormalKernel{T}) where {T} =
    NormalKernel(compose(mean(K2), mean(K1)), stein(covp(K1), mean(K2)))

compose(K2::AbstractMarkovKernel, ::IdentityKernel) = K2
compose(::IdentityKernel, K1::AbstractMarkovKernel) = K1
compose(K2::IdentityKernel, ::IdentityKernel) = K2 # tie-breaker

""" 
    ∘(K2::AbstractMarkovKernel, K1::AbstractMarkovKernel)

Computes K3, the composition of K2 ∘ K1 i.e.,

    K3(y,x) = ∫ K2(y,z) K1(z,x) dz.

See also [`compose`](@ref)
"""
∘(K2::AbstractMarkovKernel, K1::AbstractMarkovKernel) = compose(K2, K1)
