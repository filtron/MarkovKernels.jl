"""
    compose(K2::AbstractMarkovKernel, K1::AbstractMarkovKernel)

Computes K3, the composition of K2 ∘ K1 i.e.,

K3(y,x) = ∫ K2(y,z) K1(z,x) dz.

See also [`∘`](@ref)
"""
compose(K2::AffineHomoskedasticNormalKernel, K1::AffineHomoskedasticNormalKernel) =
    NormalKernel(compose(mean(K2), mean(K1)), stein(covp(K1), mean(K2), covp(K2)))

compose(K2::AffineHomoskedasticNormalKernel, K1::AffineDiracKernel) =
    NormalKernel(compose(mean(K2), mean(K1)), covp(K2))

compose(K2::AffineHeteroskedasticNormalKernel, K1::AffineDiracKernel) =
    NormalKernel(compose(mean(K2), mean(K1)), covp(K2) ∘ mean(K1))

compose(K2::AffineDiracKernel, K1::AffineDiracKernel) =
    DiracKernel(compose(mean(K2), mean(K1)))

compose(K2::AffineDiracKernel, K1::AffineHomoskedasticNormalKernel) =
    NormalKernel(compose(mean(K2), mean(K1)), stein(covp(K1), mean(K2)))

compose(K2::AbstractMarkovKernel, ::IdentityKernel) = K2
compose(::IdentityKernel, K1::AbstractMarkovKernel) = K1
compose(K2::IdentityKernel, ::IdentityKernel) = K2

"""
    ∘(K2::AbstractMarkovKernel, K1::AbstractMarkovKernel)

Computes K3, the composition of K2 ∘ K1 i.e.,

    K3(y,x) = ∫ K2(y,z) K1(z,x) dz.

See also [`compose`](@ref)
"""
∘(K2::AbstractMarkovKernel, K1::AbstractMarkovKernel) = compose(K2, K1)
