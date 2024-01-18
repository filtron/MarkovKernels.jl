"""
    marginalise(D::AbstractDistribution, K::AbstractMarkovKernel)

Computes M, the marginalization of K with respect to D, i.e.,

M(y) = ∫ K(y,x)D(x) dx
"""
marginalize(N::AbstractNormal{T}, K::AffineNormalKernel{T}) where {T} =
    Normal(mean(K)(mean(N)), stein(covp(N), mean(K), covp(K)))

marginalize(N::AbstractNormal{T}, K::AffineDiracKernel{T}) where {T} =
    Normal(mean(K)(mean(N)), stein(covp(N), mean(K)))

marginalize(D::AbstractDirac, K::AbstractMarkovKernel) = condition(K, mean(D))

marginalize(D::AbstractDistribution, ::IdentityKernel) = D # should maybe return a copy? 
marginalize(D::Dirac, ::IdentityKernel) = D # tie-breaker

function marginalize(
    P::ParticleSystem{T,U,<:AbstractArray},
    K::AffineDiracKernel{T},
) where {T,U}
    return ParticleSystem(logweights(P), mean(K).(particles(P)))
end
