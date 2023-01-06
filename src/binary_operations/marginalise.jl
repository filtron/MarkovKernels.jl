"""
    marginalise(D::AbstractDistribution, K::AbstractMarkovKernel)

Computes M, the marginalisation of K with respect to D, i.e.,

M(y) = ∫ K(y,x)D(x) dx
"""
marginalise(N::AbstractNormal{T}, K::AffineNormalKernel{T}) where {T} =
    Normal(mean(K)(mean(N)), stein(covp(N), mean(K), covp(K)))

marginalise(N::AbstractNormal{T}, K::AffineDiracKernel{T}) where {T} =
    Normal(mean(K)(mean(N)), stein(covp(N), mean(K)))

marginalise(D::AbstractDirac{T}, K::AbstractMarkovKernel{T}) where {T} =
    condition(K, mean(D))


function marginalise(P::ParticleSystem{T,U,<:AbstractVector}, K::AffineDiracKernel{T}) where {T,U}
    return ParticleSystem(logweights(P), mean(K).(particles(P)))
end
