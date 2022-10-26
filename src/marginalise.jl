"""
    marginalise(D::AbstractDistribution, K::AbstractMarkovKernel)

Computes M, the marginalisation of K with respect to D, i.e.,

M(y) = ∫ K(y,x)D(x) dx
"""
marginalise(N::AbstractNormal{T}, K::AffineNormalKernel{T}) where {T} =
    Normal(mean(K)(mean(N)), stein(covp(N), mean(K), covp(K)))

marginalise(N::AbstractNormal{T}, K::AffineDiracKernel{T}) where {T} =
    Normal(mean(K)(mean(N)), stein(covp(N), mean(K)))

marginalise(D::AbstractDirac{T}, K::AffineNormalKernel{T}) where {T} =
    Normal(mean(K)(mean(D)), covp(K))

marginalise(D::AbstractDirac{T}, K::AbstractDiracKernel{T}) where {T} =
    Dirac(mean(K)(mean(D)))
