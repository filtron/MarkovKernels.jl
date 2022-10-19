# compisitions of kernels

compose(
    K2::AffineDiracKernel{T},
    K1::AffineNormalKernel{T},
) where {T} =
    NormalKernel(compose(mean(K2), mean(K1)), stein(covp(K1), mean(K2)))

compose(
    K2::AffineNormalKernel{T},
    K1::AffineDiracKernel{T},
) where {T} =
    NormalKernel(compose(mean(K2), mean(K1)), covp(K2))
