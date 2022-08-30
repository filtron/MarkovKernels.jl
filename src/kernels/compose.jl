# compisitions of kernels

compose(
    K2::DiracKernel{T,U},
    K1::NormalKernel{T,U,V},
) where {T,U<:AbstractAffineMap,V<:AbstractMatrix} =
    NormalKernel(compose(mean(K2), mean(K1)), stein(cov(K1), mean(K2), cov(K2)))
compose(
    K2::NormalKernel{T,U,V},
    K1::DiracKernel{T,U},
) where {T,U<:AbstractAffineMap,V<:AbstractMatrix} =
    NormalKernel(compose(mean(K2), mean(K1)), cov(K2))