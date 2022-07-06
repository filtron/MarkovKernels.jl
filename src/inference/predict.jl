

# simple Kalman predict function + backwards kernel for RTS smoothing
function predict(N::AbstractNormal,K::NormalKernel{T,U,V}) where {T,U<:AbstractAffineMap,V}

    N_new, B = invert(N,K)

    return N_new, B

end