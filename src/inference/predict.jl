

# simple Kalman predict function + backwards kernel for RTS smoothing
function predict(N::AbstractNormal,K::NormalKernel)

    N_new, B = invert(N,K)

    return N_new, B

end