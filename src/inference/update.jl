

# simple Kalman update function + prediction error loglike
function update(N::AbstractNormal,L::Likelihood{NormalKernel{U,V,S},YT}) where {U,V<:AbstractAffineMap,S,YT}

    M, C = invert(N, measurement_model(L) )
    y = measurement(L)
    N_new = condition(C, y )
    loglike = logpdf(M, y )

    return N_new, M, loglike

end