

# simple Kalman update function + prediction error loglike
function update(N::AbstractNormal,L::Likelihood)

    M, C = invert(N,L.K)

    N_new = condition(C,L.y)
    loglike = logpdf(M,L.y)

    return N_new, M, loglike

end