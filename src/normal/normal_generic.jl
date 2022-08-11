
"""
    dim(N::AbstractNormal)

Returns the dimension of the random vector represented by the distribution N.
"""
dim(N::AbstractNormal) = dim(N)

"""
    mean(N::AbstractNormal)

Returns the mean vector of the distribution N.
"""
mean(N::AbstractNormal) = mean(N)

"""
    cov(N::AbstractNormal)

Returns the covariance matrix of the distribution N.
"""
cov(N::AbstractNormal) = cov(N)

"""
    var(N::AbstractNormal)

Returns a vector of marginal variances of the distribution N.
"""
var(N::AbstractNormal) = var(N)

"""
    std(N::AbstractNormal)

Returns a vector of marginal standard deviations of the distribution N.
"""
std(N::AbstractNormal) = std(N)

"""
    residual(N::AbstractNormal,x)

Given a realisation x, computes the whitened residual with respect to N.
"""
residual(N::AbstractNormal, x) = residual(N,x)


"""
    logpdf(N::AbstractNormal,x)

Evaluates the logarithm  of the probability density of N at x.
"""
logpdf(N::AbstractNormal, x) = logpdf(N,x)

"""
    entropy(N::AbstractNormal)

Computes the entropy of the distribution N.
"""
entropy(N::AbstractNormal) = entropy(N)

"""
    kldivergence(N1::AbstractNormal, N2::AbstractNormal)

Computes the Kullback-Leibler divergence between N1 and N2.
"""
kldivergence(N1::AbstractNormal, N2::AbstractNormal) = kldivergence(N1,N2)

"""
    rand(RNG::AbstractRNG, N::AbstractNormal)

Draws one random vector from N using the random number generator RNG.
"""
rand(RNG::AbstractRNG, N::AbstractNormal) = rand(RNG,N)


"""
    rand(N::AbstractNormal)

Draws one random vector using the random number generator GLOBAL_RNG.
"""
rand(N::AbstractNormal) = rand(N)