
"""
    mean(K::AbstractNormalKernel)

Returns the conditional mean function of the Normal kernel K.
"""
mean(K::AbstractNormalKernel) = mean(K)

"""
    cov(K::AbstractNormalKernel)

Returns the conditional covariance matrix of the Normal kernel K.
"""
cov(K::AbstractNormalKernel) = cov(K)

"""
    condition(K::AbstractNormalKernel,x)

Returns a Normal distribution corresponding to K evaluated at x.
"""
condition(K::AbstractNormalKernel, x) = condition(K, x)
