

abstract type AbstractFilterOutput end

abstract type AbstractFilterOptions end

struct FilterProblem{U,V,W,B1,B2,B3}
    init::U
    forward_kernels::V
    likelihoods::W
    aligned::B1
    compute_loglike_increments::B2
    compute_backward_kernels::B3
end

# simple filter for homogeneous Markov process
function filtering(ys,init::AbstractDistribution,fw_kernel::AbstractMarkovKernel,m_kernel::AbstractMarkovKernel,aligned::Bool)

    N = size(ys,1)

    # initialise recursion
    filter_distributions = AbstractDistribution[]
    backward_kernels = AbstractMarkovKernel[]
    f = init
    loglike = 0

    if aligned

        # create measurement model
        y = ys[1,:]
        likelihood = Likelihood(m_kernel,y)

        # measurement update
        f, loglike_increment = update(f,likelihood)

        push!(filter_distributions,f)
        loglike = loglike + loglike_increment

    end

    for n in 2:N

        # predict
        p, b =  predict(f,fw_kernel)
        push!(backward_kernels,b)

        # create measurement model
        y = ys[n,:]
        likelihood = Likelihood(m_kernel,y)
        f, loglike_increment = update(p,likelihood)
        push!(filter_distributions,f)
        loglike = loglike + loglike_increment
    end

    return filter_distributions, backward_kernels, loglike

end
