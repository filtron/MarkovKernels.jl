

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
function filter(ys,init::AbstractDistribution,fw_kernel::AbstractMarkovKernel,m_kernel::AbstractMarkovKernel,aligned::Bool)

    N = size(ys,1)

    # initialise recursion
    filter_distributions = AbstractDistribution[]
    prediction_distributions = AbstractDistribution[]
    backward_kernels = AbstractMarkovKernel[]
    f = init
    loglike = 0

    if aligned

        # create measurement model
        y = ys[1,:]
        likelihood = Likelihood(m_kernel,y)

        # measurement update
        f, pred, loglike_increment = update(f,likelihood)

        push!(filter_distributions,f)
        push!(prediction_distributions,pred)
        loglike = loglike + loglike_increment

    end

    for n in 2:N

        # predict
        p, b =  predict(f,fw_kernel)
        push!(backward_kernels,b)

        # create measurement model
        @inbounds y = ys[n,:]
        likelihood = Likelihood(m_kernel,y)

        # measurement update
        f, pred, loglike_increment = update(p,likelihood)

        push!(filter_distributions,f)
        push!(prediction_distributions,pred)
        loglike = loglike + loglike_increment
    end

    return filter_distributions, backward_kernels, prediction_distributions, loglike

end


# simple smoother for homogeneous Markov process
function smoother(ys,init::AbstractDistribution,fw_kernel::AbstractMarkovKernel,m_kernel::AbstractMarkovKernel,aligned::Bool)

    filter_distributions, backward_kernels, prediction_distributions, loglike = filter(ys,init,fw_kernel,m_kernel,aligned)

    N = length(backward_kernels)
    s = filter_distributions[end]
    smoother_distributions = AbstractDistribution[]

    pushfirst!(smoother_distributions,s)

    for n=0:N-1

        bw = backward_kernels[N-n]

        s = marginalise(s,bw)
        pushfirst!(smoother_distributions,s)

    end

    return smoother_distributions, filter_distributions, backward_kernels, prediction_distributions, loglike

end