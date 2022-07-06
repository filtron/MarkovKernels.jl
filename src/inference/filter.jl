

function Base.filter(problem::AbstractStateEstimationProblem)

    # initialise (should probably be done by initialise(::AbstractStateEstimationProblem))
    filter_distributions = AbstractDistribution[]
    prediction_distributions = AbstractDistribution[]
    backward_kernels = AbstractMarkovKernel[]
    filter_distribution = initial_distribution(problem)
    loglike = 0.0

    # loop variable should not be called stuff ...
    for stuff in problem

        fw_kernel, likelihood = stuff

        # prediction step
        if !isnothing(fw_kernel)

            filter_distribution, bw_kernel = predict(filter_distribution,fw_kernel)
            push!(backward_kernels,bw_kernel)

        end

        # update step
        if !isnothing(likelihood)
            filter_distribution, prediction_distribution, loglike_increment = update(filter_distribution,likelihood)
            push!(filter_distributions,filter_distribution)
            push!(prediction_distributions,prediction_distribution)
            loglike = loglike + loglike_increment
        end

    end

    # should probably be organised in ::AbstractFilterOutput ?
    return filter_distributions, backward_kernels, prediction_distributions, loglike

end

# simple filter for homogeneous Markov process
function Base.filter(ys,init::AbstractDistribution,fw_kernel::AbstractMarkovKernel,m_kernel::AbstractMarkovKernel,aligned::Bool)

    N = size(ys,1)

    # initialise recursion
    filter_distributions = AbstractDistribution[]
    prediction_distributions = AbstractDistribution[]
    backward_kernels = AbstractMarkovKernel[]
    f = init
    loglike = 0.0

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