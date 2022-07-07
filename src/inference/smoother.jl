
function smoother(problem::AbstractStateEstimationProblem)

    # check that problem has length

    filter_distributions, backward_kernels, prediction_distributions, loglike = filter(problem)

    terminal = filter_distributions[end]
    smoother_distributions = _rts_recursion(terminal,backward_kernels)

    return smoother_distributions, filter_distributions, backward_kernels, prediction_distributions, loglike

end


# simple smoother for homogeneous Markov process
function smoother(ys,init::AbstractDistribution,fw_kernel::AbstractMarkovKernel,m_kernel::AbstractMarkovKernel,aligned::Bool)

    # check that problem has length

    filter_distributions, backward_kernels, prediction_distributions, loglike = filter(ys,init,fw_kernel,m_kernel,aligned)

    terminal = filter_distributions[end]
    smoother_distributions = _rts_recursion(terminal,backward_kernels)

    return smoother_distributions, filter_distributions, backward_kernels, prediction_distributions, loglike

end


function _rts_recursion(terminal::AbstractDistribution,kernels)

    N = length(kernels)
    d = terminal
    distributions = AbstractDistribution[]

    pushfirst!(distributions,d)

    for n=0:N-1

        k = kernels[N-n]
        d = marginalise(d,k)   # this needs to call predict instead somehow
        pushfirst!(distributions,d)

    end

    return distributions

end