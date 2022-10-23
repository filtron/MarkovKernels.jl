"""
    kalman_filter(
        ys::AbstractVecOrMat,
        init::AbstractNormal,
        fw_kernel::AbstractNormalKernel,
        m_kernel::AbstractNormalKernel,
    )

Computes the filtering distributions and loglikelihood 
in the following state-space model 

x_1 ∼ init 
x_m | x_{m-1} ∼ fw_kernel(·, x_{m-1})
y_m | x_m ∼ m_kernel(·, x_m)

for the masurements y_1, ..., y_n. 

"""
function kalman_filter(
    ys::AbstractVecOrMat,
    init::AbstractNormal,
    fw_kernel::AbstractNormalKernel,
    m_kernel::AbstractNormalKernel,
)
    n = size(ys, 1)

    # initialise recursion
    filter_distribution = init
    filter_distributions = Normal[]      # filtering distributions

    # create measurement model
    y = ys[1, :]
    likelihood = LogLike(m_kernel, y)

    # measurement update
    filter_distribution, loglike_increment = bayes_rule(filter_distribution, likelihood)
    push!(filter_distributions, filter_distribution)
    loglike = loglike_increment

    for m in 2:n

        # predict
        filter_distribution = marginalise(filter_distribution, fw_kernel)

        # create measurement model
        y = ys[m, :]
        likelihood = LogLike(m_kernel, y)

        # measurement update
        filter_distribution, loglike_increment = bayes_rule(filter_distribution, likelihood)

        push!(filter_distributions, filter_distribution)
        loglike = loglike + loglike_increment
    end

    return filter_distributions, loglike
end
