"""
    kalman_smoother(
        ys::AbstractVecOrMat,
        init::AbstractNormal,
        fw_kernel::AbstractNormalKernel,
        m_kernel::AbstractNormalKernel,
    )

Computes the smoothing, filtering distributions and loglikelihood
in the following state-space model

x_1 ∼ init
x_m | x_{m-1} ∼ fw_kernel(·, x_{m-1})
y_m | x_m ∼ m_kernel(·, x_m)

for the masurements y_1, ..., y_n.

"""

function rts(
    ys::AbstractVecOrMat,
    init::AbstractNormal,
    fw_kernel::AbstractNormalKernel,
    m_kernel::AbstractNormalKernel,
)
    # run a Kalman filter
    filter_distributions, loglike = kalman_filter(ys, init, fw_kernel, m_kernel)

    # compute the backward kenrles for the Rauch-Tung-Striebel recursion
    bw_kernels = NormalKernel[]
    for m in 1:length(filter_distributions)-1
        pred, bw_kernel = invert(filter_distributions[m], fw_kernel)
        push!(bw_kernels, bw_kernel)
    end

    # compute the smoother estimates
    smoother_distributions = Normal[]
    smoother_distribution = filter_distributions[end]
    pushfirst!(smoother_distributions, smoother_distribution)
    for m in length(filter_distributions)-1:-1:1
        smoother_distribution = marginalize(smoother_distribution, bw_kernels[m])
        pushfirst!(smoother_distributions, smoother_distribution)
    end

    return smoother_distributions, filter_distributions, loglike
end
