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

function kalman_smoother(
    ys::AbstractVecOrMat,
    init::AbstractNormal,
    fw_kernel::AbstractNormalKernel,
    m_kernel::AbstractNormalKernel,
)
    filter_distributions, loglike = kalman_filter(ys, init, fw_kernel, m_kernel)

    bw_kernels = NormalKernel[]

    for m in 1:length(filter_distributions)-1
        pred, bw_kernel = invert(filter_distributions[m], fw_kernel)
        push!(bw_kernels, bw_kernel)
    end

    smoother_distributions = Normal[]
    pushfirst!(smoother_distributions, filter_distributions[end])

    for m in length(filter_distributions)-1:-1:1
        smoother_distribution = marginalise(filter_distributions[m+1], bw_kernels[m])
        pushfirst!(smoother_distributions, smoother_distribution)
    end

    return smoother_distributions, filter_distributions, loglike
end
