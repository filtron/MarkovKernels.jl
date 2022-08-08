
function kalman_filter(ys::AbstractVecOrMat,init::AbstractNormal,fw_kernel::AbstractNormalKernel,m_kernel::AbstractNormalKernel,aligned::Bool)

    N = size(ys,1)

    # initialise recursion
    filter_distribution = init
    filter_distributions = AbstractNormal[]      # filtering distributions
    prediction_distributions = AbstractNormal[]  # one step-ahead measurement predictions
    backward_kernels = AbstractNormalKernel[]    # backward kernels (used for rts smoothing)
    loglike = 0.0

    if aligned

        # create measurement model
        y = ys[1,:]
        likelihood = Likelihood(m_kernel,y)

        # measurement update
        filter_distribution, pred_distribution, loglike_increment = update(filter_distribution,likelihood)

        push!(prediction_distributions,pred_distribution)
        loglike = loglike + loglike_increment
    end

    push!(filter_distributions,filter_distribution)

    for n in 2:N

        # predict
        filter_distribution, bw_kernel = predict(filter_distribution,fw_kernel)
        push!(backward_kernels,bw_kernel)

        # create measurement model
        y = ys[n,:]
        likelihood = Likelihood(m_kernel,y)

        # measurement update
        filter_distribution, pred_distribution, loglike_increment = update(filter_distribution,likelihood)

        push!(filter_distributions,filter_distribution)
        push!(prediction_distributions,pred_distribution)
        loglike = loglike + loglike_increment
    end

    return filter_distributions, prediction_distributions, backward_kernels, loglike

end

function predict(N::AbstractNormal,K::NormalKernel{T,U,V}) where {T,U<:AbstractAffineMap,V}

    N_new, B = invert(N,K)

    return N_new, B

end

function update(N::AbstractNormal,L::Likelihood{NormalKernel{U,V,S},YT}) where {U,V<:AbstractAffineMap,S,YT}

    M, C = invert(N, measurement_model(L) )
    y = measurement(L)
    N_new = condition(C, y )
    loglike = logpdf(M, y )

    return N_new, M, loglike

end