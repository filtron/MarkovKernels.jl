
abstract type AbstractFilterOutput end

mutable struct FilterOutput{VD,VD,VK,L} <: AbstractFilterOutput
    filter_distributions::VD
    prediction_distributions::VD
    backward_kernels::VK
    loglikelihood::L
    function FilterOutput(fd::VD,pd::VD,bk::VK,ll::Number) where {VD<:AbstractVector{<:AbstractDistribution},VK<:AbstractVector{<:AbstractMarkovKernel},L<:Number}
        new{typeof(fd),typeof(pd),typeof(bk),typeof(ll)}(fd,pd,bk,ll)
    end
end

Base.iterate(fo::FilterOutput) = (fo.filter_distributions,Val(:prediction_distributions))
Base.iterate(fo::FilterOutput, ::Val{:prediction_distributions}) = (fo.prediction_distributions,Val(:backward_kernels))
Base.iterate(fo::FilterOutput, ::Val{:backward_kernels}) = (fo.backward_kernels,Val(:loglikelihood))
Base.iterate(fo::FilterOutput, ::Val{:loglikelihood}) = (fo.loglikelihood,Val(:done))
Base.iterate(fo::FilterOutput, ::Val{:done}) = nothing

add_filter_distribution!(fo::FilterOutput,d::AbstractDistribution) = push!(fo.filter_distributions,d)
add_prediction_distribution!(fo::FilterOutput,d::AbstractDistribution) = push!(fo.prediction_distributions,d)
add_backward_kernel!(fo::FilterOutput,k::AbstractMarkovKernel) = push!(fo.backward_kernels,k)
update_loglikelihood!(fo::FilterOutput,inc::Number) = fo.loglikelihood += inc


initialise_filter() = FilterOutput(AbstractDistribution[],AbstractDistribution[],AbstractMarkovKernel[],0.0)


# repeated code in filters needs to be fixed...
function bayes_filter(problem::AbstractStateEstimationProblem)

    filter_output = initialise_filter()
    filter_distribution = initial_distribution(problem)

    # loop variable should not be called stuff ...
    for stuff in problem

        fw_kernel, likelihood = stuff

        # think this will be needed
        #isnothing(likelihood) ? push!(filter_distributions,filter_distribution) : nothing

        # prediction step
        if !isnothing(fw_kernel)
            filter_distribution, bw_kernel = predict(filter_distribution,fw_kernel)
            add_backward_kernel!(filter_output,bw_kernel)
        end

        # update step
        if !isnothing(likelihood)
            filter_distribution, prediction_distribution, loglike_increment = update(filter_distribution,likelihood)
            add_filter_distribution!(filter_output,filter_distribution)
            add_prediction_distribution!(filter_output,prediction_distribution)
            update_loglikelihood!(filter_output,loglike_increment)
        end

    end

    return filter_output
end

# simple filter for homogeneous Markov process
function bayes_filter(ys,init::AbstractDistribution,fw_kernel::AbstractMarkovKernel,m_kernel::AbstractMarkovKernel,aligned::Bool)

    N = size(ys,1)

    # initialise recursion
    filter_output = initialise_filter()
    filter_distribution = init

    if aligned

        # create measurement model
        y = ys[1,:]
        likelihood = Likelihood(m_kernel,y)

        # measurement update
        filter_distribution, prediction_distribution, loglike_increment = update(filter_distribution,likelihood)
        add_filter_distribution!(filter_output,filter_distribution)
        add_prediction_distribution!(filter_output,prediction_distribution)
        update_loglikelihood!(filter_output,loglike_increment)

    end

    for n in 2:N

        # predict
        filter_distribution, bw_kernel = predict(filter_distribution,fw_kernel)
        add_backward_kernel!(filter_output,bw_kernel)

        # create measurement model
        @inbounds y = ys[n,:]
        likelihood = Likelihood(m_kernel,y)

        # measurement update
        filter_distribution, prediction_distribution, loglike_increment = update(filter_distribution,likelihood)
        add_filter_distribution!(filter_output,filter_distribution)
        add_prediction_distribution!(filter_output,prediction_distribution)
        update_loglikelihood!(filter_output,loglike_increment)
    end

    return filter_output

end