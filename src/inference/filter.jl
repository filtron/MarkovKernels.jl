

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


function filter(init::AbstractNormal,forwardks::AbstractVector{U},likelihoods::AbstractVector{V}) where {U<:AbstractNormalKernel,V<:Likelihood{AbstractNormalKernel}}


end

function filter(init::AbstractDistribution,forwardks::AbstractVector{U},likelihoods::AbstractVector{V}) where {U<:AbstractMarkovKernel,V<:AbstractLikelihood}


    for n in 1:N

        predict(forwardks[n],output)

        update(likelihoods[n],output)

    end


    return output

end