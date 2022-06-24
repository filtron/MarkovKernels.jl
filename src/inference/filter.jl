

abstract type AbstractFilterOutput end

abstract type AbstractFilterOptions end



function filter(init::AbstractNormal,forwardks::AbstractVector{U},likelihoods::AbstractVector{V}) where {U<:AbstractNormalKernel,V<:Likelihood{AbstractNormalKernel}}


end

function filter(init::AbstractDistribution,forwardks::AbstractVector{U},likelihoods::AbstractVector{V}) where {U<:AbstractMarkovKernel,V<:AbstractLikelihood}


    for n in 1:N

        predict(forwardks[n],output)

        update(likelihoods[n],output)

    end


    return output

end