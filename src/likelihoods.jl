abstract type AbstractLikelihood end


struct Likelihood{U<:AbstractMarkovKernel,V} <: AbstractLikelihood
    K::U
    y::V
end

struct FlatLikelihood <: AbstractLikelihood end