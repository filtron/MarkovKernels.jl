"""
    sample(
        RNG::AbstractRNG,
        init::AbstractDistribution,
        K::AbstractMarkovKernel,
        nstep::Integer,
    )

samples a trajectory of length nstep + 1 from the Markov model

x_1 ∼ init
x_m | x_{m-1} ∼ fw_kernel(·, x_{m-1})
"""

function sample(
    RNG::AbstractRNG,
    init::AbstractDistribution,
    K::AbstractMarkovKernel,
    nstep::Integer,
)
    x = rand(RNG, init)
    xs = zeros(nstep + 1, length(x))
    xs[1, :] = x

    for n in 1:nstep
        x = rand(RNG, K, x)
        xs[n+1, :] = x
    end

    return xs
end
