function bootstrap_smoother(
    rng::AbstractRNG,
    ys::AbstractVecOrMat,
    init::AbstractDistribution,
    fw_kernel::AbstractMarkovKernel,
    m_kernel::AbstractMarkovKernel,
    K::Integer,
)

    # initialize
    X = permutedims([rand(rng, init) for k in 1:K])
    P = ParticleSystem(zeros(K), X)
    loglike = 0.0
    L = Likelihood(m_kernel, ys[1, :])
    loglike_incr = bayes_rule!(P, L)
    loglike = loglike + loglike_incr
    resample!(rng, P)

    for m in 2:size(ys, 1)
        L = Likelihood(m_kernel, ys[m, :])

        P = predict(rng, P, fw_kernel)
        loglike_incr = bayes_rule!(P, L)

        loglike = loglike + loglike_incr
        resample!(rng, P)
    end

    return P, loglike
end

function resample!(rng::AbstractRNG, P::ParticleSystem{T,U,<:AbstractMatrix}) where {T,U}
    idx = wsample(rng, eachindex(logweights(P)), weights(P), nparticles(P))
    logweights(P)[:] .= zero(logweights(P))
    particles(P)[:, :] .= particles(P)[:, idx]
end

function predict(
    rng::AbstractRNG,
    P::ParticleSystem{T,U,<:AbstractMatrix},
    K::AbstractMarkovKernel,
) where {T,U}
    X = [rand(rng, K, particles(P)[end, i]) for i in 1:nparticles(P)]

    return ParticleSystem(logweights(P), vcat(particles(P), permutedims(X)))
end
