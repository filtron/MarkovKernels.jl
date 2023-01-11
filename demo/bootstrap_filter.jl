import StatsBase

function bootstrap_filter(
    rng::AbstractRNG,
    ys::AbstractVecOrMat,
    init::AbstractDistribution,
    fw_kernel::AbstractMarkovKernel,
    m_kernel::AbstractMarkovKernel,
    K::Integer,
)

    # initialize
    X = [rand(rng, init) for k in 1:K]
    P = ParticleSystem(zeros(K), X)
    loglike = 0.0
    L = LogLike(m_kernel, ys[1, :])
    loglike_incr = bayes_rule!(P, L)
    loglike = loglike + loglike_incr
    resample!(rng, P)

    Ps = [P]
    sizehint!(Ps, size(ys, 1))

    for m in 2:size(ys, 1)
        L = LogLike(m_kernel, ys[m, :])
        P = predict(rng, P, fw_kernel)
        loglike_incr = bayes_rule!(P, L)
        loglike = loglike + loglike_incr
        resample!(rng, P)
        push!(Ps, P)
    end

    return Ps, loglike
end

function resample!(rng::AbstractRNG, P::ParticleSystem{T,U,<:AbstractVector}) where {T,U}
    idx = StatsBase.wsample(rng, eachindex(logweights(P)), weights(P), nparticles(P))
    logweights(P)[:] .= zero(logweights(P))
    particles(P)[:] .= particles(P)[idx]
end

function predict(
    rng::AbstractRNG,
    P::ParticleSystem{T,U,<:AbstractVector},
    K::AbstractMarkovKernel,
) where {T,U}
    X = copy.(particles(P))

    for i in eachindex(X)
        X[i][:] .= rand(rng, K, X[i])
    end

    return ParticleSystem(copy(logweights(P)), X)
end
