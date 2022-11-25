
function initialize_particle_filter(
    rng::AbstractRNG,
    y::AbstractVector,
    init::AbstractDistribution,
    m_kernel::AbstractMarkovKernel,
    P::Int,
)
    L = LogLike(m_kernel, y)
    xs = [[rand(rng, init)] for p in 1:P]
    logws = L.(last.(xs))
    logws = logws .- maximum(logws)
    ws = exp.(logws)
    ws = ws / sum(ws)

    #return Mixture(ws, Dirac.(xs))
    return rand(rng, MultinomialResampler(), Mixture(ws, Dirac.(xs)))
end

function particle_filter(
    rng::AbstractRNG,
    ys::AbstractVecOrMat,
    init::AbstractDistribution,
    fw_kernel::AbstractMarkovKernel,
    m_kernel::AbstractMarkovKernel,
    P,
)
    n = size(ys, 1)

    particles = initialize_particle_filter(rng, ys[1, :], init, m_kernel, P)

    logws = similar(weights(particles))

    for m in 2:n

        # create measurement model
        y = ys[m, :]
        L = LogLike(m_kernel, y)

        # bootstrap proposal
        logws[:] = log.(weights(particles))
        for p in 1:P
            xcurr = last(components(particles)[p].μ)
            xnew = rand(rng, fw_kernel, xcurr)
            push!(components(particles)[p].μ, xnew)

            logws[p] = weights(particles)[p] + L(xnew)

            #=
            logws[:] = log.(weights(particles))
            mcurr = copy(mean(components(particles)[p]))
            mcont = rand(rng, fw_kernel, last(mcurr))

            m = push!(mcurr, mcont)
            dists[p] = Dirac(m)
            =#

        end
        logws = logws .- maximum(logws)
        ws = exp.(logws)
        ws = ws / sum(ws)
        #normalize weights
        particles.weights[:] .= ws

        particles = rand(rng, MultinomialResampler(), particles)
    end

    return particles
end
