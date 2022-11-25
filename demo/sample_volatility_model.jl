using MarkovKernels
using Plots, Random

rng = MersenneTwister(1991)

include("sampling_implementation.jl")
# time grid
m = 25
T = 5
ts = collect(LinRange(0, T, m))
dt = T / (m - 1)

# define transtion kernel
λ = 1.0
Φ = exp(-λ * dt) .* [1.0 0.0; -2*λ*dt 1.0]
Q = I - exp(-2 * λ * dt) .* [1.0 -2*λ*dt; -2*λ*dt 1+(2*λ*dt)^2]
fw_kernel = NormalKernel(Φ, Q)

# initial distribution
init = Normal(zeros(2), 1.0I(2))

# sample state
xs = sample(rng, init, fw_kernel, m - 1)

# output kernel
σ = 1.0
C = σ / sqrt(2) * [1.0 -1.0]

output_kernel = DiracKernel(C)

variance(x) = fill(exp.(x)[1], 1, 1)
m_kernel = compose(NormalKernel(zeros(1, 1), variance), output_kernel)
#m_kernel = compose(NormalKernel(ones(1, 1), fill(0.1,1,1)), output_kernel)

# sample output
outs = mapreduce(z -> rand(rng, output_kernel, xs[z, :]), vcat, 1:m)
ys = mapreduce(z -> rand(rng, m_kernel, xs[z, :]), vcat, 1:m)

state_plt = plot(
    ts,
    xs,
    layout = (2, 1),
    xlabel = "t",
    labels = ["x1" "x2"],
    title = ["Latent Gauss-Markov process" ""],
)
display(state_plt)

output_plot = plot(ts, outs, label = "output", xlabel = "t")
display(output_plot)

mplot = scatter(ts, ys, label = "measurement", color = "black")
display(mplot)

stdplot = plot(ts, exp.(outs / 2.0))


P = 500

function initialize_particle_filter(rng::AbstractRNG, y::AbstractVector, init::AbstractDistribution, m_kernel::AbstractMarkovKernel, P::Int)

    L = LogLike(m_kernel, y)
    xs = [[rand(rng, init)] for p in 1:P]
    logws = L.(last.(xs))
    logws = logws .- maximum(logws)
    ws = exp.(logws) 
    ws = ws / sum(ws)

    return  Mixture(ws, Dirac.(xs))

end

function particle_filter(rng::AbstractRNG, ys::AbstractVecOrMat, init::AbstractDistribution, fw_kernel::AbstractMarkovKernel, m_kernel::AbstractMarkovKernel, P)

    n = size(ys, 1)

    particles = initialize_particle_filter(rng, ys[1,:], init, m_kernel, P)

    logws = similar(weights(particles))
    for m in 2:n

        # create measurement model
        y = ys[m, :]
        L = LogLike(m_kernel, y)

        # bootstrap proposal

        for p in 1:P

            logws[:] = log.(weights(particles))
            xcurr = last(components(particles)[p].μ)
            xnew = rand(rng, fw_kernel, xcurr)
            push!(components(particles)[p].μ, xnew)
            
            logws[p] = weights(particles)[p] + L(xnew)
        end
        logws = logws .- maximum(logws)
        ws = exp.(logws) 
        ws = ws / sum(ws)

        #normalize weights
        particles.weights[:] .= ws

       #particles = rand(rng, MultinomialResampler(), particles)
       #rand!(particles, rng, MultinomialResampler())

    end

    return particles
end


particles = particle_filter(rng, ys, init, fw_kernel, m_kernel, P)

state_plt2 = plot(
    ts,
    xs,
    layout = (2, 1),
    xlabel = "t",
    labels = ["x1" "x2"],
    title = ["Latent Gauss-Markov process" ""]
)
plot!(ts, 
    mapreduce(permutedims,
    vcat,
    mean(particles)),
    color = "black"
    )
for p in 1:P
    plot!(ts,
        mapreduce(permutedims, vcat, mean(components(particles)[p])),
        color = "black",
        alpha =0.1,
        label = ""
        )
end
display(state_plt2)

output_plot2 = plot(ts, outs, label = "output", xlabel = "t")
plot!(ts,
    mapreduce(permutedims, vcat, mean(particles))*C',
    color = "black"
    )
for p in 1:P
    plot!(ts,
        mapreduce(permutedims, vcat, mean(components(particles)[p]))*C',
        color = "black",
        alpha = 0.2,
        label = "")
end

display(output_plot2)