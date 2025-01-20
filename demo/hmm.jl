import Pkg
Base.active_project() != joinpath(@__DIR__, "Project.toml") && Pkg.activate(@__DIR__)
haskey(Pkg.project().dependencies, "MarkovKernels") ||
    Pkg.develop(path = joinpath(@__DIR__, "../"))
isfile(joinpath(@__DIR__, "Manifest.toml")) && Pkg.resolve()
Pkg.instantiate()

using MarkovKernels
using LinearAlgebra, Random
using Plots

function sample(rng::AbstractRNG, init, fw_kernels)
    x = rand(rng, init)
    n = length(fw_kernels) + 1
    xs = Vector{typeof(x)}(undef, n)
    xs[begin] = x

    for (m, fw_kernel) in pairs(fw_kernels)
        x = rand(rng, fw_kernel, x)
        xs[begin+m] = x
    end
    return xs
end

function sample(rng::AbstractRNG, init, fw_kernels, obs_kernels)
    # sample initial values
    x = rand(rng, init)
    y = rand(rng, first(obs_kernels), x)

    # allocate output
    n = length(obs_kernels)
    xs = Vector{typeof(x)}(undef, n)
    ys = Vector{typeof(y)}(undef, n)

    xs[begin] = x
    ys[begin] = y

    for (m, fw_kernel) in pairs(fw_kernels)
        obs_kernel = obs_kernels[begin+m]
        x = rand(rng, fw_kernel, x)
        y = rand(rng, obs_kernel, x)
        xs[begin+m] = x
        ys[begin+m] = y
    end
    return xs, ys
end

m, n = 10, 10
init = MarkovKernels.Categorical(ones(m))
Pxx = Matrix(Tridiagonal(ones(m - 1), 5 * ones(m), ones(m - 1)))
Kxx = StochasticMatrix(Pxx)

Pyx = (ones(m, m) - I)
Kyx = StochasticMatrix(Pyx)

T = 2^8 + 1
fw_kernels = fill(Kxx, T - 1)
obs_kernels = fill(Kyx, T)

rng = Random.Xoshiro(19910215)
xs, ys = sample(rng, init, fw_kernels, obs_kernels)

hmm_plt = Plots.scatter(
    eachindex(xs),
    xs,
    color = "black",
    title = "hidden state",
    layout = (1, 2),
)
Plots.scatter!(
    hmm_plt,
    eachindex(ys),
    ys,
    color = "red",
    title = "observations",
    subplot = 2,
)

function backward_recursion(init, forward_kernels, likelihoods)
    h = last(likelihoods)
    KT = Base.promote_op(first ∘ htransform, eltype(forward_kernels), typeof(h))
    post_forward_kernels = Vector{KT}(undef, length(forward_kernels))

    for m in eachindex(forward_kernels)
        fw_kernel = forward_kernels[end-m+1]
        post_fw_kernel, h = htransform(fw_kernel, h)
        post_forward_kernels[end-m+1] = post_fw_kernel

        like = likelihoods[end-m]
        h = compose(h, like)
    end
    post_init, loglike = posterior_and_loglike(init, h)
    return post_init, post_forward_kernels, loglike
end

likes = [Likelihood(Kobs, y) for (Kobs, y) in zip(obs_kernels, ys)]
post_init, post_fw_kernels = backward_recursion(init, fw_kernels, likes)

let xs
    nsample = 10
    for _ in 1:nsample
        xs = sample(rng, post_init, post_fw_kernels)
        Plots.scatter!(hmm_plt, eachindex(xs), xs, label = "", color = "blue", alpha = 0.1)
    end
end
display(hmm_plt)

#=
fig = Figure()
axstate = Axis(fig[1, 1], title = "state sequence", xlabel = "n", ylabel = "x")
scatter!(axstate, eachindex(xs), xs, color = "black")
axobs = Axis(fig[1, 2], title = "observartion sequence", xlabel = "n", ylabel = "y")
scatter!(axobs, eachindex(ys), ys, color = "black")
display(fig)

function forward_backward_recursion(init_dist, fw_kernels, likes)

    # initial filter update
    filt = posterior(init_dist, first(likes))

    # allocate output
    bw_type = Base.promote_op(last ∘ invert, typeof(filt), eltype(fw_kernels)) # assumes all elements of fw_kernels are of same type
    bw_kernels = Vector{bw_type}(undef, length(fw_kernels))

    for (m, fw_kernel) in pairs(fw_kernels)
        like = likes[begin+m]
        pred, bw_kernel = invert(filt, fw_kernel)
        filt = posterior(pred, like)
        bw_kernels[m] = bw_kernel
    end

    term = filt
    return bw_kernels, term
end

function fb_posterior_marginals(init_dist, fw_kernels, likes)
    bw_kernels, term = forward_backward_recursion(init_dist, fw_kernels, likes)
    n = length(bw_kernels)
    marginals = Vector{typeof(term)}(undef, n + 1)

    marginal = term
    marginals[end] = marginal
    for (m, bw_kernel) in pairs(bw_kernels)
        marginal = marginalize(marginal, bw_kernel)
        marginals[end-m] = marginal
    end
    return marginals
end

likes = [Likelihood(Kobs, y) for (Kobs, y) in zip(obs_kernels, ys)]
bw_kernels, term = forward_backward_recursion(init, fw_kernels, likes)

fig = Figure()
ax = Axis(fig[1, 1], title = "state sequence", xlabel = "n", ylabel = "x")
scatter!(eachindex(xs), xs, color = "black", alpha = 0.5)
nsample = 10
for _ in 1:nsample
    xs_post = sample(rng, term, reverse(bw_kernels))
    xs_post = reverse(xs_post)
    scatter!(ax, eachindex(xs_post), xs_post, color = "red", alpha = 0.05)
end
display(fig)
=#
