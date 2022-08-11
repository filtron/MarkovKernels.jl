# Implementing a Kalman filter and a Rauch-Tung-Striebel smoother

This tutorial describes how to implement a Kalman filter for the following state-space model

```math
\begin{aligned}
x_0 &\sim \mathcal{N}(\mu_0 ,\Sigma_0), \\
x_n \mid x_{n-1} &\sim \mathcal{N}(\Phi  x_{n-1}, Q),\\
z_n \mid x_n &\sim \mathcal{N}(Cx_n,R),
\end{aligned}
```
given a measurement sequence $z_{0:N}$.



### Kalman filter implementation
The classical imlementation of a Kalman filter is decomposed into a prediction step and an update step.
These can be implemented as follows.

```@example 2
using MarkovKernels, LinearAlgebra, Plots

function predict(N::AbstractNormal, K::NormalKernel{T,U,V}) where {T,U<:AbstractAffineMap,V}
    N_new, B = invert(N, K)
    return N_new, B
end

function update(
    N::AbstractNormal,
    L::Likelihood{NormalKernel{U,V,S},YT},
) where {U,V<:AbstractAffineMap,S,YT}
    M, C = invert(N, measurement_model(L))
    y = measurement(L)
    N_new = condition(C, y)
    loglike = logpdf(M, y)

    return N_new, M, loglike
end
```

Note that the above implementation of the prediction step also computes the backward kernel required for the Rauch-Tung-Striebel smoother recursion.
If no smoothing is to be performed, an alternative to the prediction step is simply to call marginalise, that is.

```@example
using MarkovKernels # hide
function predict2(N::AbstractNormal, K::NormalKernel{T,U,V}) where {T,U<:AbstractAffineMap,V}
    N_new = marginalise(N,K)
    return N_new
end
```
In any case, this tutorial does indeed demonstrate how to implement the smoother as well.
The Kalman filter may thus be implemented by the following.
```@example 2

function kalman_filter(
    ys::AbstractVecOrMat,
    init::AbstractNormal,
    fw_kernel::AbstractNormalKernel,
    m_kernel::AbstractNormalKernel,
)
    N = size(ys, 1)

    # initialise recursion
    filter_distribution = init
    filter_distributions = AbstractNormal[]      # filtering distributions
    prediction_distributions = AbstractNormal[]  # one step-ahead measurement predictions
    backward_kernels = AbstractNormalKernel[]    # backward kernels (used for rts smoothing)
    loglike = 0.0

    # create measurement model
    y = ys[1, :]
    likelihood = Likelihood(m_kernel, y)

    # measurement update
    filter_distribution, pred_distribution, loglike_increment =
        update(filter_distribution, likelihood)

    push!(prediction_distributions, pred_distribution)
    loglike = loglike + loglike_increment
    push!(filter_distributions, filter_distribution)

    for n in 2:N

        # predict
        filter_distribution, bw_kernel = predict(filter_distribution, fw_kernel)
        push!(backward_kernels, bw_kernel)

        # create measurement model
        y = ys[n, :]
        likelihood = Likelihood(m_kernel, y)

        # measurement update
        filter_distribution, pred_distribution, loglike_increment =
            update(filter_distribution, likelihood)

        push!(filter_distributions, filter_distribution)
        push!(prediction_distributions, pred_distribution)
        loglike = loglike + loglike_increment
    end

    return filter_distributions, prediction_distributions, backward_kernels, loglike
end
```

It remains to generate some data to try it out.

```@example 2
N = 2^6
ns = 0:N

# define a Markov kernel for a homogeneous Markov proces
λ = 0.5

dt = 1.0
dimx = 2
Φ = [1.0 0; dt 1.0]
Q = [dt dt^2/2; dt^2/2 dt^3/3]
forward_kernel = NormalKernel(Φ, Q)

# define initial distribution
init = Normal(zeros(dimx), 10.0*I(dimx))

# sample Gauss-Markov process
xs = rand(init, forward_kernel, N)

# define output process
C = [0.0 1.0]
output_kernel = DiracKernel(C)

# define measurements of output process
R = fill(150.0,1,1)
measurement_kernel = NormalKernel(C,R)

# sample outputs, measurements and plot
output = rand(output_kernel,xs)
zs = rand(measurement_kernel,xs)

filter_distributions, prediction_distributions, backward_kernels, loglike =
        kalman_filter(zs, init, forward_kernel, measurement_kernel)
```

Plotting the state estimates.

```@example 2
plot(
    ns,
    xs,
    layout=(dimx,1),
    xlabel = ["" "t"],
    label = ["x0" "x1"],
    title = ["Filter estimates of the state" ""]
)
plot!(
    ns,
    filter_distributions,
    layout=(dimx,1),
    label = ["x0filter" "x1filter"]
)
```

Plotting one-step ahead predictions of the measurement.

```@example 2
plot(
    ns,
    prediction_distributions,
    xlabel = "t",
    ylabel = "y",
    label = "one-step ahead prediction",
    title = "One-step ahead predictions"
)

scatter!(
    ns,
    zs,
    label = "measurements",
    color="black"
)
```

#### Implementing the Rauch-Tung-Striebel smoother

Given the previous implementation of the Kalman filter, implementing the smoother is straight-forward.
```@example 2
function rts_recursion(
    terminal::AbstractNormal,
    kernels::AbstractVector{<:AbstractNormalKernel},
)
    N = length(kernels)
    d = terminal
    distributions = AbstractNormal[]

    pushfirst!(distributions, d)

    for n in 0:N-1
        k = kernels[N-n]
        d = marginalise(d, k)
        pushfirst!(distributions, d)
    end

    return distributions
end
```

The smoother estimates may thus be calculated from the previous output of the Kalman filter as follows.
```@example 2
smoother_distributions = rts_recursion(filter_distributions[end],backward_kernels)
```

Plotting the smoother estimates of the state.

```@example 2
plot(
    ns,
    xs,
    layout=(dimx,1),
    xlabel = ["" "t"],
    label = ["x0" "x1"],
    title = ["Filter estimates of the state" ""]
)
plot!(
    ns,
    filter_distributions,
    layout=(dimx,1),
    label = ["x0filter" "x1filter"]
)
plot!(
    ns,
    smoother_distributions,
    layout=(dimx,1),
    label = ["x0smoother" "x1smoother"]
)
```
