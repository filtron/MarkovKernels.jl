using MarkovKernels
using SpecialFunctions
using Plots

# plotting backend
# pgfplotsx()
gr()

# convert matern models of half integer index into continuous-time state-space models
include("matern2ssm.jl")

# discretise continuous-time state-space model
include("lti_disc.jl")

# implementation of kalman filter for homogeneous problems
include("kalman_filter.jl")

# make time span and stamps
T = 10
N = 2^6+1
dt = T/(N-1)
ts = 0:dt:(N-1)*dt

# define stationary Gauss-Markov process
ν = 3
λ = 2.5
σ = 1

# state dimension
dimx = ν+1

# stationary distribution of state
init = Normal( zeros(dimx),Matrix(1.0*I,dimx,dimx) )

# continuous-time transition parameters
A, B, C = matern2ssm(ν,λ,σ)

# discrete-time transition parameters
Φ, Q  = lti_disc(A,B,dt)

# transition density
forward_kernel = NormalKernel(Φ,Matrix(Q))

# map state to matern process
output_kernel = DiracKernel(Matrix(C'))

# sample Gauss-Markov process
xs = rand(init,forward_kernel,N-1)

# measurement covariance matrix
R = fill(0.01,1,1)

# measure a Matern process of smoothness ν
matern_process = rand(output_kernel,xs)

# noisy measurements of Matern process
measurement_kernel = NormalKernel(Matrix(C'),R)
ys = rand(measurement_kernel,xs)

# kalman filtering
filter_distributions, prediction_distributions, backward_kernels, loglikelihood = kalman_filter(ys,init,forward_kernel,measurement_kernel,true)

# compute measurement residuals
residuals = mapreduce(residual,vcat,prediction_distributions,ys)

# matern filter estimates
filter_output_estimate = map( x-> marginalise(x,output_kernel), filter_distributions )

# plot state filter estimates
plt_state_filter = plot(ts,xs,layout=(dimx,1), title=["state filter estimates" "" "" ""], color="black")
plot!(ts,filter_distributions,layout=(dimx,1),)
display(plt_state_filter)

# plot one-step ahead predictions
plt_pred = plot(ts,matern_process,xlabel="t",label="ground-truth",color="red",title="one-step ahead predictions")
scatter!(ts,ys,label="measurement",color="black")
plot!(ts,prediction_distributions,label="one-step ahead prediction")
display(plt_pred)

# plot residuals
plt_residuals = scatter(ts,residuals,color="black",title="one-step ahead prediction residuals")
display(plt_residuals)

# plot filter estimate
plt_filter = plot(ts,matern_process,xlabel="t",label="ground-truth",color="red",title="filter estimate")
scatter!(ts,ys,label="measurement",color="black")
plot!(ts,filter_output_estimate,label="filter estimate")
display(plt_filter)

# Rauch--Tung--Striebel smoother
function rts_recursion(terminal::AbstractNormal,kernels::AbstractVector{<:AbstractNormalKernel})

    N = length(kernels)
    d = terminal
    distributions = AbstractNormal[]

    pushfirst!(distributions,d)

    for n=0:N-1

        k = kernels[N-n]
        d = marginalise(d,k)
        pushfirst!(distributions,d)

    end

    return distributions

end

terminal = filter_distributions[end]
smoother_distributions = rts_recursion(terminal,backward_kernels)

# matern smoother estimates
smoother_output_estimate = map( x-> marginalise(x,output_kernel), smoother_distributions )

# plot state smoother estimates
plt_state_smoother = plot(ts,xs,layout=(dimx,1), title=["state smoother estimates" "" "" ""], color="black")
plot!(ts,smoother_distributions,layout=(dimx,1),)
display(plt_state_smoother)

# plot smoother estimate
plt_smoother = plot(ts,matern_process,xlabel="t",label="ground-truth",color="red",title="smoother estimate")
scatter!(ts,ys,label="measurement",color="black")
plot!(ts,smoother_output_estimate,label="smoother estimate")
display(plt_smoother)
