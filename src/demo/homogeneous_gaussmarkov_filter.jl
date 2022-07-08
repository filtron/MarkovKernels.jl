using MarkovKernels
using SpecialFunctions
using Plots

# plotting backend
# pgfplotsx()
gr()

include("matern2ssm.jl")
include("lti_disc.jl")

# make time span and stamps
T = 10
N = 2^8+1
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


# define state estimation problem
problem = HomogeneousStateEstimationProblem(ys,init,forward_kernel,measurement_kernel,true)

# state estimates
#filter_output = bayes_filter(ys,init,forward_kernel,measurement_kernel,true)
filter_output = bayes_filter(problem)
filter_distributions, prediction_distributions, backward_kernels, loglikelihood = filter_output

# compute measurement residuals
residuals = mapreduce(residual,vcat,prediction_distributions,ys)

# matern estimates
filter_output_estimate = map( x-> marginalise(x,output_kernel), filter_distributions )

# plot state
plt_state = plot(ts,xs,layout=(dimx,1), title=["state" "" "" ""], color="black")
plot!(ts,filter_distributions,layout=(dimx,1),)
display(plt_state)

# plot output estimate
plt_filter = plot(ts,matern_process,xlabel="t",label="ground-truth",color="red")
scatter!(ts,ys,label="measurement",color="black")
plot!(ts,filter_output_estimate,label="filter estimate")
plot!(ts,prediction_distributions,label="one-step ahead prediction")
display(plt_filter)

# plot residuals
plt_residuals = scatter(ts,residuals,color="black")
display(plt_residuals)
