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
#smoother_distributions, filter_distributions, prediction_distributions, backward_kernels, loglike = bayes_smoother(ys,init,forward_kernel,measurement_kernel,true)
smoother_distributions, filter_distributions, prediction_distributions, backward_kernels, loglike = bayes_smoother(problem)

# matern estimates
smoother_output_estimate = map( x-> marginalise(x,output_kernel), smoother_distributions )

# plot state
plt_state = plot(ts,xs,layout=(dimx,1), title=["state" "" "" ""], color="black")
plot!(ts,smoother_distributions,layout=(dimx,1),)
display(plt_state)

# plot output estimate
plt_smoother = plot(ts,matern_process,xlabel="t",label="ground-truth",color="red")
scatter!(ts,ys,label="measurement",color="black")
plot!(ts,smoother_output_estimate,label="smoother estimate")
display(plt_smoother)

filter_state = mapreduce(permutedims,vcat,mean.(fs))
smoother_state = mapreduce(permutedims,vcat,mean.(ss))

rmse(r) = sqrt( mean( [LinearAlgebra.norm_sqr(r[i,:]) for i in 1:size(r,1)]  )  )

filter_state_residuals = mapreduce(permutedims,vcat,mean.(fs)) - xs
smoother_state_residuals = mapreduce(permutedims,vcat,mean.(ss)) - xs

filter_output_residuals = mapreduce(permutedims,vcat,mean.(f_est)) - matern_process
smoother_output_residuals = mapreduce(permutedims,vcat,mean.(s_est)) - matern_process

filter_output_rmse = rmse(filter_output_residuals)
smoother_output_rmse = rmse(smoother_output_residuals)

filter_state_rmse = rmse(filter_state_residuals)
smoother_state_rmse = rmse(smoother_state_residuals)

# smoother is suspicious
display("filter output error: $(filter_output_rmse)")
display("smoother output error: $(smoother_output_rmse)")

display("filter state error: $(filter_state_rmse)")
display("smoother state error: $(smoother_state_rmse)")

#plt_state = plot(fs)
#display(plt_state)