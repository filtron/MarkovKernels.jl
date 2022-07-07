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

plt_state = plot(ts,xs)
display(plt_state)

# measurement covariance matrix
R = fill(0.01,1,1)

# measure a Matern process of smoothness ν
matern_process = rand(output_kernel,xs)

# noisy measurements of Matern process
measurement_kernel = NormalKernel(Matrix(C'),R)
ys = rand(measurement_kernel,xs)

plt_matern = plot(ts,matern_process)
scatter!(ts,ys,color="black")
display(plt_matern)

# define state estimation problem
problem = HomogeneousStateEstimationProblem(ys,init,forward_kernel,measurement_kernel,true)

# state estimates
ss, fs, bws, mls, loglike = smoother(ys,init,forward_kernel,measurement_kernel,true)
#ss, fs, bws, mls, loglike = smoother(problem)

# compute measurement residuals
residuals = mapreduce(residual,vcat,mls,ys)

# matern estimates
f_est = map( x-> marginalise(x,output_kernel), fs )
s_est = map( x-> marginalise(x,output_kernel), ss )

# plot filter estimate
plt_filter = plot(ts,matern_process,xlabel="t",label="ground-truth")
plot!(ts,f_est,label="filter estimate")
display(plt_filter)

# plot one-step ahead
plt_pred = plot(ts,matern_process,xlabel="t",label="ground-truth")
plot!(ts,mls,label="one-step ahead prediction")
display(plt_pred)

# plot prediction errors
plt_pe = scatter(ts,residuals,color="black")
display(plt_pe)

# plot smoother estimate
plt_smoother = plot(ts,matern_process,xlabel="t",label="ground-truth")
plot!(ts,s_est,label="smoother estimate")
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