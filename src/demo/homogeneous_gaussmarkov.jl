using MarkovKernels
using SpecialFunctions
using Plots

# plotting backend
pgfplotsx()
#gr()

include("matern2ssm.jl")
include("lti_disc.jl")


# make time span and stamps
T = 10
N = 2^8+1
dt = T/(N-1)
ts = 0:dt:(N-1)*dt

# define stationary Gauss-Markov process
ν = 2
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
R = fill(0.005,1,1)

# measure a Matern process of smoothness ν
matern_process = rand(output_kernel,xs)

# noisy measurements of Matern process
measurement_kernel = NormalKernel(Matrix(C'),R)
ys = rand(measurement_kernel,xs)

# state estimates
#fs, bws, mls, loglike = filtering(ys2,init,forward_kernel,measurement_kernel,true)
ss, fs, bws, mls, loglike = smoother(ys,init,forward_kernel,measurement_kernel,true)

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

# plot smoother estimate
plt_smoother = plot(ts,matern_process,xlabel="t",label="ground-truth")
plot!(ts,s_est,label="smoother estimate")
display(plt_smoother)


#plt_state = plot(fs)
#display(plt_state)