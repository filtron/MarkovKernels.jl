using MarkovKernels
using SpecialFunctions
using Plots

# plotting backend
pgfplotsx()

# make time span and stamps
T = 10
N = 2^9+1
dt = T/(N-1)
ts = 0:dt:(N-1)*dt

# define stationary Gauss-Markov process
ν = 2
λ = 5
σ = 1

# state dimension
dimx = ν+1

# stationary distribution
init = Normal( zeros(dimx),Matrix(1.0*I,dimx,dimx) )

# continuous-time transition parameters
A = λ*( I -2*tril(ones(dimx,dimx)) )
B = sqrt(2*λ)*ones(dimx,1)

# discrete-time transition parameters
W = exp([A Hermitian(B*B'); zeros(dimx,dimx) -A']*dt)
Φ = W[1:dimx,1:dimx]
Q = Hermitian(W[1:dimx,dimx+1:2*dimx]*Φ')

# transition density
forward_kernel = NormalKernel(Φ,Q)

# sample Gauss-Markov process
xs = rand(init,forward_kernel,N-1)

# plot latent Gauss-Markov process
pltx = plot(ts,xs)
display(pltx)


# observation matrix
C = σ*exp.( - logfactorial.( collect(0:ν) )
- logfactorial.( collect(ν:-1:0) ) .+ 2*logfactorial(ν) .- logfactorial(2*ν)/2 ).*
(-1).^collect(0:ν)

# measurement covariance matrix
R = fill(0.05,1,1)


# measure a Matern process of smoothness ν
m1 = DiracKernel(Matrix(C'))
ys1 = rand(m1,xs)

# noisy measurements of Matern process
m2 = NormalKernel(Matrix(C'),R)
ys2 = rand(m2,xs)

# plot matern process, woaah!
plty = plot(ts,[ys1 ys2])
display(plty)

# does filtering work?
fs, bws, loglike = filtering(ys2,init,forward_kernel,m2,true)