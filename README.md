# MarkovKernels.jl 

A package implementing defining a Distributions, Markov kernels, and likelihoods that all play nice with eachother. 
The main motivation is to simplify the implementation of Bayesian filtering and smoothing algorithms. 
Let $\pi(x)$ is a probability distribution and $k(y\mid x)$ is a Markov kernel then the only the following operations are required for Bayesian state estimation

* Marginalisation: 

$$
k(y) = \int k(y\mid x) \pi(x) \mathrm{d} x, 
$$ 

which gives the prediction step in Bayesian filtering. 

* Inverse factorisation: 

$$
k(y\mid x)\pi(x) = \pi(x \mid y) k(y),  
$$

where evaluation of $\pi(x \mid y)$ at $y$ gives Bayes' rule and $k(y)$ is the marginal distribution of $y$ (used for prediction error decomposition of the marginal likelihood). In fact, the prediction step may be implemented with the inverse factorisation operation as well, in which case $\pi(x\mid y)$ is the backwards kernel used to compute smoothing distributions in the Rauch-Tung-Striebel recursion, see the demo in src/demo. 

[![Build Status](https://github.com/filtron/MarkovKernels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/filtron/MarkovKernels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/filtron/MarkovKernels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/filtron/MarkovKernels.jl)

## Package specific types

* Types for representing marginal distributions, Markov kernels, and likelihoods:

```julia
abstract type AbstractDistribution end
abstract type AbstractMarkovKernel end
abstract type AbstractLikelihood end
```
* In practice, to implement Bayesian state estimation algorithms, it is up to the user to define appropriate prediction / update functions: 

```julia
predict(D::AbstractDistribution,K::AbstractMarkovKernel)
update(D::AbstractDistribution,L::AbstractLikelihood)
```




### Normal distributions 

* Mathematical definition: 

$$
\pi(x) = \mathcal{N}(x ; \mu  , \Sigma ),  
$$

the explicit expression for the probability density function  depends on whether $x$ takes valued in real-valued or complex-valued Euclidean space. 

* Types:

```julia
abstract type AbstractNormal{T<:Number}  <: AbstractDistribution end # normal distributions with realisations in real / complex Euclidean spaces  
Normal{T} <: AbstractNormal{T} # mean vector / covariance matrix parametrisation of normal distributions 
Dirac{T}  <: AbstractNormal{T} # normal distribution with zero covariance 
```

* Functionality: 

```julia
dim(N::AbstractNormal)  # dimension  of the normal distribution 

mean(N::AbstractNormal) # mean vector 
cov(N::AbstractNormal)  # covariance matrix 
var(N::AbstractNormal)  # vector of marginal variances 
std(N::AbstractNormal)  # vector of marginal standard deviations 

residual(N::AbstractNormal,x) # whitened residual of realisation x
logpdf(N::AbstractNormal,x)   # logarithm of the probability density function at x 
entropy(N::AbstractNormal)   
kldivergence(N1::AbstractNormal,N2::AbstractNormal) 
rand(N::AbstractNormal) 
```

### Normal kernels 

* Mathematical definition: 

$$
k(y\mid x) = \mathcal{N}( \mu(x), \Sigma  )
$$

* Types: 

```julia
abstract type AbstractNormalKernel{T<:Number}  <: AbstractMarkovKernel end # normal kernel over real / complex Euclidean spaces  
NormalKernel{T} <:  AbstractNormalKernel{T}  # normal kernels with mean function / homoscedastic covariance 
DiracKernel{T}  <:  AbstractNormalKernel{T}  # same as above buit with zero covariance 
```

* Constructors: 

```julia
NormalKernel(Φ::AbstractMatrix,Σ::AbstractMatrix)  # Linear conditional mean with slope Φ
DiracKernel(Φ::AbstractMatrix)                     # same as above but with zero covariance
NormalKernel(Φ::AbstractMatrix,b::AbstractVector,Σ::AbstractMatrix) # affine conditional me an with slope Φ and intercept b
DiracKernel(Φ::AbstractMatrix,b::AbstractVector)                    # same as above but with zero covariance 
```

* Functionality: 

```julia
mean(K::AbstractNormalKernel)  # returns callable conditional mean function 
cov(K::AbstractNormalkernel)   # returns (non-callable) conditional covariance matrix 

condition(K::AbstractNormalKernel,x) # returns normal distribution by evaluating the conditional argument of the kernel 
compose(K2::AbstractNormalKernel,K1::AbstractNormalKernel) # Chapman-Kolmogorov 
marginalise(N::AbstractNormal,K::AbstractNormalKernel)   # marginalise out the conditional argument in K w.r.t N
invert(N::AbstractNormal,K::AbstractNormalKernel) inverts the factorisation N(x)*K(y,x) such that Nout(y)*Kout(x,y) = N(x)*K(y,x)

rand(K::AbstractNormalKernel,x)   # samples from K conditioned on x 
```
