# MarkovKernels.jl 

A package implementing Bayesian filtering and smoothing by manipulating marginal distributions and Markov kernels.

[![Build Status](https://github.com/filtron/MarkovKernels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/filtron/MarkovKernels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/filtron/MarkovKernels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/filtron/MarkovKernels.jl)

## Package specific types

*Type for encoding marginal distributions 

```julia
abstract type AbstractDistribution end
abstract type AbstractMarkovKernel end
abstract type AbstractLikelihood end
```
