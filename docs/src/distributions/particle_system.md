# ParticleSystem

```@meta
CurrentModule = MarkovKernels
```

A Particle system is a mixture of Dirac distributions.
```math
P(x) = \sum_{i= 1}^n w_i \delta(x - μ^{(i)})
```

It can also be used to represent a mixture of trajectories.
```math
P(x) = \sum_{i= 1}^n w_i \delta(x_{1:T} - μ^{(i)}_{1:T})
```


### Types
```@docs
AbstractParticleSystem{T}
ParticleSystem{T}
```

### Constructors
```@docs
ParticleSystem(logws::AbstractVector{<:Real}, X::AbstractArray{<:AbstractVector{T}}) where {T}
```

### Basics

```@docs
dim(::ParticleSystem)
logweights(::ParticleSystem)
weights(::ParticleSystem)
nparticles(::ParticleSystem)
particles(::ParticleSystem)
mean(::ParticleSystem)
```
