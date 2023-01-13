```@meta
CurrentModule = MarkovKernels
```

### composition

Given Markov kernels ``k_2(y,z)`` and ``k_1(z,x)``, composition is a binary operator producing a third kernel ``k_3(y,x)`` according to

```math
k_3(y,x) = \int k_2(y,x) k_1(z,x) \mathrm{d} z.
```

```@docs
compose(K2::AffineNormalKernel{T}, K1::AffineNormalKernel{T}) where {T}
```


### marginalisation

Given a distribution ``\pi(x)`` and a Markov kernel ``k(y,x)``, marginalisation is a binary operator producing a new distriubution ``p(y)`` according to

```math
p(y) = \int k(y, x) \pi(x) \mathrm{d} x.
```

```@docs
marginalise(N::AbstractNormal{T}, K::AffineNormalKernel{T}) where {T}
```

### invert & Bayes' rule

Given a distribution ``\pi(x)`` and a Markov kernel ``k(y,x)``, invert is a binary operator producing a new distribution ``m(y)`` and a new Markov kernel ``p(x , y)`` according to

```math
\pi(x) k(y,x) = m(y) p(x,y).
```

The related binary operator, Bayes' rule also evalautes the output of invert at some measurement ``y``.
That is, given a measurmeent ``y``, ``m`` evaluated at ``y`` is the marginal likelihood and ``p`` evaluated at ``y`` is the conditional distribution of ``x`` given ``y``.

```@docs
invert(N::AbstractNormal{T}, K::AffineNormalKernel{T}) where {T}
bayes_rule(D::AbstractDistribution, K::AbstractMarkovKernel, y)
bayes_rule(D::AbstractDistribution, L::AbstractLogLike)
bayes_rule!(P::ParticleSystem{T,U,<:AbstractVector}, L::AbstractLogLike) where {T,U}
```
