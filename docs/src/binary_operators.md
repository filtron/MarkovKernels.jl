```@meta
CurrentModule = MarkovKernels
```

### Composition

Given Markov kernels ``k_2(y,z)`` and ``k_1(z,x)``, composition is a binary operator producing a third kernel ``k_3(y,x)`` according to

```math
k_3(y,x) = \int k_2(y,x) k_1(z,x) \mathrm{d} z.
```

```@docs
compose(::AbstractMarkovKernel, ::AbstractMarkovKernel)
∘(K2::AbstractMarkovKernel, K1::AbstractMarkovKernel)
```

Additionally, given likelihoods ``l_1(x)`` and ``l_2(x)``, composition is a binary operator producing a third likelihood ``l_3(x)`` according to

```math
l_3(x) = l_2(x) l_1(x).
```

```@docs
compose(::AbstractLikelihood, ::AbstractLikelihood)
∘(K2::AbstractLikelihood, K1::AbstractLikelihood)
```


### Algebra

```@docs
+(D::AbstractDistribution, v::AbstractNumOrVec)
-(N::Normal)
-(v::AbstractNumOrVec, D::AbstractDistribution)
*(C, D::AbstractDistribution)
```


### Marginalization

Given a distribution ``\pi(x)`` and a Markov kernel ``k(y,x)``, marginalization is a binary operator producing a new distriubution ``p(y)`` according to

```math
p(y) = \int k(y, x) \pi(x) \mathrm{d} x.
```

```@docs
marginalize(N::AbstractNormal, K::AffineHomoskedasticNormalKernel)
```

### Bayes' rule & invert

Given a distribution ``\pi(x)`` and a Markov kernel ``k(y,x)``, invert is a binary operator producing a new distribution ``m(y)`` and a new Markov kernel ``p(x , y)`` according to

```math
\pi(x) k(y,x) = m(y) p(x,y).
```

The related binary operator, Bayes' rule also evalautes the output of invert at some measurement ``y``.
That is, given a measurmeent ``y``, ``m`` evaluated at ``y`` is the marginal likelihood and ``p`` evaluated at ``y`` is the conditional distribution of ``x`` given ``y``.



```@docs
invert(N::AbstractDistribution, K::AbstractMarkovKernel)
posterior_and_loglike(D::AbstractDistribution, K::AbstractMarkovKernel, y)
posterior_and_loglike(::AbstractDistribution, ::AbstractLikelihood)
posterior(D::AbstractDistribution, K::AbstractMarkovKernel, y)
posterior(D::AbstractDistribution, L::AbstractLikelihood)
```

### Doob's h-transform

Given a Markov kernel ``k(y, x)`` and a likelihood function ``h(y)``, computes a new Markov kernel ``f(y, x)`` and new likelihood function ``g(x)``

```@docs
htransform_and_likelihood(::AbstractMarkovKernel, ::AbstractLikelihood)
```
