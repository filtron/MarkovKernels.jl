#=

* AbstractCategorical
* Categorical
* CategoricalLikelihood
* covp
* UvNormal
* compose

=#

Base.@deprecate htransform(kernel, like) htransform_and_likelihood(kernel, like)
Base.@deprecate marginalize(dist, kernel) forward_operator(kernel, dist)
