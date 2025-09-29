#=

* AbstractCategorical
* Categorical
* CategoricalLikelihood

=#

Base.@deprecate covp(distlike)  covparam(distlike)
Base.@deprecate htransform(kernel, like) htransform_and_likelihood(kernel, like)
Base.@deprecate marginalize(dist, kernel) forward_operator(kernel, dist)
