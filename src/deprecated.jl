Base.@deprecate covp(distlike) covparam(distlike)
Base.@deprecate htransform(kernel, like) htransform_and_likelihood(kernel, like)
Base.@deprecate marginalize(dist, kernel) forward_operator(kernel, dist)

Base.@deprecate_binding AbstractCategorical AbstractProbabilityVector
Base.@deprecate_binding Categorical ProbabilityVector
Base.@deprecate_binding CategoricalLikelihood LikelihoodVector
