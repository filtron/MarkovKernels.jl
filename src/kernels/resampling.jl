abstract type ResamplingMethod end

struct MultinomialResampler end

rand(R::ResamplingMethod, M::Mixture) = nothing
