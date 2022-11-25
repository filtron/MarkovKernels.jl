abstract type ResamplingMethod end


rand(R::ResamplingMethod, M::Mixture) = rand(GLOBAL_RNG, R, M)



struct MultinomialResampler <: ResamplingMethod end

function rand(rng::AbstractRNG, ::MultinomialResampler, M::Mixture)
        idx = StatsBase.wsample(rng, eachindex(weights(M)), weights(M), length(weights(M)))
        ws = one.(weights(M))
        ws = ws / sum(ws)
        return Mixture(ws, components(M)[idx])
end

function rand!(M::Mixture, rng::AbstractRNG, ::MultinomialResampler)
        idx = StatsBase.wsample(rng, eachindex(weights(M)), weights(M), length(weights(M)))
        ws = one.(weights(M))
        ws = ws / sum(ws)
        components(M)[1:end] = components(M)[idx]
        weights(M)[1:end] = ws
end
