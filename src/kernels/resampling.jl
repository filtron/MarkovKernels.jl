abstract type ResamplingMethod end

rand(R::ResamplingMethod, M::Mixture) = rand(GLOBAL_RNG, R, M)

struct MultinomialResampler <: ResamplingMethod end

function rand(rng::AbstractRNG, ::MultinomialResampler, M::Mixture)
    idx = StatsBase.wsample(rng, eachindex(weights(M)), weights(M), length(weights(M)))
    ws = one.(weights(M))
    ws = ws / sum(ws)
    #return Mixture(ws, components(M)[idx])
    return Mixture(ws, copy.(components(M)[idx]))
end
