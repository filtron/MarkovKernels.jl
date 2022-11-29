function sample(rng, init, K, nstep)
    it = Iterators.take(iterated(z -> rand(rng, K, z), rand(rng, init)), nstep + 1)
    return mapreduce(permutedims, vcat, collect(it))
end
