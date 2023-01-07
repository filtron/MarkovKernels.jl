function particle_system_test()
    k = 10
    d = 2

    X = [rand(d) for i in 1:k]
    logws = randn(k)
    μ = reduce(hcat, X) * exp.(logws) / mean(exp.(logws))

    P = ParticleSystem(logws, X)

    @testset "ParticleSystem | time marginal " begin
        dim(P) == d
        logweights(P) ≈ logws
        weights(P) ≈ exp.(logws) / sum(exp.(logws))
        nparticles(P) == k
        particles(P) == X
        mean(P) ≈ μ
    end
end
