@safetestset "htransform" begin
    using MarkovKernels, LinearAlgebra

    n, m = 2, 3
    etys = (Float64,)

    @testset "htransform | StochasticMatrix / CategoricalLikelihood" begin
        for T in etys
            π = exp.(randn(T, m))
            π = π / sum(π)

            C0 = Categorical(π)

            Pxx = exp.(randn(T, m, m))
            Pxx = Pxx * Diagonal(1 ./ [sum(p) for p in eachcol(Pxx)])
            Kxx = StochasticMatrix(Pxx)

            Pyx = exp.(randn(T, n, m))
            Pyx = Pyx * Diagonal(1 ./ [sum(p) for p in eachcol(Pyx)])
            Kyx = StochasticMatrix(Pyx)
            y = rand(1:n)

            L = Likelihood(Kyx, y)

            # forward / backward
            C1, Bxx = invert(C0, Kxx)
            C1, ll_fb = posterior_and_loglike(C1, L)
            C0_fb, Fxx_fb = invert(C1, Bxx)

            # backward / forward
            Fxx_bf, L0 = htransform(Kxx, L)
            C0_bf, ll_bf = posterior_and_loglike(C0, L0)

            @test ll_fb ≈ ll_bf
            @test C0_fb ≈ C0_bf
            @test Fxx_fb ≈ Fxx_bf
        end
    end
end
