@safetestset "LikelihoodVector" begin
    using MarkovKernels, LinearAlgebra
    etys = (Float64,)
    m, n = 2, 3

    for T in etys
        P1 = exp.(randn(T, m, m))
        P1 = P1 * Diagonal(1 ./ [sum(p) for p in eachcol(P1)])
        K1 = StochasticMatrix(P1)

        P2 = exp.(randn(T, m, n))
        P2 = P2 * Diagonal(1 ./ [sum(p) for p in eachcol(P2)])
        K2 = StochasticMatrix(P2)

        ys = 1:m
        x1s = 1:m
        x2s = 1:n

        for y in ys
            for K in (K1, K2)
                LSM = Likelihood(K, y)
                ls = likelihood_vector(LSM)
                CL1 = LikelihoodVector(ls)
                CL2 = LikelihoodVector(LSM)

                @test CL1 == CL2
                @test likelihood_vector(CL1) ≈ likelihood_vector(CL2) ≈ ls
            end
        end
    end
end
