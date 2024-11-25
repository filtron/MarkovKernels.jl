@safetestset "Likelihood" begin
    using MarkovKernels, LinearAlgebra
    etys = (Float64, ComplexF64)
    m, n = 2, 3

    for T in etys
        C = randn(T, m, n)
        FC = LinearMap(C)
        R = Cholesky(UpperTriangular(ones(m, m)))
        x = randn(T, n)
        y = randn(T, m)

        @testset "Likelihood | AffineNormal" begin
            K = NormalKernel(FC, R)
            L = Likelihood(K, y)

            @test typeof(L) <: Likelihood
            @test L == Likelihood(K, y)
            @test measurement(L) == y
            @test measurement_model(L) == K
            @test log(L, x) ≈ logpdf(condition(K, x), y)
        end

        @testset "Likelihood | AffineDirac" begin
            K = DiracKernel(C)
            L = Likelihood(K, y)

            @test typeof(L) <: Likelihood
            @test L == Likelihood(K, y)
            @test measurement(L) == y
            @test measurement_model(L) == K
        end
    end

    etys = (Float64,)
    m = 2
    for T in etys
        P = ones(T, m, m) / m
        K = StochasticMatrix(P)
        y = 1
        L = Likelihood(K, y)
        xs = 1:m

        @test typeof(L) <: Likelihood
        @test L == Likelihood(K, y)
        @test measurement(L) == y
        @test measurement_model(L) == K
        @test all(x -> log(L, x) ≈ logpdf(condition(K, x), y), xs)
    end
end
