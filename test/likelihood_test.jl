@testset "Likelihood" begin
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

            @test L == Likelihood(K, y)
            @test measurement(L) == y
            @test measurement_model(L) == K
            @test log(L, x) ≈ logpdf(condition(K, x), y)
        end

        @testset "Likelihood | AffineDirac" begin
            K = DiracKernel(C)
            L = Likelihood(K, y)

            @test L == Likelihood(K, y)
            @test measurement(L) == y
            @test measurement_model(L) == K
        end

        @testset "FlatLikelihood" begin
            L1 = FlatLikelihood()
            L2 = FlatLikelihood()
            x = randn(T, 1)
            @test_nowarn FlatLikelihood()
            @test L1 === L2
            @test log(L1, x) == zero(real(eltype(x)))
            @test typeof(log(L1, x)) == typeof(zero(real(eltype(x))))
        end
    end
end
