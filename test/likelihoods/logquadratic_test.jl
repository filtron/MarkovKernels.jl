@safetestset "LogQuadraticLikelihood" begin
    using MarkovKernels, LinearAlgebra
    etys = (Float64, ComplexF64)
    m, n = 2, 3

    for T in etys
        C1 = randn(T, m, n)
        FC1 = LinearMap(C1)
        R = Cholesky(UpperTriangular(ones(m, m)))

        logc2 = randn(real(T))
        y2 = randn(T, m)
        C2 = randn(T, m, n)

        x = randn(T, n)
        y = randn(T, m)

        @testset "Likelihood | AffineNormal" begin
            K = NormalKernel(FC1, R)
            L1 = Likelihood(K, y)
            L2 = LogQuadraticLikelihood(L1)

            @test typeof(L2) <: LogQuadraticLikelihood
            @test log(L2, x) ≈ log(L1, x)
        end
    end
end
