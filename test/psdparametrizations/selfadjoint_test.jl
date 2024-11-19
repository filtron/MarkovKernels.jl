@testset "PSDParametrizations | SelfAdjoint" begin
    etys = (Float64, ComplexF64)

    for T in etys
        m, n = 2, 3

        LΣ = randn(T, n, n)
        Σ = selfadjoint(LΣ * adjoint(LΣ))

        C = randn(T, m, n)
        LR = randn(T, m, m)
        R = selfadjoint(LR * adjoint(LR))

        C2 = adjoint(randn(T, n))
        LR2 = randn(real(T))
        R2 = abs2(LR2)

        @testset "CovarianceParamater | SelfAdjoint | $(T)" begin
            @test rsqrt(Σ) ≈ cholesky(Σ).U
            @test lsqrt(Σ) ≈ cholesky(Σ).L

            @test stein(Σ, C) ≈ _stein(Σ, C)
            @test stein(Σ, C, R) ≈ _stein(Σ, C, R)
            @test all(isapprox.(schur_reduce(Σ, C), _schur_reduce(Σ, C)))
            @test all(isapprox.(schur_reduce(Σ, C, R), _schur_reduce(Σ, C, R)))

            @test stein(Σ, C2) ≈ _stein(Σ, C2)
            @test stein(Σ, C2, R2) ≈ _stein(Σ, C2, R2)
            @test all(isapprox.(schur_reduce(Σ, C2), _schur_reduce(Σ, C2)))
            @test all(isapprox.(schur_reduce(Σ, C2, R2), _schur_reduce(Σ, C2, R2)))
        end
    end
end
