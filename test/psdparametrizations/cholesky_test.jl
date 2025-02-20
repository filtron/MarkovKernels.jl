@testset "PSDParametrizations | Cholesky" begin
    etys = (Float64, ComplexF64)

    for T in etys
        m, n = 2, 3

        LΣ = randn(T, n, n)
        Σ = selfadjoint(LΣ * adjoint(LΣ))
        CΣ = cholesky(Σ)

        C = randn(T, m, n)
        LR = randn(T, m, m)
        R = selfadjoint(LR * adjoint(LR))
        CR = cholesky(R)

        C2 = adjoint(randn(T, n))
        LR2 = randn(real(T))
        R2 = abs2(LR2)

        R4 = R2 * I

        @testset "PSDParametrizations | Cholesky | $(T)" begin
            @test rsqrt(CΣ) ≈ cholesky(Σ).U
            @test lsqrt(CΣ) ≈ cholesky(Σ).L
            @test lsqrt(CΣ) == adjoint(rsqrt(CΣ))

            @test _to_matrix(stein(CΣ, C)) ≈ _stein(Σ, C)
            @test _to_matrix(stein(CΣ, C, CR)) ≈ _stein(Σ, C, R)
            @test all(isapprox.(_to_matrix.(schur_reduce(CΣ, C)), _schur_reduce(Σ, C)))
            @test all(
                isapprox.(_to_matrix.(schur_reduce(CΣ, C, CR)), _schur_reduce(Σ, C, R)),
            )

            @test _to_matrix(stein(CΣ, C2)) ≈ _stein(Σ, C2)
            @test _to_matrix(stein(CΣ, C2, R2)) ≈ _stein(Σ, C2, R2)
            @test all(isapprox.(_to_matrix.(schur_reduce(CΣ, C2)), _schur_reduce(Σ, C2)))
            @test all(
                isapprox.(_to_matrix.(schur_reduce(CΣ, C2, R2)), _schur_reduce(Σ, C2, R2)),
            )

            @test _to_matrix(stein(CΣ, C, R4)) ≈ _stein(Σ, C, R4)
            @test all(
                isapprox.(_to_matrix.(schur_reduce(CΣ, C, R4)), _schur_reduce(Σ, C, R4)),
            )
        end

        RD = Diagonal(abs2.(randn(real(T), m)))

        @testset "PSDParametrizations | Cholesky | Diagonal | $(T)" begin
            @test _to_matrix(stein(CΣ, C, RD)) ≈ _stein(Σ, C, RD)
            @test all(
                isapprox.(_to_matrix.(schur_reduce(CΣ, C, RD)), _schur_reduce(Σ, C, RD)),
            )
        end
    end
end
