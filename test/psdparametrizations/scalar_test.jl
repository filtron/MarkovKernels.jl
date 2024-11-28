@testset "PSDParametrizations | Scalar" begin
    etys = (Float64, ComplexF64)

    for T in etys
        LΣ = randn(real(T))
        Σ = abs2(LΣ)

        C = real(T)(0.25) * randn(T)

        LR = randn(real(T))
        R = abs2(LR)

        @testset "PSDParametrizations | Real | $(T)" begin
            @test lsqrt(Σ) == rsqrt(Σ) == sqrt(Σ)
            @test Σ ≈ abs2(rsqrt(Σ)) ≈ abs2(lsqrt(Σ))
            @test lsqrt(Σ) == adjoint(rsqrt(Σ))
            @test typeof(rsqrt(Σ)) == typeof(lsqrt(Σ))
            @test typeof(rsqrt(Σ)) <: real(T)

            @test stein(Σ, C) ≈ _stein(Σ, C)
            @test stein(Σ, C, R) ≈ _stein(Σ, C, R)

            @test all(isapprox.(schur_reduce(Σ, C), _schur_reduce(Σ, C); atol = 1e-10))
            @test all(
                isapprox.(schur_reduce(Σ, C, R), _schur_reduce(Σ, C, R); atol = 1e-14),
            )
        end
    end
end
