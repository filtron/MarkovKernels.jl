@testset "PSDParametrizations | UniformScaling" begin
    etys = (Float64, ComplexF64)

    for T in etys
        LΣ = randn(real(T)) * I
        Σ = LΣ^2

        C = real(T)(0.25) * randn(T) * I

        LR = randn(real(T)) * I
        R = LR^2

        @testset "PSDParametrizations | Real | $(T)" begin
            @test lsqrt(Σ) == rsqrt(Σ) == sqrt(Σ)
            @test Σ ≈ rsqrt(Σ)^2 ≈ lsqrt(Σ)^2
            @test lsqrt(Σ) == adjoint(rsqrt(Σ))
            @test typeof(rsqrt(Σ)) == typeof(lsqrt(Σ))
            @test typeof(rsqrt(Σ)) <: UniformScaling{<:real(T)}

            @test stein(Σ, C) ≈ _stein(Σ, C)
            @test stein(Σ, C, R) ≈ _stein(Σ, C, R)

            @test all(isapprox.(schur_reduce(Σ, C), _schur_reduce(Σ, C); atol = 1e-10))
            @test all(
                isapprox.(schur_reduce(Σ, C, R), _schur_reduce(Σ, C, R); atol = 1e-14),
            )
        end
    end
end
