@testset "DiracKernels | IdentityKernel" begin
    etys = (Float64, Complex{Float64})
    for T in etys, m in (2, 3)
        K = IdentityKernel()
        x = randn(T, m)

        @testset "DiracKernels | IdentityKernel $(T)" begin
            @test_nowarn repr(K)
            @test mean(K)(x) == identity(x) == x
            @test condition(K, x) == Dirac(x)
        end
    end
end
