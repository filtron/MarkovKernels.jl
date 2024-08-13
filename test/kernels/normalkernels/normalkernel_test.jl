@testset "NormalKernels | NormalKernel" begin
    etys = (Float64, Complex{Float64})
    for T in etys
        A = ones(T, 1, 1)
        Σ(x) = selfadjoint(eltype(x), eltype(x).(diagm(exp.(abs.(x)))))
        F = LinearMap(A)
        K = NormalKernel(F, Σ)
        x = randn(T, 1)

        @testset "NormalKernels | NormalKernel | Unary | $(T)" begin
            @test_nowarn repr(K)
            @test mean(K)(x) == A * x
            @test cov(K)(x) == Σ(x)
            @test condition(K, x) == Normal(A * x, Σ(x))
        end
    end
end
