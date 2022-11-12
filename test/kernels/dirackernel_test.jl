function dirackernel_test(T, n, affine_types, cov_types)
    for at in affine_types
        slope, intercept, F = _make_affinemap(T, n, n, at)
        K = DiracKernel(F)
        x = randn(T, n)

        eltypes = T <: Real ? (Float32, Float64) : (ComplexF32, ComplexF64)

        @testset "AffineDiracKernel | Unary | $(T) | $(at)" begin
            @test_nowarn repr(K)
            @test eltype(K) == T
            @test typeof(K) <: AffineDiracKernel
            @test K == DiracKernel(mean(K)...)
            @test convert(typeof(K), K) == K
            for U in eltypes
                @test AbstractMarkovKernel{U}(K) ==
                      AbstractDiracKernel{U}(K) ==
                      DiracKernel{U}(K)
                @test eltype(AbstractDiracKernel{U}(K)) == U
            end
            @test mean(K)(x) == F(x)
            @test condition(K, x) == Dirac(F(x))
            @test eltype(rand(K, x)) == T
        end
    end
end
