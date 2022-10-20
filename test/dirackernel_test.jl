function dirackernel_test(T, n, affine_types, cov_types)
    for at in affine_types
        slope, intercept, F = _make_affinemap(T, n, n, at)
        K = DiracKernel(F)
        x = randn(T, n)

        eltypes = T <: Real ? (Float32, Float64) : (ComplexF32, ComplexF64)

        @testset "AffineDiracKernel | Unary | $(T) | $(at)" begin
            @test eltype(K) == T
            @test typeof(K) <: AffineDiracKernel
            @test K == DiracKernel(mean(K)...)
            @test convert(typeof(K), K) == K
            for U in eltypes
                eltype(AbstractDiracKernel{U}(K)) == U
                convert(AbstractDiracKernel{U}, K) == AbstractDiracKernel{U}(K)
            end
            @test mean(K)(x) == F(x)
            @test cov(K)(x) == Diagonal(zeros(T, n))
            @test condition(K, x) == Dirac(F(x))
        end
    end
end
