function dirackernel_test(T, n, matrix_types)
    eltypes = T <: Real ? (Float32, Float64) : (ComplexF32, ComplexF64)

    Ap = randn(T, n, n)
    xp = randn(T, n)

    for matrix_t in matrix_types
        A = _make_matrix(Ap, matrix_t)
        x = _make_vector(xp, matrix_t)
        K = DiracKernel(A)
        @testset "AffineDiracKernel | Unary | $(T)" begin
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
            @test mean(K)(x) == A * x
            @test condition(K, x) == Dirac(A * x)
            @test eltype(rand(K, x)) == T
        end
    end
end
