function dirackernel_test(T, n, matrix_types)
    @testset "NormalKernel | AbstractMatrix constructor" begin
        @test mean(DiracKernel(AffineMap(1.0I(2), ones(2)))) == AffineMap(1.0I(2), ones(2))
        @test mean(DiracKernel(AffineCorrector(1.0I(2), ones(2), zeros(2)))) ==
              AffineCorrector(1.0I(2), ones(2), zeros(2))
    end

    eltypes = T <: Real ? (Float32, Float64) : (ComplexF32, ComplexF64)

    Ap = randn(T, n, n)
    xp = randn(T, n)

    for matrix_t in matrix_types
        A = _make_matrix(Ap, matrix_t)
        FA = LinearMap(A)
        x = _make_vector(xp, matrix_t)
        K = DiracKernel(FA)
        @testset "AffineDiracKernel | Unary | $(T)" begin
            @test_nowarn repr(K)

            @test !(copy(K) === K)
            @test typeof(copy(K)) === typeof(K)
            @test typeof(similar(K)) === typeof(K)
            @test copy!(similar(K), K) == K

            @test typeof(K) <: AffineDiracKernel
            @test convert(typeof(K), K) == K

            @test mean(K)(x) == A * x
            @test condition(K, x) == Dirac(A * x)
            @test eltype(rand(K, x)) == T
        end
    end
end
