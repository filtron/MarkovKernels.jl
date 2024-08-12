function normalkernel_test(T)
    A = ones(T, 1, 1)
    Σ = x -> _symmetrise(eltype(x), T.(diagm(exp.(abs.(x)))))
    F = LinearMap(A)
    K = NormalKernel(F, Σ)
    x = randn(T, 1)

    @testset "NormalKernel | Unary | $(T)" begin
        @test_nowarn repr(K)
        @test mean(K)(x) == A * x
        @test cov(K)(x) == Σ(x)
        @test condition(K, x) == Normal(A * x, Σ(x))
    end
end

function affine_normalkernel_test(T, n, cov_types, matrix_types)
    eltypes = T <: Real ? (Float32, Float64) : (ComplexF32, ComplexF64)

    Ap = randn(T, n, n)
    Vp = randn(T, n, n)
    Vp = Vp * Vp'
    μp = randn(T, n)
    xp = randn(T, n)

    for cov_t in cov_types, matrix_t in matrix_types
        A = _make_matrix(Ap, matrix_t)
        μ = _make_vector(μp, matrix_t)
        Σ = _make_covp(_make_matrix(Vp, matrix_t), cov_t)
        x = _make_vector(xp, matrix_t)
        F = LinearMap(A)
        K = NormalKernel(F, Σ)

        @testset "AffineNormalKernel | Unary | $(T) | $(cov_t) | $(matrix_t)" begin
            @test_nowarn repr(K)

            @test !(copy(K) === K)
            @test typeof(copy(K)) === typeof(K)
            @test typeof(similar(K)) === typeof(K)
            @test copy!(similar(K), K) == K

            @test typeof(K) <: AffineNormalKernel
            # @test K == NormalKernel(mean(K)..., Σ)
            @test mean(K)(x) == F(x)
            @test _ofsametype(x, F(x))
            @test cov(K)(x) == Σ
            @test covp(K) == Σ
            @test condition(K, x) == Normal(F(x), Σ)
            @test eltype(rand(K, x)) == T
            @test _ofsametype(μ, rand(K, x))
        end
    end
end
