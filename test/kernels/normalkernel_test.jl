function normalkernel_test(T, affine_types)
    A = ones(T, 1, 1)
    Σ = x -> diagm(exp.(abs.(x)))
    K = NormalKernel(A, Σ)
    x = randn(T, 1)

    @testset "NormalKernel | Unary | $(T)" begin
        @test_nowarn repr(K)
        @test mean(K)(x) == A * x
        @test cov(K)(x) == Σ(x)
        @test condition(K, x) == Normal(A * x, Σ(x))
    end
end

function affine_normalkernel_test(T, n, affine_types, cov_types)
    @testset "NormalKernel | AbstractMatrix constructor" begin
        @test_throws DomainError NormalKernel(ones(2, 2), tril(ones(2, 2)))
        @test_throws DomainError NormalKernel(ones(ComplexF64, 2, 2), tril(ones(2, 2)))
        @test_throws DomainError NormalKernel(
            ones(ComplexF64, 2, 2),
            Symmetric(diagm(ones(2))),
        )
    end

    covar_types = (:HermOrSym, :Cholesky)
    matrix_types = (:Matrix, :SMatrix)
    eltypes = T <: Real ? (Float32, Float64) : (ComplexF32, ComplexF64)

    Ap = randn(T, n, n)
    Vp = randn(T, n, n)
    Vp = Vp * Vp'
    μp = randn(T, n)
    xp = randn(T, n)

    for cov_t in covar_types, matrix_t in matrix_types
        A = _make_matrix(Ap, matrix_t)
        μ = _make_vector(μp, matrix_t)
        Σ = _make_covp(_make_matrix(Vp, matrix_t), cov_t)
        x = _make_vector(xp, matrix_t)
        K = NormalKernel(A, Σ)

        @testset "AffineNormalKernel | Unary | $(T) | $(cov_t) | $(matrix_t)" begin
            @test_nowarn repr(K)
            @test eltype(K) == T
            @test typeof(K) <: AffineNormalKernel
            @test K == NormalKernel(mean(K)..., Σ)
            for U in eltypes
                @test AbstractMarkovKernel{U}(K) ==
                      AbstractNormalKernel{U}(K) ==
                      NormalKernel{U}(K)
                @test eltype(AbstractNormalKernel{U}(K)) == U
            end
            @test mean(K)(x) == A * x
            @test _ofsametype(x, A * x)
            @test cov(K)(x) == Σ
            @test covp(K) == Σ
            @test condition(K, x) == Normal(A * x, Σ)
            @test eltype(rand(K, x)) == T
            @test _ofsametype(μ, rand(K, x))
        end
    end
end
