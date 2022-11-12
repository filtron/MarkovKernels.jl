function normalkernel_test(T, affine_types)
    Σ = x -> hcat(exp.(abs.(x)))

    for t in affine_types
        slope, intercept, F = _make_affinemap(T, 1, 1, t)
        K = NormalKernel(F, Σ)
        x = randn(T, 1)

        @testset "NormalKernel | Unary | $(T) | $(t)" begin
            #@test_nowarn show(K)
            @test mean(K)(x) == F(x)
            @test cov(K)(x) == Σ(x)
            @test condition(K, x) == Normal(F(x), Σ(x))
        end
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

    kernel_type_parameters = Iterators.product(affine_types, cov_types)

    for ts in kernel_type_parameters
        atype, ctype = ts

        M, cov_mat, cov_param, K = _make_normalkernel(T, n, n, atype, ctype)
        x = randn(T, n)

        eltypes = T <: Real ? (Float32, Float64) : (ComplexF32, ComplexF64)

        @testset "AffineNormalKernel | Unary | $(T) | $(atype) | $(ctype)" begin
            #@test_nowarn show(K)
            @test_nowarn repr(K)
            @test eltype(K) == T
            @test typeof(K) <: AffineNormalKernel
            @test K == NormalKernel(mean(K)..., cov_param)
            for U in eltypes
                @test AbstractMarkovKernel{U}(K) ==
                      AbstractNormalKernel{U}(K) ==
                      NormalKernel{U}(K)
                @test eltype(AbstractNormalKernel{U}(K)) == U
            end
            @test mean(K)(x) == M(x)
            @test cov(K)(x) == cov_param
            @test covp(K) == cov_param
            @test condition(K, x) == Normal(M(x), cov_param)
            @test eltype(rand(K, x)) == T
        end
    end
end
