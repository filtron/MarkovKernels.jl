function normalkernel_test(T, affine_types)
    Σ = x -> hcat(exp.(abs.(x)))

    for t in affine_types
        slope, intercept, F = _make_affinemap(T, 1, 1, t)
        K = NormalKernel(F, Σ)
        x = randn(T, 1)

        @testset "NormalKernel | Unary | $(T) | $(t)" begin
            @test mean(K)(x) == F(x)
            @test cov(K)(x) == Σ(x)
            @test condition(K, x) == Normal(F(x), Σ(x))
        end
    end
end

function affine_normalkernel_test(T, n, affine_types, cov_types)
    kernel_type_parameters = Iterators.product(affine_types, cov_types)
    normal_type_parameters = cov_types

    for ts in kernel_type_parameters
        atype, ctype = ts

        M, cov_mat, cov_param, K = _make_normalkernel(T, n, n, atype, ctype)
        x = randn(T, n)

        eltypes = T <: Real ? (Float32, Float64) : (ComplexF32, ComplexF64)

        @testset "AffineNormalKernel | Unary | $(T) | $(atype) | $(ctype)" begin
            @test eltype(K) == T
            @test typeof(K) <: AffineNormalKernel
            @test K == NormalKernel(mean(K)..., cov_param)
            @test convert(typeof(K), K) == K
            for U in eltypes
                eltype(AbstractNormalKernel{U}(K)) == U
                convert(AbstractNormalKernel{U}, K) == AbstractNormalKernel{U}(K)
            end
            @test mean(K)(x) == M(x)
            @test cov(K)(x) == cov_param
            @test covp(K) == cov_param
            @test condition(K, x) == Normal(M(x), cov_param)
        end
    end
end
