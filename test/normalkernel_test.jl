function normalkernel_test(T, affine_types)
    Σ = x -> hcat(exp.(abs.(x)))

    for t in affine_types
        slope, intercept, F = _make_affinemap(T, 1, 1, t)
        K = NormalKernel(F, Σ)
        x = randn(T, 1)

        @testset "NormalKernel | Unary | $(T) | $(t)" begin
            @test mean(K)(x) == F(x)
            @test cov(K)(x) == Σ(x)
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

    for kt in kernel_type_parameters, nt in normal_type_parameters
        katype, kctype = kt
        nctype = nt
        M, kcov_mat, kcov_param, K = _make_normalkernel(T, n, n, katype, kctype)
        m, ncov_mat, ncov_param, N = _make_normal(T, n, nctype)

        NC, KC = invert(N, K)
        x = randn(T, n)

        pred, S, G, Π = _schur(ncov_mat, m, slope(M), kcov_mat) # _schur should not return pred
        pred = M(m)
        Ngt = Normal(pred, S)
        Kgt = NormalKernel(G, m, pred, Π)

        @testset "NormalKernel | {$(T),$(katype),$(kctype)} | Normal | {$(T),$(nctype)}" begin
            @test mean(NC) ≈ mean(Ngt)
            @test cov(NC) ≈ cov(Ngt)

            @test slope(mean(KC)) ≈ slope(mean(Kgt))
            @test intercept(mean(KC)) ≈ intercept(mean(Kgt))
            @test cov(condition(KC, x)) ≈ cov(condition(Kgt, x))
            @test mean(condition(KC, x)) ≈ mean(condition(Kgt, x))
        end
    end
end
