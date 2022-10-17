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

        @testset "AffineNormalKernel | Unary | $(T) | $(atype) | $(ctype)" begin
            @test eltype(K) == T
            @test typeof(K) <: AffineNormalKernel
            @test mean(K)(x) == M(x)
            @test cov(K)(x) == cov_param
            @test covp(K) == cov_param
            @test condition(K, x) == Normal(M(x), cov_param)
        end
    end

    for kt1 in kernel_type_parameters, kt2 in kernel_type_parameters
        atype1, ctype1 = kt1
        atype2, ctype2 = kt2

        M1, cov_mat1, cov_param1, K1 = _make_normalkernel(T, n, n, atype1, ctype1)
        M2, cov_mat2, cov_param2, K2 = _make_normalkernel(T, n, n, atype2, ctype2)
        x = randn(T, n)

        @testset "AffineNormalKernel | Binary | {$(T),$(atype1),$(ctype1)} | {$(T),$(atype2),$(ctype2)}" begin
            @test slope(mean(compose(K2, K1))) ≈ slope(compose(M2, M1))
            @test cov(condition(compose(K2, K1), x)) ≈
                  slope(mean(K2)) * cov_mat1 * slope(mean(K2))' + cov_mat2
            # insert test for correct covp 
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
            @test mean(marginalise(N, K)) ≈ M(m)
            @test cov(marginalise(N, K)) ≈ slope(M) * ncov_mat * slope(M)' + kcov_mat

            @test mean(NC) ≈ mean(Ngt)
            @test cov(NC) ≈ cov(Ngt)

            @test slope(mean(KC)) ≈ slope(mean(Kgt))
            @test intercept(mean(KC)) ≈ intercept(mean(Kgt))
            @test cov(condition(KC, x)) ≈ cov(condition(Kgt, x))
            @test mean(condition(KC, x)) ≈ mean(condition(Kgt, x))
        end
    end
end
