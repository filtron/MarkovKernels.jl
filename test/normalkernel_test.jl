function normalkernel_test(T, n, affine_types, cov_types)
    kernel_type_parameters = Iterators.product(affine_types, cov_types)
    normal_type_parameters = cov_types

    x = randn(T, n)
    for ts in kernel_type_parameters
        atype, ctype = ts

        M, cov_mat, cov_param, K = _make_normalkernel(T, n, n, atype, ctype)
        @testset "NormalKernel | Unary | $(T) | $(atype) | $(ctype)" begin
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

        @testset "NormalKernel | Binary | {$(T),$(atype1),$(ctype1)} | {$(T),$(atype2),$(ctype2)}" begin
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

        @testset "NormalKernel | {$(T),$(katype),$(kctype)} | Normal | {$(T),$(nctype)}" begin
            @test mean(marginalise(N, K)) ≈ M(m)
            @test cov(marginalise(N, K)) ≈ slope(M) * ncov_mat * slope(M)' + kcov_mat
        end
    end

    # define a normal distribution
    Φ1 = randn(T, n, n)
    RQ1 = randn(T, n, n)
    Q1 = RQ1' * RQ1
    K1 = NormalKernel(Φ1, Q1)

    Φ2 = randn(T, n, n)
    RQ2 = randn(T, n, n)
    Q2 = RQ2' * RQ2
    K2 = NormalKernel(Φ2, Q2)

    x = randn(T, n)

    μ, Σ, N1 = _make_normal(T, n)
    pred, S, G, Π = _schur(Σ, μ, Φ1, Q1)
    N_gt1 = Normal(pred, S)
    corrector = AffineMap(G, μ, pred)
    K_gt1 = NormalKernel(corrector, Π)

    NC1, KC1 = invert(N1, K1)

    @testset "AffineNormalKernel / Normal | $(T) " begin
        @test mean(NC1) ≈ mean(N_gt1)
        @test cov(NC1) ≈ cov(N_gt1)
        @test cov(KC1)(x) ≈ cov(K_gt1)(x)

        @test slope(mean(KC1)) ≈ slope(mean(K_gt1))
        @test intercept(mean(KC1)) ≈ intercept(mean(K_gt1))
    end

    λ1 = 2.0
    IN1 = IsoNormal(μ, λ1)

    pred, S, G, Π = _schur(λ1 * I, μ, Φ1, Q1)
    N_gt2 = Normal(pred, S)
    corrector = AffineMap(G, μ, pred)
    K_gt2 = NormalKernel(corrector, Π)

    NC2, KC2 = invert(IN1, K1)

    @testset "AffineNormalKernel / IsoNormal | $(T) " begin
        @test mean(NC2) ≈ mean(N_gt2)
        @test cov(NC2) ≈ cov(N_gt2)
        @test cov(KC2)(x) ≈ cov(K_gt2)(x)
        @test slope(mean(KC2)) ≈ slope(mean(K_gt2))
        @test intercept(mean(KC2)) ≈ intercept(mean(K_gt2))
    end

    λ2 = 3.0
    λ3 = 1.0 / 2.0

    IK1 = NormalKernel(Φ1, λ2 * I)

    pred, S, G, Π = _schur(Σ, μ, Φ1, λ2 * I)
    N_gt3 = Normal(pred, S)
    corrector = AffineMap(G, μ, pred)
    K_gt3 = NormalKernel(corrector, Π)

    NC3, KC3 = invert(N1, IK1)

    @testset "AffineIsoNormalKernel / Normal | $(T) " begin
        @test mean(NC3) ≈ mean(N_gt3)
        @test cov(NC3) ≈ cov(N_gt3)
        @test cov(KC3)(x) ≈ cov(K_gt3)(x)
        @test slope(mean(KC3)) ≈ slope(mean(K_gt3))
        @test intercept(mean(KC3)) ≈ intercept(mean(K_gt3))
    end

    pred, S, G, Π = _schur(λ1 * I, μ, Φ1, λ2 * I)
    N_gt4 = Normal(pred, S)
    corrector = AffineMap(G, μ, pred)
    K_gt4 = NormalKernel(corrector, Π)

    NC4, KC4 = invert(IN1, IK1)

    @testset "AffineIsoNormalKernel / IsoNormal | $(T) " begin
        @test mean(NC4) ≈ mean(N_gt4)
        @test cov(NC4) ≈ cov(N_gt4)
        @test cov(KC4)(x) ≈ cov(K_gt4)(x)
        @test slope(mean(KC4)) ≈ slope(mean(K_gt4))
        @test intercept(mean(KC4)) ≈ intercept(mean(K_gt4))
    end
end
