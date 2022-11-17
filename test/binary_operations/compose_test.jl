function compose_test(T, n, cov_types, matrix_types)
    A1p = randn(T, n, n)
    A2p = randn(T, n, n)
    V1p = randn(T, n, n)
    V1p = V1p * V1p'
    V2p = randn(T, n, n)
    V2p = V2p * V2p'
    xp = randn(T, n)

    for cov_t in cov_types, matrix_t in matrix_types
        A1 = _make_matrix(A1p, matrix_t)
        A2 = _make_matrix(A2p, matrix_t)
        Σ1 = _make_matrix(V1p, matrix_t)
        Σ2 = _make_matrix(V2p, matrix_t)
        x = _make_vector(xp, matrix_t)

        NK1 = NormalKernel(A1, _make_covp(Σ1, cov_t))
        NK2 = NormalKernel(A2, _make_covp(Σ2, cov_t))

        DK1 = DiracKernel(A1)
        DK2 = DiracKernel(A2)

        kernel_pairs = ((NK1, NK2), (NK1, DK2), (DK1, NK2), (DK1, DK2))

        for kernels in kernel_pairs
            K1, K2 = kernels
            @testset "compose | $(nameof(typeof(K2))) | $(nameof(typeof(K1)))" begin
                @test mean(compose(K2, K1)) == compose(LinearMap(A2), LinearMap(A1))

                if all(typeof.(kernels) .<: NormalKernel)
                    @test cov(condition(compose(K2, K1), x)) ≈
                          slope(mean(K2)) * Σ1 * slope(mean(K2))' + Σ2
                elseif typeof(K1) <: DiracKernel && typeof(K2) <: NormalKernel
                    @test cov(condition(compose(K2, K1), x)) ≈ Σ2
                end
            end
        end
    end
end
