function marginalise_test(T, n, m, cov_types, matrix_types)
    μp = randn(T, n)
    Σp = randn(T, n, n)
    Σp = Σp * Σp'
    Ap = randn(T, n, n)
    Qp = randn(T, n, n)
    Qp = Qp * Qp'

    for cov_t in cov_types, matrix_t in matrix_types
        μ = _make_vector(μp, matrix_t)
        Σ = _make_matrix(Σp, matrix_t)
        A = _make_matrix(Ap, matrix_t)
        Q = _make_matrix(Qp, matrix_t)

        N = Normal(μ, _make_covp(Σ, cov_t))
        NK = NormalKernel(A, _make_covp(Q, cov_t))
        DK = DiracKernel(A)

        for kernel in (NK, DK)
            @testset "marginalise | $(nameof(typeof(N))) | $(nameof(typeof(kernel)))" begin
                @test mean(marginalise(N, kernel)) ≈ A * μ
                if typeof(kernel) <: NormalKernel
                    @test cov(marginalise(N, kernel)) ≈ A * Σ * A' + Q
                elseif typeof(kernel) <: DiracKernel
                    @test cov(marginalise(N, kernel)) ≈ A * Σ * A'
                end
            end
        end
    end
end
