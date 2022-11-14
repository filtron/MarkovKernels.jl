function covariance_parameter_test(T, cov_types, matrix_types)
    n = 1
    m = 2

    Vp = tril(ones(T, m, m))
    Vp = Vp * Vp'
    Cp = ones(T, n, m)
    Rp = ones(T, n, n)

    for matrix_t in matrix_types, cov_t in cov_types
        V = _make_covp(_make_matrix(Vp, matrix_t), cov_t)
        C = _make_matrix(Cp, matrix_t)
        R = _make_covp(_make_matrix(Rp, matrix_t), cov_t)

        @testset "stein | $(matrix_t) | $(cov_t)" begin
            @test _ofsametype(_make_matrix(Rp, matrix_t), stein(V, C))
            @test _ofsametype(_make_matrix(Rp, matrix_t), stein(V, C, R))
        end

        S1, K1, Σ1 = schur_reduce(V, C)
        S2, K2, Σ2 = schur_reduce(V, C, R)

        @testset "schur_reduce | $(matrix_t) | $(cov_t)" begin
            @test _ofsametype(_make_matrix(Rp, matrix_t), S1)
            @test _ofsametype(permutedims(C), K1)
            @test _ofsametype(_make_matrix(Vp, matrix_t), Σ1)

            @test _ofsametype(_make_matrix(Rp, matrix_t), S2)
            @test _ofsametype(permutedims(C), K2)
            @test _ofsametype(_make_matrix(Vp, matrix_t), Σ2)
        end
    end
end
