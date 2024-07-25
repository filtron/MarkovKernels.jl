@testset "CovarianceParameter" begin
    n = 1
    m = 2

    etys = (Float64, Complex{Float64})
    matrix_types = (Matrix,)
    affine_types = (LinearMap, AffineMap, AffineCorrector)
    cov_types = (HermOrSym, Cholesky)

    for T in etys
        Vp = tril(ones(T, m, m))
        Vp = Vp * Vp'
        Cp = ones(T, n, m)
        Rp = ones(T, n, n)

        @testset "CovarianceParameter | Scalar | Unary | $(T)" begin
            l = rand(T)
            v = abs2(l)
            @test typeof(lsqrt(v)) <: real(T)
            @test v ≈ l * adjoint(l)
        end

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
end
