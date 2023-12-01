function bayes_rule_test(T, n, m, cov_types, matrix_types)
    μp = randn(T, m)
    Vp = randn(T, m, m)
    Vp = Vp * Vp'
    Cp = randn(T, n, m)
    yp = randn(T, n)
    Rp = randn(T, n, n)
    Rp = Rp * Rp'

    for cov_t in cov_types, matrix_t in matrix_types
        C = _make_matrix(Cp, matrix_t)
        y = _make_vector(yp, matrix_t)
        R = _make_covp(_make_matrix(Rp, matrix_t), cov_t)

        μ = _make_vector(μp, matrix_t)
        Σ = _make_covp(_make_matrix(Vp, matrix_t), cov_t)
        N = Normal(μ, Σ)

        K = NormalKernel(C, R)
        L = Likelihood(K, y)
        @testset "bayes_rule | $(nameof(typeof(N))) | $(nameof(typeof(L)))" begin
            M, KC = invert(N, K)

            NC, loglike = bayes_rule(N, L)
            @test mean(NC) ≈ mean(condition(KC, y))
            @test cov(NC) ≈ cov(condition(KC, y))
            @test loglike ≈ logpdf(M, y)
        end

        K = DiracKernel(C)
        L = Likelihood(K, y)
        @testset "bayes_rule | $(nameof(typeof(N))) | $(nameof(typeof(L)))" begin
            M, KC = invert(N, K)

            NC, loglike = bayes_rule(N, L)
            @test mean(NC) ≈ mean(condition(KC, y))
            @test cov(NC) ≈ cov(condition(KC, y))
            @test loglike ≈ logpdf(M, y)
        end

        @testset "bayes_rule | $(nameof(typeof(N))) | $(nameof(typeof(L)))" begin
            NC, loglike = bayes_rule(N, FlatLikelihood())
            @test NC === N
            @test loglike === 0.0
        end
    end

    _test_bayes_rule_particle_system(T, n, m)
end

function _test_bayes_rule_particle_system(T, n, m)
    k = 10

    X = [randn(T, n) for i in 1:k]
    logws = randn(real(T), k)
    #ws = exp.(logws) / sum(exp, logws)
    P1 = ParticleSystem(logws, X)
    P2 = ParticleSystem(copy(logws), copy.(X))

    C = randn(T, m, n)
    K = NormalKernel(C, diagm(ones(T, m)))
    y = randn(T, m)
    L = Likelihood(K, y)
    loglike_gt = _loglike(logws, [log(L, X[i]) for i in eachindex(X)])
    logws_gt = logws + [log(L, X[i]) for i in eachindex(X)]
    ws_gt = exp.(logws_gt) / sum(exp, logws_gt)
    @testset "bayes_rule | $(typeof(P1)) | $(typeof(K))" begin
        PC1, loglike1 = bayes_rule(P1, L)
        loglike2 = bayes_rule!(P2, L)

        @test loglike1 ≈ loglike2 ≈ loglike_gt
        @test weights(PC1) ≈ weights(P2) ≈ ws_gt
        @test particles(P1) == particles(P2) == particles(PC1)
    end

    X2 = permutedims(X)
    P3 = ParticleSystem(logws, X2)
    P4 = ParticleSystem(copy(logws), copy.(X2))

    @testset "bayes_rule | $(typeof(P3)) | $(typeof(K))" begin
        PC3, loglike3 = bayes_rule(P3, L)
        loglike4 = bayes_rule!(P4, L)

        @test loglike3 ≈ loglike4 ≈ loglike_gt
        @test weights(PC3) ≈ weights(P4) ≈ ws_gt
        @test particles(P3) == particles(P4) == particles(PC3)
    end
end

function _loglike(logws, logls)
    logc1 = maximum(logws)
    logc2 = maximum(logws + logls)

    logs1 = log(sum(exp, logws .- logc1))
    logs2 = log(sum(exp, logls + logws .- logc2))
    return logc2 - logc1 + logs2 - logs1
end
