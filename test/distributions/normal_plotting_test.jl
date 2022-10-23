function normal_plotting_test()
    K = 100
    ks = 1:K

    covs = [Diagonal([n / K, n / K]) for n in ks]
    means = [[1 - n / K, n / K] for n in ks]

    Ns = map(x -> Normal(x...), zip(means, covs))

    @testset "Normal plotting" begin
        @test_nowarn plot(ks, Ns)
        @test_throws ErrorException plot([Normal(zeros(k), 1.0I(k)) for k in 1:3])
    end
end
