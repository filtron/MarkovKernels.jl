@safetestset "StochasticMatrix" begin
    using MarkovKernels, LinearAlgebra
    etys = (Float64,)
    m = 2

    for T in etys
        P = ones(T, m, m) / m
        K = StochasticMatrix(P)
        xs = 1:m

        @test AbstractStochasticMatrix <: AbstractMarkovKernel
        @test StochasticMatrix <: AbstractStochasticMatrix

        @test_nowarn repr(K)
        @test typeof(K) <: StochasticMatrix

        ps = [probability_vector(condition(K, x)) for x in xs]
        @test all(splat(isapprox), zip(ps, eachcol(P)))

        @test sample_type(condition(K, 1)) == typeof(rand(K, 1))
        @test sample_eltype(condition(K, 1)) == eltype(rand(K, 1))
    end
end
