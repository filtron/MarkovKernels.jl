@safetestset "Categorical" begin
    using MarkovKernels, LinearAlgebra
    etys = (Float64,)

    for T in etys
        p1 = T[1, 2, 3]
        p1 = p1 / sum(p1)
        C1 = Categorical(p1)

        p2 = T[3, 2, 1]
        p2 = p2 / sum(p2)
        C2 = Categorical(p2)

        p3 = T[2, 1]
        p3 = p3 / sum(p3)
        C3 = Categorical(p3)

        @test_nowarn repr(C1)
        @test copy(C1) == C1
        @test !(copy(C1) === C1)
        @test typeof(similar(C1)) == typeof(C1)

        @test dim(C1) == 1
        @test sample_type(C1) <: Int
        @test sample_eltype(C1) <: Int

        @test all(splat(isapprox), zip([logpdf(C1, x) for x in eachindex(p1)], log.(p1)))
        @test entropy(C1) ≈ -dot(log.(p1), p1)

        @test typeof(kldivergence(C1, C2)) <: Real
        @test kldivergence(C1, C2) ≈ dot(log.(p1 ./ p2), p1)
        @test kldivergence(C1, C3) == kldivergence(C2, C3) == Inf
    end
end
