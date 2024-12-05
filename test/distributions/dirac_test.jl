@safetestset "Dirac" begin
    using MarkovKernels, LinearAlgebra
    etys = (Float64, Complex{Float64})
    n = 2

    for T in etys
        @testset "UvDirac | $(T)" begin
            μ = randn(T)
            D = Dirac(μ)

            @test_nowarn repr(D)

            @test sample_type(D) == typeof(μ)
            @test sample_eltype(D) == eltype(μ)

            @test typeof(copy(D)) === typeof(D)
            @test dim(D) == 1
            @test mean(D) == μ
            @test rand(D) == mean(D)
            @test typeof(rand(D)) == sample_type(D)
            @test eltype(rand(D)) == sample_eltype(D)
        end

        @testset "Dirac | $(T) " begin
            μ = randn(T, n)
            D = Dirac(μ)

            @test_nowarn repr(D)

            @test sample_type(D) == typeof(μ)
            @test sample_eltype(D) == eltype(μ)

            @test !(copy(D) === D)
            @test typeof(copy(D)) === typeof(D)
            @test typeof(similar(D)) === typeof(D)
            @test copy!(similar(D), D) == D

            @test dim(D) == n
            @test mean(D) == μ
            @test rand(D) == mean(D)
            @test typeof(rand(D)) == sample_type(D)
            @test eltype(rand(D)) == sample_eltype(D)
        end
    end
end
