@safetestset "Dirac" begin
    using MarkovKernels, LinearAlgebra
    etys = (Float64, Complex{Float64})
    n = 2

    for T in etys
        compatible_eltypes = T <: Real ? (Float32, Float64, ComplexF32, ComplexF64) : ()
        incompatible_eltypes = T <: Complex ? (Float32, Float64) : ()

        @testset "UvDirac | $(T)" begin
            μ = randn(T)
            D = Dirac(μ)

            @test_nowarn repr(D)
            @test eltype(D) == T

            @test typeof(copy(D)) === typeof(D)
            @test dim(D) == 1

            for U in compatible_eltypes
                @test AbstractDistribution{U}(D) == AbstractDirac{U}(D) == Dirac{U}(D)
                @test eltype(AbstractDirac{U}(D)) == U
            end
            for U in incompatible_eltypes
                @test_throws InexactError AbstractDistribution{U}(D)
                @test_throws InexactError AbstractDirac{U}(D)
                @test_throws InexactError Dirac{U}(D)
            end

            @test mean(D) == μ
            @test rand(D) == mean(D)
            @test typeof(rand(D)) == sample_type(D)
            @test eltype(rand(D)) == sample_eltype(D)
        end

        @testset "Dirac | $(T) " begin
            μ = randn(T, n)
            D = Dirac(μ)

            @test_nowarn repr(D)
            @test eltype(D) == T

            @test !(copy(D) === D)
            @test typeof(copy(D)) === typeof(D)
            @test typeof(similar(D)) === typeof(D)
            @test copy!(similar(D), D) == D

            @test dim(D) == n
            for U in compatible_eltypes
                @test AbstractDistribution{U}(D) == AbstractDirac{U}(D) == Dirac{U}(D)
                @test eltype(AbstractDirac{U}(D)) == U
            end
            for U in incompatible_eltypes
                @test_throws InexactError AbstractDistribution{U}(D)
                @test_throws InexactError AbstractDirac{U}(D)
                @test_throws InexactError Dirac{U}(D)
            end
            @test mean(D) == μ
            @test rand(D) == mean(D)
            @test typeof(rand(D)) == sample_type(D)
            @test eltype(rand(D)) == sample_eltype(D)
        end
    end
end
