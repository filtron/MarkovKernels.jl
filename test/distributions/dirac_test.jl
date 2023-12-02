
function dirac_test(T, n)
    μ = randn(T, n)
    D = Dirac(μ)

    compatible_eltypes = T <: Real ? (Float32, Float64) : (ComplexF32, ComplexF64)
    incompatible_eltypes = T <: Real ? (ComplexF32, ComplexF64) : (Float32, Float64)
    @testset "Dirac | $(T) " begin
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
            @test_throws ErrorException AbstractDistribution{U}(D)
            @test_throws ErrorException AbstractDirac{U}(D)
            @test_throws ErrorException Dirac{U}(D)
        end
        @test mean(D) == μ
        @test rand(D) == mean(D)
        @test typeof(rand(D)) == typeof_sample(D)
        @test eltype(rand(D)) == eltype_sample(D)
    end

    # Dirac over trajectories
    μ = [[T(k)] for k in 1:5]
    D = Dirac(μ)

    @testset "TrajectoryDirac | $(T) | eltype(D)" begin
        @test eltype(D) == T
        @test mean(D) == μ
        for U in compatible_eltypes
            @test AbstractDistribution{U}(D) == AbstractDirac{U}(D) == Dirac{U}(D)
            @test eltype(AbstractDirac{U}(D)) == U
        end
        for U in incompatible_eltypes
            @test_throws ErrorException AbstractDistribution{U}(D)
            @test_throws ErrorException AbstractDirac{U}(D)
            @test_throws ErrorException Dirac{U}(D)
        end
    end
end
