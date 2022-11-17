
function dirac_test(T, n)
    μ = randn(T, n)
    D = Dirac(μ)

    eltypes = T <: Real ? (Float32, Float64) : (ComplexF32, ComplexF64)

    @testset "Dirac | $(T) " begin
        @test_nowarn repr(D)
        @test eltype(D) == T
        @test dim(D) == n
        for U in eltypes
            @test AbstractDistribution{U}(D) == AbstractDirac{U}(D) == Dirac{U}(D)
            @test eltype(AbstractDirac{U}(D)) == U
        end
        @test mean(D) == μ
        @test rand(D) == mean(D)
    end

    # Dirac over trjaectories
    μ = [[T(k)] for k in 1:5]
    D = Dirac(μ)

    @testset "TrajectoryDirac | $(T) | eltype(D)" begin
        for U in eltypes
            @test AbstractDistribution{U}(D) == AbstractDirac{U}(D) == Dirac{U}(D)
            @test eltype(AbstractDirac{U}(D)) == U
        end
    end
end
