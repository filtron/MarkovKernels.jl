
function dirac_test(T, n)
    μ = randn(T, n)
    D = Dirac(μ)

    eltypes = T <: Real ? (Float32, Float64) : (ComplexF32, ComplexF64)

    @testset "Dirac | $(T) " begin
        @test eltype(D) == T
        for U in eltypes
            AbstractDistribution{U}(D) == AbstractDirac{U}(D) == Dirac{U}(D)
            eltype(AbstractDirac{U}(D)) == U
        end
        @test mean(D) == μ
        @test cov(D) == Diagonal(zeros(T, n))
        @test var(D) == zeros(T, n)
        @test std(D) == zeros(T, n)
        @test rand(D) == mean(D)

        @test eltype(var(D)) <: Real
        @test eltype(std(D)) <: Real
    end
end