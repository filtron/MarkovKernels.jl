
function dirac_test(T, n)
    μ = randn(T, n)
    D = Dirac(μ)

    eltypes = T <: Real ? (Float32, Float64) : (ComplexF32, ComplexF64)

    @testset "Dirac | $(T) " begin
        @test eltype(D) == T
        @test convert(typeof(D), D) == D
        for U in eltypes
            eltype(AbstractDirac{U}(D)) == U
            convert(AbstractDirac{U}, D) == AbstractDirac{U}(D)
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
