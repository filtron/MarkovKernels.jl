@testset "AffineMaps | LinearMap" begin
    etys = (Float64, ComplexF64)
    m, n = 2, 3

    for T in etys
        A1 = randn(T, m, n)
        x1 = randn(T, n)
        F1 = LinearMap(A1)

        Us = T <: Real ? (Float32, Float64) : (ComplexF32, ComplexF64)

        A2 = randn(T, m, m)
        F2 = LinearMap(A2)

        A3 = adjoint(randn(T, m))
        x3 = randn(T, m)
        F3 = LinearMap(A3)

        A4 = randn(T)
        x4 = randn(T)
        F4 = LinearMap(A4)

        @testset "AffineMaps | LinearMap | $(T)" begin
            @test_nowarn (At,) = F1
            @test_nowarn repr(F1)
            @test eltype(F1) == T
            @test convert(typeof(F1), F1) == F1

            @test !(copy(F1) === F1)
            @test typeof(copy(F1)) === typeof(F1)
            @test typeof(similar(F1)) === typeof(F1)
            @test copy!(similar(F1), F1) == F1

            for U in Us
                eltype(AbstractAffineMap{U}(F1)) == U
                convert(AbstractAffineMap{U}, F1) == AbstractAffineMap{U}(F1)
            end

            @test AbstractAffineMap{T}(F1) == F1
            @test convert(AbstractAffineMap{T}, F1) == F1

            @test slope(F1) == A1
            @test intercept(F1) ≈ zeros(T, m)
            @test F1(x1) ≈ slope(F1) * x1 + intercept(F1)

            @test F3(x3) ≈ slope(F3) * x3 + intercept(F3)
            @test typeof(F3(x3)) <: T

            #
            @test F4(x4) ≈ slope(F4) * x4 + intercept(F4)
            @test typeof(F4(x4)) <: T
            #

            F21 = F2 ∘ F1
            @test slope(F21) ≈ A2 * A1
            @test intercept(F21) ≈ zeros(T, m)
            @test F21 == compose(F2, F1)

            F31 = F3 ∘ F1
            @test slope(F31) ≈ A3 * A1
            @test intercept(F31) ≈ zero(T)
            @test F31 == compose(F3, F1)

            F32 = F3 ∘ F2
            @test slope(F32) ≈ A3 * A2
            @test intercept(F32) ≈ zero(T)
            @test F32 == compose(F3, F2)

            F43 = F4 ∘ F3
            @test slope(F43) ≈ A4 * A3
            @test intercept(F43) ≈ zero(T)
            @test F43 == compose(F4, F3)
        end
    end
end
