function affinemap_test(T, affine_types, n)
    for t in affine_types
        slopegt, interceptgt, F = _make_affinemap(T, n, n, t)
        x = randn(T, n)

        eltypes = T <: Real ? (Float32, Float64) : (ComplexF32, ComplexF64)

        @testset "AffineMap | Unary | $(T) | $(t)" begin
            @test_nowarn repr(F)
            @test eltype(F) == T
            @test convert(typeof(F), F) == F
            for U in eltypes
                eltype(AbstractAffineMap{U}(F)) == U
                convert(AbstractAffineMap{U}, F) == AbstractAffineMap{U}(F)
            end
            @test AbstractAffineMap{T}(F) == F
            @test convert(AbstractAffineMap{T}, F) == F
            @test slope(F) ≈ slopegt
            @test intercept(F) ≈ interceptgt
            @test F(x) ≈ slopegt * x + interceptgt
        end
    end

    for t1 in affine_types, t2 in affine_types
        slope1, intercept1, F1 = _make_affinemap(T, n, n, t1)
        slope2, intercept2, F2 = _make_affinemap(T, n, n, t2)

        slopegt = slope2 * slope1
        interceptgt = intercept2 + slope2 * intercept1
        @testset "AffineMap | Binary | {$(T),$(t1)} | {$(T),$(t2)}" begin
            @test slope(compose(F2, F1)) ≈ slopegt
            @test intercept(compose(F2, F1)) ≈ interceptgt
            @test compose(F2, F1) == F2 * F1
        end
    end
end

function _make_affinemap(T, n::Int, m::Int, t::Symbol)
    if t === :LinearMap
        A = randn(T, n, m)
        slope = A
        intercept = zeros(T, n)
        F = LinearMap(A)
    elseif t === :AffineMap
        A = randn(T, n, m)
        b = randn(T, n)
        slope = A
        intercept = b
        F = AffineMap(A, b)
    elseif t == :AffineCorrector
        A = randn(T, n, m)
        b = randn(T, n)
        c = randn(T, m)
        slope = A
        intercept = b - A * c
        F = AffineCorrector(A, b, c)
    end

    return slope, intercept, F
end
