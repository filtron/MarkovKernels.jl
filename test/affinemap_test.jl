function affinemap_test(T, n, affine_types, matrix_types)
    A1p = randn(T, n, n)
    b1p = randn(T, n)
    c1p = randn(T, n)
    xp = randn(T, n)

    Us = T <: Real ? (Float32, Float64) : (ComplexF32, ComplexF64)

    for affine_t in affine_types, matrix_t in matrix_types
        A, b, c = to_affine_parameters(A1p, b1p, c1p, affine_t, matrix_t)
        F = to_affine_map(A, b, c, affine_t)

        if affine_t == LinearMap
            @test_nowarn (At,) = F
        elseif affine_t == AffineMap
            @test_nowarn (At, bt) = F
        elseif affine_t == AffineCorrector
            @test_nowarn (At, bt, ct) = F
        end

        x = _make_vector(xp, matrix_t)

        @testset "AffineMap | Unary | $(T) | $(affine_t) | $(matrix_t)" begin
            @test_nowarn repr(F)
            @test eltype(F) == T
            @test convert(typeof(F), F) == F

            @test !(copy(F) === F)
            @test typeof(copy(F)) === typeof(F)
            @test typeof(similar(F)) === typeof(F)
            @test copy!(similar(F), F) == F

            for U in Us
                eltype(AbstractAffineMap{U}(F)) == U
                convert(AbstractAffineMap{U}, F) == AbstractAffineMap{U}(F)
            end
            @test AbstractAffineMap{T}(F) == F
            @test convert(AbstractAffineMap{T}, F) == F
            @test slope(F) == A
            @test intercept(F) ≈ b - A * c
            @test F(x) ≈ slope(F) * x + intercept(F)
            @test _ofsametype(x, F(x))
        end
    end

    A2p = randn(T, n, n)
    b2p = randn(T, n)
    c2p = randn(T, n)

    for affine_t1 in affine_types, affine_t2 in affine_types, matrix_t in matrix_types
        A1, b1, c1 = to_affine_parameters(A1p, b1p, c1p, affine_t1, matrix_t)
        F1 = to_affine_map(A1, b1, c1, affine_t1)
        A2, b2, c2 = to_affine_parameters(A2p, b2p, c2p, affine_t2, matrix_t)
        F2 = to_affine_map(A2, b2, c2, affine_t2)

        @testset "AffineMap | Binary | {$(T),$(affine_t1),$(matrix_t)} | {$(T),$(affine_t2),$(matrix_t)}" begin
            slope(compose(F2, F1)) ≈ A2 * A1
            intercept(compose(F2, F1)) ≈ b2 + A2 * (b1 - A1 * c1 - c2)
            compose(F2, F1) == F2 * F1
        end
    end
end

function to_affine_parameters(
    A::AbstractMatrix,
    b::AbstractVector,
    c::AbstractVector,
    ::Type{LinearMap},
    matrix_t,
)
    A = _make_matrix(A, matrix_t)
    b = _make_vector(zero(b), matrix_t)
    c = _make_vector(zero(c), matrix_t)
    return A, b, c
end

function to_affine_parameters(
    A::AbstractMatrix,
    b::AbstractVector,
    c::AbstractVector,
    ::Type{AffineMap},
    matrix_t,
)
    A = _make_matrix(A, matrix_t)
    b = _make_vector(b, matrix_t)
    c = _make_vector(zero(c), matrix_t)
    return A, b, c
end

function to_affine_parameters(
    A::AbstractMatrix,
    b::AbstractVector,
    c::AbstractVector,
    ::Type{AffineCorrector},
    matrix_t,
)
    A = _make_matrix(A, matrix_t)
    b = _make_vector(b, matrix_t)
    c = _make_vector(c, matrix_t)
    return A, b, c
end

to_affine_map(A::AbstractMatrix, b::AbstractVector, c::AbstractVector, ::Type{LinearMap}) =
    LinearMap(A)
to_affine_map(A::AbstractMatrix, b::AbstractVector, c::AbstractVector, ::Type{AffineMap}) =
    AffineMap(A, b)
to_affine_map(
    A::AbstractMatrix,
    b::AbstractVector,
    c::AbstractVector,
    ::Type{AffineCorrector},
) = AffineCorrector(A, b, c)
