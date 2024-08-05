@testset "AffineMaps | Mixed Type Operations" begin
    etys = (Float64, ComplexF64)
    amtys = (LinearMap, AffineMap, AffineCorrector)

    m, n = 2, 3

    for T in etys
        @testset "AffineMaps | vector2vector/vector2vector | $(T)" begin
            for TF1 in amtys, TF2 in amtys
                A1p = randn(T, m, n)
                b1p = randn(T, m)
                c1p = randn(T, n)
                A1, b1, c1 = to_affine_parameters(TF1, A1p, b1p, c1p)
                F1 = to_affine_map(TF1, A1, b1, c1)

                A2p = randn(T, m, m)
                b2p = randn(T, m)
                c2p = randn(T, m)
                A2, b2, c2 = to_affine_parameters(TF2, A2p, b2p, c2p)
                F2 = to_affine_map(TF2, A2, b2, c2)

                F21 = F2 ∘ F1
                @testset "AffineMaps | vector2vector/vector2vector | $(T) | $(TF1) | $(TF2)" begin
                    @test slope(F21) ≈ A2 * A1
                    @test intercept(F21) ≈ b2 + A2 * (b1 - A1 * c1 - c2)
                    @test compose(F2, F1) == F21
                end
            end
        end

        @testset "AffineMaps | vector2scalar/vector2vector | $(T)" begin
            for TF1 in amtys, TF2 in amtys
                A1p = randn(T, m, n)
                b1p = randn(T, m)
                c1p = randn(T, n)
                A1, b1, c1 = to_affine_parameters(TF1, A1p, b1p, c1p)
                F1 = to_affine_map(TF1, A1, b1, c1)

                A2p = adjoint(randn(T, m))
                b2p = randn(T)
                c2p = randn(T, m)
                A2, b2, c2 = to_affine_parameters(TF2, A2p, b2p, c2p)
                F2 = to_affine_map(TF2, A2, b2, c2)

                F21 = F2 ∘ F1
                @testset "AffineMaps | vector2scalar/vector2vector | $(T) | $(TF1) | $(TF2)" begin
                    @test slope(F21) ≈ A2 * A1
                    @test intercept(F21) ≈ b2 + A2 * (b1 - A1 * c1 - c2)
                    @test compose(F2, F1) == F21
                end
            end
        end

        @testset "AffineMaps | scalar2scalar/vector2scalar | $(T)" begin
            for TF1 in amtys, TF2 in amtys
                A1p = adjoint(randn(T, n))
                b1p = randn(T)
                c1p = randn(T, n)
                A1, b1, c1 = to_affine_parameters(TF1, A1p, b1p, c1p)
                F1 = to_affine_map(TF1, A1, b1, c1)

                A2p = randn(T)
                b2p = randn(T)
                c2p = randn(T)
                A2, b2, c2 = to_affine_parameters(TF2, A2p, b2p, c2p)
                F2 = to_affine_map(TF2, A2, b2, c2)

                F21 = F2 ∘ F1
                @testset "AffineMaps | scalar2scalar/vector2scalar | $(T) | $(TF1) | $(TF2)" begin
                    @test slope(F21) ≈ A2 * A1
                    @test intercept(F21) ≈ b2 + A2 * (b1 - A1 * c1 - c2)
                    @test compose(F2, F1) == F21
                end
            end
        end

        @testset "AffineMaps | scalar2scalar/scalar2scalar | $(T)" begin
            for TF1 in amtys, TF2 in amtys
                A1p = randn(T)
                b1p = randn(T)
                c1p = randn(T)
                A1, b1, c1 = to_affine_parameters(TF1, A1p, b1p, c1p)
                F1 = to_affine_map(TF1, A1, b1, c1)

                A2p = randn(T)
                b2p = randn(T)
                c2p = randn(T)
                A2, b2, c2 = to_affine_parameters(TF2, A2p, b2p, c2p)
                F2 = to_affine_map(TF2, A2, b2, c2)

                F21 = F2 ∘ F1
                @testset "AffineMaps | scalar2scalar/scalar2scalar | $(T) | $(TF1) | $(TF2)" begin
                    @test slope(F21) ≈ A2 * A1
                    @test intercept(F21) ≈ b2 + A2 * (b1 - A1 * c1 - c2)
                    @test compose(F2, F1) == F21
                end
            end
        end
    end
end
