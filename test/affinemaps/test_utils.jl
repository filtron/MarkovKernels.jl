to_affine_parameters(::Type{LinearMap}, A, b, c) = A, zero(b), zero(c)
to_affine_parameters(::Type{AffineMap}, A, b, c) = A, b, zero(c)
to_affine_parameters(::Type{AffineCorrector}, A, b, c) = A, b, c

to_affine_map(::Type{LinearMap}, A, b, c) = LinearMap(A)
to_affine_map(::Type{AffineMap}, A, b, c) = AffineMap(A, b)
to_affine_map(::Type{AffineCorrector}, A, b, c) = AffineCorrector(A, b, c)
