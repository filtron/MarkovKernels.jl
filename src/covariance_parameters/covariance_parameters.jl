include("utils.jl")
include("selfadjoint.jl")
include("cholesky.jl")
include("real.jl")

# add Real
const CovarianceParameter{T} = Union{SelfAdjoint{T},Factorization{T}}

stein(Σ, A::AbstractAffineMap) = stein(Σ, slope(A))
stein(Σ, A::AbstractAffineMap, Q) = stein(Σ, slope(A), Q)

schur_reduce(Π, C::AbstractAffineMap) = schur_reduce(Π, slope(C))
schur_reduce(Π, C::AbstractAffineMap, R) = schur_reduce(Π, slope(C), R)
