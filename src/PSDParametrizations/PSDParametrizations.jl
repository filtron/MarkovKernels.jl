
abstract type PSDTrait end
struct IsPSD <: PSDTrait end
struct IsNotPSD <: PSDTrait end

include("utils.jl")
include("interface.jl")
include("scalar.jl")
include("uniformscaling.jl")
include("diagonal.jl")
include("selfadjoint.jl")
include("cholesky.jl")

stein(Σ, A::AbstractAffineMap) = stein(Σ, slope(A))
stein(Σ, A::AbstractAffineMap, Q) = stein(Σ, slope(A), Q)

schur_reduce(Π, C::AbstractAffineMap) = schur_reduce(Π, slope(C))
schur_reduce(Π, C::AbstractAffineMap, R) = schur_reduce(Π, slope(C), R)
