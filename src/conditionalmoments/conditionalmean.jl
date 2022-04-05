

abstract type AbstractConditionalMean{T<:Number} end


eltype(μ::AbstractConditionalMean{T}) where T = T


abstract type AbstractAffineMap{T} <: AbstractConditionalMean{T} end

# representing x -> A*x + b
struct AffineMap{T,U,V} <: AbstractAffineMap{T}
    A::U
    b::V
    function AffineMap(A::AbstractMatrix,b::AbstractVector)
        nrow = size(A,1)
        nb = length(b)
        if nb != nrow
            error("The number of rows in A $(nrow) is not equal to the number of elements in b $(nb)")
        else
            T = promote_type(eltype(A),eltype(b))
            A = convert(AbstractMatrix{T},A)
            b = convert(AbstractVector{T},b)
            new{T,typeof(A),typeof(b)}(A,b)
        end
    end
end

# representing x -> A*x
struct LinearMap{T,U} <: AbstractAffineMap{T}
    A::U
    LinearMap(A::AbstractMatrix) = new{eltype(A),typeof(A)}(A)
end

nin(M::AbstractAffineMap) = size(M.A,2)
nout(M::AbstractAffineMap) = size(M.A,1)

slope(M::AffineMap) = M.A
slope(M::LinearMap) = M.A

intercept(M::AffineMap) = M.b
intercept(M::LinearMap) = zeros(eltype(M),nout(M))

(M::AffineMap)(x) = A*x + b
(M::LinearMap)(x) = A*x

compose(M2::AbstractAffineMap,M1::AbstractAffineMap) = AffineMap(slope(M2)*slope(M1), slope(M2)*intercept(M1) + intercept(M2)) # fallback
compose(M2::AbstractAffineMap,M1::LinearMap) = AffineMap(slope(M2)*slope(M1), intercept(M2))
compose(M2::LinearMap,M1::AbstractAffineMap) = AffineMap(slope(M2)*slope(M1), slope(M2)*intercept(M1))
compose(M2::LinearMap,M1::LinearMap) = LinearMap(slope(M2)*slope(M1))

*(M2::AbstractAffineMap,M1::AbstractAffineMap) = compose(M2,M1)














