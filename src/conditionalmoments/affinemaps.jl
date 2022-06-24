# types for representing affine conditional means

abstract type AbstractAffineMap{T<:Number} <: AbstractConditionalMean{T}  end

(M::AbstractAffineMap)(x) = slope(M)*x + intercept(M)

# representing x -> Φ*x
struct LinearMap{T,U} <: AbstractAffineMap{T}
    Φ::U
    LinearMap(Φ::AbstractMatrix) = new{eltype(Φ),typeof(Φ)}(Φ)
end

# representing x -> Φ*(y-x)
struct LinearCorrector{T,U,V} <: AbstractAffineMap{T}
    Φ::U
    y::V
    function LinearCorrector(Φ::AbstractMatrix,y::AbstractVector)
        T = promote_type(eltype(Φ),eltype(y))
        new{T,AbstractMatrix{T},AbstractVector{T}}( convert(AbstractMatrix{T},Φ), convert(AbstractVector{T},y) )
    end
end

# representing x -> Φ*x + b
struct AffineMap{T,U,V} <: AbstractAffineMap{T}
    Φ::U
    b::V
    function AffineMap(Φ::AbstractMatrix,b::AbstractVector)
        nrow = size(Φ,1)
        nb = length(b)
        if nb != nrow
            error("The number of rows in A $(nrow) is not equal to the number of elements in b $(nb)")
        else
            T = promote_type(eltype(Φ),eltype(b))
            new{T,typeof(Φ),typeof(b)}(Φ,b)
        end
    end
end

slope(M::AffineMap) = M.Φ
slope(M::LinearMap) = M.Φ

intercept(M::AffineMap) = M.b
intercept(M::LinearMap) = zeros(eltype(M),size(slope(M),1))

stein(Σ::Hermitian,M::AbstractAffineMap,Q::Hermitian) = Hermitian(slope(M)*Σ*slope(M)' + Q)

compose(M2::AbstractAffineMap,M1::AbstractAffineMap) = AffineMap(slope(M2)*slope(M1), slope(M2)*intercept(M1) + intercept(M2))
*(M2::AbstractAffineMap,M1::AbstractAffineMap) = compose(M2,M1)
