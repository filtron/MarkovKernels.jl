

abstract type AbstractAffineMap{T<:Number}  end

eltype(μ::AbstractAffineMap{T}) where T = T

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

# represents x -> prior + Φ*(x - pred)
struct AffineCorrector{T,U,V,W} <: AbstractAffineMap{T}
    Φ::U
    prior::V
    pred::W
    function AffineCorrector(Φ::AbstractMatrix,prior::AbstractVector,pred::AbstractVector)
        T = promote_type(eltype(Φ),eltype(prior),eltype(pred))
        new{T,typeof(Φ),typeof(prior),typeof(pred)}(Φ,prior,pred)
    end
end

# representing x -> Φ*x
struct LinearMap{T,U} <: AbstractAffineMap{T}
    Φ::U
    LinearMap(Φ::AbstractMatrix) = new{eltype(Φ),typeof(Φ)}(Φ)
end

nin(M::AbstractAffineMap) = size(M.Φ,2)
nout(M::AbstractAffineMap) = size(M.Φ,1)

slope(M::AffineMap) = M.Φ
slope(M::LinearMap) = M.Φ
slope(M::AffineCorrector) = M.Φ

intercept(M::AffineMap) = M.b
intercept(M::LinearMap) = zeros(eltype(M),nout(M))
intercept(M::AffineCorrector) = M.prior - M.Φ*M.pred

(M::AbstractAffineMap)(x) = slope(M)*x + intercept(M)
(M::AffineCorrector)(x) = M.prior + M.Φ*(x - M.pred)


stein(Σ::Hermitian,M::AbstractAffineMap,Q::Hermitian) = Hermitian(slope(M)*Σ*slope(M)' + Q)

compose(M2::AbstractAffineMap,M1::AbstractAffineMap) = AffineMap(slope(M2)*slope(M1), slope(M2)*intercept(M1) + intercept(M2))
*(M2::AbstractAffineMap,M1::AbstractAffineMap) = compose(M2,M1)














