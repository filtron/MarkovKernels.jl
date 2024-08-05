"""
    LinearMap{T,U}

Type for representing affine maps with zero intercept.
"""
struct LinearMap{T,U} <: AbstractAffineMap{T}
    A::U
end

"""
    LinearMap(A::AbstractMatrix)

Creates a LinearMap with slope A.
"""
LinearMap(A) = LinearMap{eltype(A),typeof(A)}(A)

Base.iterate(F::LinearMap) = (F.A, Val(:done))
Base.iterate(::LinearMap, ::Val{:done}) = nothing

slope(F::LinearMap) = F.A
intercept(F::LinearMap{T}) where {T} = zeros(T, size(slope(F), 1))
intercept(::LinearMap{T,<:Adjoint}) where {T} = zero(T)
intercept(::LinearMap{T,<:Number}) where {T} = zero(T)

(F::LinearMap)(x) = slope(F) * x
compose(F2::LinearMap, F1::LinearMap) = LinearMap(slope(F2) * slope(F1))

LinearMap{T}(F::LinearMap) where {T} = LinearMap(convert(AbstractMatrix{T}, F.A))
LinearMap{T}(F::LinearMap{<:Number,<:Number}) where {T} = LinearMap(convert(T, F.A))
AbstractAffineMap{T}(F::LinearMap) where {T} = LinearMap{T}(F)

function Base.show(io::IO, F::LinearMap{T,U}) where {T,U}
    print(io, summary(F))
    print(io, "\n A = ")
    show(io, (F.A))
end
