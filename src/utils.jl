const AbstractNumOrVec{T} = Union{T,AbstractVector{T}} where {T<:Number}
Scalar{T} = Union{T,Base.RefValue{T}} where {T<:Number}
