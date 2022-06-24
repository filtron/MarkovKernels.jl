
abstract type AbstractConditionalMean{T<:Number}  end

eltype(M::AbstractConditionalMean{T}) where T = T