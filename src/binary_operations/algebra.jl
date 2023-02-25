+(v::AbstractVector{T}, N::Normal{T}) where {T} = Normal(v + mean(N), covp(N))
+(v::AbstractVector{T}, D::Dirac{T}) where {T} = Dirac(v + mean(D))

"""
-(D::AbstractDistribution)

Computes the image distribution of D under negation, i.e. if 
x ∼ D then -x ∼ -D. 
"""
-(N::Normal) = Normal(-mean(N), covp(N)) # should techically be in src/distributions
-(D::Dirac) = Dirac(-mean(D)) # should techically be in src/distributions

"""
+(v::AbstractVector{T}, D::AbstractDistribution{T})
+(D::AbstractDistribution{T}, v::AbstractVector{T})

Computes a translation of D by v, i.e. if 
x ∼ D then x + v ∼ D + v.
"""
+(D::AbstractDistribution{T}, v::AbstractVector{T}) where {T} = +(v, D)

"""
-(v::AbstractVector{T}, D::AbstractDistribution{T})

Equivalent to +(v, -D).

-(D::AbstractDistribution{T}, v::AbstractVector{T})

Equivalent to +(D, -v).
"""
-(v::AbstractVector{T}, D::AbstractDistribution{T}) where {T} = +(v, -D)
-(D::AbstractDistribution{T}, v::AbstractVector{T}) where {T} = +(D, -v)

"""
    *(C::AbstractMatrix{T}, D::AbstractDistribution{T})

Equivalent to marginalize(D, DiracKernel(C)). 
"""
*(C::AbstractMatrix{T}, D::AbstractDistribution{T}) where {T} =
    marginalize(D, DiracKernel(C))
