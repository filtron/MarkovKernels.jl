+(v::AbstractVector{T}, N::Normal{T}) where {T} = Normal(v + mean(N), covp(N))
+(N::Normal{T}, v::AbstractVector{T}) where {T} = +(v, N)

-(N::Normal) = Normal(-mean(N), covp(N))
-(v::AbstractVector{T}, N::Normal{T}) where {T} = +(v, -N)

+(v::AbstractVector{T}, D::Dirac{T}) where {T} = Dirac(v + mean(D))

-(D::Dirac) = Dirac(-mean(D))
-(v::AbstractVector{T}, D::Dirac{T}) where {T} = +(v, -D)

+(D::AbstractDistribution{T}, v::AbstractVector{T}) where {T} = +(v, D)
-(D::AbstractDistribution{T}, v::AbstractVector{T}) where {T} = +(D, -v)

*(C::AbstractMatrix{T}, D::AbstractDistribution{T}) where {T} =
    marginalize(D, DiracKernel(C))
