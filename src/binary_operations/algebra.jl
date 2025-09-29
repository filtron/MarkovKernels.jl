+(v::AbstractNumOrVec, N::Normal) = Normal(v + mean(N), covp(N))
+(v::AbstractNumOrVec, D::Dirac) = Dirac(v + mean(D))

"""
-(D::AbstractDistribution)

Computes the image distribution of D under negation, i.e. if
x ∼ D then -x ∼ -D.
"""
-(N::Normal) = Normal(-mean(N), covp(N)) # should techically be in src/distributions
-(D::Dirac) = Dirac(-mean(D)) # should techically be in src/distributions

"""
+(v::AbstractNumOrVec, D::AbstractDistribution)
+(D::AbstractDistribution, v::AbstractNumOrVec)

Computes a translation of D by v, i.e. if
x ∼ D then x + v ∼ D + v.
"""
+(D::AbstractDistribution, v::AbstractNumOrVec) = +(v, D)

"""
-(v::AbstractNumOrVec, D::AbstractDistribution)

Equivalent to +(v, -D).

-(D::AbstractDistribution, v::AbstractNumOrVec)

Equivalent to +(D, -v).
"""
-(v::AbstractNumOrVec, D::AbstractDistribution) = +(v, -D)
-(D::AbstractDistribution, v::AbstractNumOrVec) = +(D, -v)

"""
    *(C, D::AbstractDistribution)

Equivalent to forward_operator(D, DiracKernel(LinearMap(C))).
"""
*(C, D::AbstractDistribution) = forward_operator(DiracKernel(LinearMap(C)), D)
