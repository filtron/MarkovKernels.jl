"""
    IdentityKernel

Struct for representing kernels that act like identity under marginalization.
"""
struct IdentityKernel <: AbstractDiracKernel end

mean(::IdentityKernel) = identity
