psdcheck(::UniformScaling{<:Real}) = IsPSD()
convert_psd_eltype(::Type{T}, J::UniformScaling) where {T} = convert(real(T), real(J.λ)) * I
convert_psd_eltype(J::UniformScaling) = convert_psd_eltype(eltype(J), J)

rsqrt(J::UniformScaling) = sqrt(J.λ) * I
lsqrt(J::UniformScaling) = rsqrt(J)

stein(Σ::UniformScaling, Φ::UniformScaling) = stein(Σ.λ, Φ.λ) * I
stein(Σ::UniformScaling, Φ::UniformScaling, Q::UniformScaling) = stein(Σ.λ, Φ.λ, Q.λ) * I

function schur_reduce(Π::UniformScaling, C::UniformScaling)
    S, K, Σ = schur_reduce(Π.λ, C.λ)
    return S * I, K * I, Σ * I
end

function schur_reduce(Π::UniformScaling, C::UniformScaling, R::UniformScaling)
    S, K, Σ = schur_reduce(Π.λ, C.λ, R.λ)
    return S * I, K * I, Σ * I
end
