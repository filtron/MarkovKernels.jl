psdcheck(::Diagonal) = IsPSD()
convert_psd_eltype(::Type{T}, D::Diagonal) where {T} = convert(AbstractMatrix{real(T)}, D)

rsqrt(D::Diagonal) = sqrt(D)
lsqrt(D::Diagonal) = rsqrt(D)
