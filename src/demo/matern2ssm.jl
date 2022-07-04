# state-space realisation of the matern process of smoothness index ν+1/2

function matern2ssm(ν,λ,σ)

# continuous-time transition parameters
A = λ*( I -2*tril(ones(dimx,dimx)) )
B = sqrt(2*λ)*ones(dimx,1)

# observation matrix
C = σ*exp.( - logfactorial.( collect(0:ν) )
- logfactorial.( collect(ν:-1:0) ) .+ 2*logfactorial(ν) .- logfactorial(2*ν)/2 ).*
(-1).^collect(0:ν)

return A, B, σ*C

end