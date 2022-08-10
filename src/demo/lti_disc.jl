# compute discrete-time transitions from continuous-time state space

function lti_disc(A, B, dt)
    dimx = size(B, 1)

    W = exp([A Hermitian(B * B'); zeros(dimx, dimx) -A'] * dt)
    Φ = W[1:dimx, 1:dimx]
    Q = Matrix(Hermitian(W[1:dimx, dimx+1:2*dimx] * Φ'))

    return Φ, Q
end
