# sample many times from (vector valued) Markov kernel
function rand(RNG::AbstractRNG,K::AbstractMarkovKernel,xs)

    N = size(xs,1)

    y = rand(RNG,K,xs[1,:])
    m = length(y)

    ys = zeros(N,m)
    ys[1,:] = y

    for n in 2:N
        y = rand(RNG,K,xs[n,:])
        ys[n,:] = y
    end

    return ys

end

rand(K::AbstractMarkovKernel,xs) = rand(GLOBAL_RNG,K,xs)


# sample homogeneous (vector valued) Markov process
function rand(RNG::AbstractRNG,init::AbstractDistribution,k::AbstractMarkovKernel,N::Integer)

    x = rand(RNG,init)
    m = length(x)

    xs = zeros(N+1,m)
    xs[1,:] = x

    for n in 1:N
        x = rand(RNG,k,x)
        xs[n+1,:] = x
    end

    return xs

end

rand(init::AbstractDistribution,k::AbstractMarkovKernel,N::Integer) = rand(GLOBAL_RNG,init,k,N)

# sample hetrogeneous (vector valued) Markov process
function rand(RNG::AbstractRNG,init::AbstractDistribution,ks)

    x = rand(RNG,init)
    m = length(x)

    N = length(ks)

    xs = zeros(N+1,m)
    xs[1,:] = x

    for n in 1:N
        k = ks[n]
        x = rand(RNG,k,x)
        xs[n+1,:] = x
    end

    return xs

end

rand(init::AbstractDistribution,ks) = rand(GLOBAL_RNG,init,ks)

