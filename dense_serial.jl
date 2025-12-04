module DenseOps
using BenchmarkTools
using Base.Threads

"""
    dense_serial(x, w, b)
    PURE SERIAL: No Threads, No Explicit SIMD
"""

function dense_serial(A::Matrix{Float32}, W::Matrix{Float32}, b::Vector{Float32})

    batch_size, in_dim = size(A)
    _, out_dim = size(W)

    # Allocate output
    out = Matrix{Float32}(undef, batch_size, out_dim)

    # Triple loop FC layer: out = A * W + b
    for i in 1:batch_size
        for j in 1:out_dim
            acc::Float32 = 0.0f0
            for k in 1:in_dim
                acc += A[i, k] * W[k, j]
            end
            out[i, j] = acc + b[j]
        end
    end

    return out
end


"""
    dense_parallel(x, w, b)
    MULTITHREADED ONLY: Threads, No Explicit SIMD
"""
function dense_threaded(A::Matrix{Float32}, W::Matrix{Float32}, b::Vector{Float32})
    batch_size, in_features = size(A)
    _, out_features = size(W)

    # Preallocate output
    output = Matrix{Float32}(undef, batch_size, out_features)

    # Transpose W for memory-friendly access
    Wt = W'  # Now Wt[j,k] instead of W[k,j], row-major for inner loop

    # Flatten i,j loops for better thread load balancing
    @threads for idx in 1:(batch_size*out_features)
        i = (idx - 1) ÷ out_features + 1
        j = (idx - 1) % out_features + 1
        acc::Float32 = 0.0f0
        @inbounds @simd for k in 1:in_features
            acc += A[i, k] * Wt[j, k]   # contiguous access along k
        end
        @inbounds output[i, j] = acc + b[j]
    end

    return output
end

function time_kernel(f, A, W, b)
    t = @elapsed out = f(A, W, b)
    return t, out
end

end