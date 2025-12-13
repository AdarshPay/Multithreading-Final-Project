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

using Base.Threads
using LoopVectorization

function dense_threaded(A::Matrix{Float32}, W::Matrix{Float32}, b::Vector{Float32};
                                 Ti::Int=16, Tj::Int=16)
    batch_size, in_features = size(A)
    _, out_features = size(W)

    output = Matrix{Float32}(undef, batch_size, out_features)
    Wt = transpose(W)  # (out_features, in_features), contiguous in k

    # Number of tiles in each dimension
    nTi = cld(batch_size, Ti)
    nTj = cld(out_features, Tj)

    @threads for tile_idx in 1:(nTi * nTj)
        ti = (tile_idx - 1) ÷ nTj + 1
        tj = (tile_idx - 1) % nTj + 1

        i_start = (ti - 1) * Ti + 1
        j_start = (tj - 1) * Tj + 1

        i_end = min(i_start + Ti - 1, batch_size)
        j_end = min(j_start + Tj - 1, out_features)

        @inbounds for i in i_start:i_end
            for j in j_start:j_end
                acc::Float32 = 0.0f0
                @turbo for k in 1:in_features
                    acc += A[i, k] * Wt[j, k]
                end
                output[i, j] = acc + b[j]
            end
        end
    end

    return output
end

function time_kernel(f, A, W, b)
    t = @elapsed out = f(A, W, b)
    return t, out
end

end
