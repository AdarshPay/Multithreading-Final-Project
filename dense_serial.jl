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
                        Ti::Int = 64, Tj::Int = 64)

    batch_size, in_features = size(A)         # A: (batch, in_features)
    in_features_W, out_features = size(W)     # W: (in_features, out_features)
    @assert in_features == in_features_W "A and W have incompatible shapes"

    # Output
    output = Matrix{Float32}(undef, batch_size, out_features)

    # --- Make memory layout friendly for the k-loop ---
    # At: (in_features, batch_size), contiguous along k when indexing At[k, i]
    At = permutedims(A)  # this is a copy, but pays off for big layers

    # W is already (in_features, out_features), contiguous along k when W[k, j] with fixed j
    Wc = copy(W)         # ensure it's dense & contiguous (optional but safe)

    # Pre-fill with bias so we can use += in the @turbo loop
    @inbounds for i in 1:batch_size
        for j in 1:out_features
            output[i, j] = b[j]
        end
    end

    # Number of tiles
    nTi = cld(batch_size, Ti)
    nTj = cld(out_features, Tj)

    # --- Parallel over tiles ---
    @threads for tile_idx in 1:(nTi * nTj)
        ti = (tile_idx - 1) ÷ nTj + 1
        tj = (tile_idx - 1) % nTj + 1

        i_start = (ti - 1) * Ti + 1
        j_start = (tj - 1) * Tj + 1

        i_end = min(i_start + Ti - 1, batch_size)
        j_end = min(j_start + Tj - 1, out_features)

        @inbounds @turbo for i in i_start:i_end, j in j_start:j_end, k in 1:in_features
            # reduction over k, LV will handle vectorization
            output[i, j] += At[k, i] * Wc[k, j]
        end
    end

    return output
end


function time_kernel(f, A, W, b)
    t = @elapsed out = f(A, W, b)
    return t, out
end

end
