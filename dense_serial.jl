module DenseOps

using Base.Threads

"""
    dense_serial(x, w, b)
    PURE SERIAL: No Threads, No Explicit SIMD
"""
function dense_serial(x::Matrix{Float32}, w::Matrix{Float32}, b::Vector{Float32})
    batch_size, in_features = size(x)
    out_features = length(b)
    
    output = Array{Float32}(undef, batch_size, out_features)
    
    for i in 1:batch_size
        for j in 1:out_features
            acc = 0.0f0
            for k in 1:in_features
                #removed the @inbounds here
                acc += x[i, k] * w[k, j]
            end
            #removed the @inbounds here
            output[i, j] = acc + b[j]
        end
    end
    
    return output
end

"""
    dense_parallel(x, w, b)
    MULTITHREADED ONLY: Threads, No Explicit SIMD
"""
function dense_parallel(x::Matrix{Float32}, w::Matrix{Float32}, b::Vector{Float32})
    batch_size, in_features = size(x)
    out_features = length(b)
    
    output = Array{Float32}(undef, batch_size, out_features)
    
    # Parallelize only the outer loop
    @threads for i in 1:batch_size
        for j in 1:out_features
            acc = 0.0f0
            # REMOVED @simd here
            for k in 1:in_features
                @inbounds acc += x[i, k] * w[k, j]
            end
            @inbounds output[i, j] = acc + b[j]
        end
    end
    
    return output
end

end