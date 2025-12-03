using LoopVectorization

# Serial 2D convolution
function conv2d(x::Array{Float32,4}, w::Array{Float32,4}, b::Vector{Float32})
    # x: H x W x C_in x N
    # w: kh x kw x C_in x C_out
    H, W, C_in, N = size(x)
    kh, kw, _, C_out = size(w)
    H_out = H - kh + 1
    W_out = W - kw + 1

    # Output array
    y = zeros(Float32, H_out, W_out, C_out, N)

    # Serial convolution loop
    for n in 1:N
        for co in 1:C_out
            @turbo for i in 1:H_out, j in 1:W_out
                acc = 0.0f0
                for ci in 1:C_in
                    for ki in 1:kh
                        for kj in 1:kw
                            acc += x[i+ki-1, j+kj-1, ci, n] * w[ki, kj, ci, co]
                        end
                    end
                end
                y[i, j, co, n] = acc + b[co]
            end
        end
    end

    return y
end

