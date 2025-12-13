using CUDA

########################################################
# Serial kernel: one thread per output element
# Straightforward, no shared memory or tiling
########################################################
function conv2d_kernel_serial!(
    y::CuDeviceArray{Float32,4},
    x::CuDeviceArray{Float32,4},
    w::CuDeviceArray{Float32,4},
    b::CuDeviceVector{Float32},
    H::Int, W::Int, C_in::Int,
    kh::Int, kw::Int,
    H_out::Int, W_out::Int, C_out::Int, N::Int
)
    tx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    ty = (blockIdx().y-1) * blockDim().y + threadIdx().y
    tz = (blockIdx().z-1) * blockDim().z + threadIdx().z + 1  # 1-based

    stride = blockDim().z * gridDim().z
    while tz <= C_out * N
        co = fld(tz-1, N) + 1
        n  = mod(tz-1, N) + 1

        if tx <= W_out && ty <= H_out
            acc = 0.0f0
            @inbounds for ci in 1:C_in, ki in 1:kh, kj in 1:kw
                x_i = ty + ki - 1
                x_j = tx + kj - 1
                if 1 <= x_i <= H && 1 <= x_j <= W
                    w_ki = kh - ki + 1
                    w_kj = kw - kj + 1
                    x_idx = ((n-1)*C_in + (ci-1))*H*W + (x_i-1)*W + x_j
                    w_idx = ((co-1)*C_in + (ci-1))*kh*kw + (w_ki-1)*kw + w_kj
                    acc += x[x_idx] * w[w_idx]
                end
            end
            y_idx = ((n-1)*C_out + (co-1))*H_out*W_out + (ty-1)*W_out + tx
            y[y_idx] = acc + b[co]
        end

        tz += stride
    end
    return nothing
end

function conv2d_serial_gpu(x_d::CuArray{Float32,4}, w_d::CuArray{Float32,4}, b_d::CuArray{Float32,1})
    H, W, C_in, N = size(x_d)
    kh, kw, _, C_out = size(w_d)
    H_out = H - kh + 1
    W_out = W - kw + 1

    y_d = CUDA.zeros(Float32, H_out, W_out, C_out, N)

    threads = (16,8,8)
    blocks = (cld(W_out,16), cld(H_out,8), min(C_out*N, 1024))  # safe z-dimension

    @cuda threads=threads blocks=blocks conv2d_kernel_serial!(y_d, x_d, w_d, b_d,
        H, W, C_in, kh, kw, H_out, W_out, C_out, N)
    synchronize()
    return y_d
end