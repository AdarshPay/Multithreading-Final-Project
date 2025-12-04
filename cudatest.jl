# conv2d_gpu.jl
#
# Simple single-channel 2D convolution on the GPU using CUDA.jl.
# Includes CPU reference and a correctness test.

using CUDA

###############################
# GPU KERNEL (CUDA-safe)
###############################
function conv2d_kernel!(out, input, kernel, pad_h::Int, pad_w::Int)
    T = eltype(out)

    # Thread coordinates
    tx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ty = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    H_out = size(out, 1)
    W_out = size(out, 2)

    if 1 <= ty <= H_out && 1 <= tx <= W_out
        sum = zero(T)
        Kh = size(kernel, 1)
        Kw = size(kernel, 2)

        @inbounds for r in 1:Kh
            @inbounds for c in 1:Kw
                in_r = ty + r - pad_h - 1
                in_c = tx + c - pad_w - 1
                if 1 <= in_r <= size(input,1) && 1 <= in_c <= size(input,2)
                    sum += input[in_r, in_c] * kernel[r, c]
                end
            end
        end

        out[ty, tx] = sum
    end

    return nothing
end

###############################
# HOST FUNCTION (GPU wrapper)
###############################
function gpu_conv2d(input::Array{T,2}, kernel::Array{T,2}; pad=:same) where {T<:Union{Float32,Float64}}
    H_in, W_in = size(input)
    Kh, Kw = size(kernel)

    # Determine padding
    if pad === :same
        pad_h = fld(Kh, 2)
        pad_w = fld(Kw, 2)
    elseif pad === :valid
        pad_h = 0
        pad_w = 0
    else
        error("pad must be :same or :valid")
    end

    # Output dims
    H_out = H_in + (-Kh + 1) + 2*pad_h
    W_out = W_in + (-Kw + 1) + 2*pad_w
    H_out <= 0 && error("Invalid output size – kernel too large")
    W_out <= 0 && error("Invalid output size – kernel too large")

    # Move data to GPU
    d_input  = CuArray(input)
    d_kernel = CuArray(kernel)
    d_out    = CUDA.zeros(T, H_out, W_out)

    # Launch config
    threads = (16, 16)
    blocks  = (cld(W_out, threads[1]), cld(H_out, threads[2]))

    @cuda threads=threads blocks=blocks conv2d_kernel!(d_out, d_input, d_kernel, pad_h, pad_w)
    synchronize()

    return Array(d_out)
end


###############################
# CPU REFERENCE
###############################
function cpu_conv2d(input::Array{T,2}, kernel::Array{T,2}; pad=:same) where {T}
    H_in, W_in = size(input)
    Kh, Kw = size(kernel)

    if pad === :same
        pad_h = fld(Kh, 2)
        pad_w = fld(Kw, 2)
    elseif pad === :valid
        pad_h = 0
        pad_w = 0
    else
        error("pad must be :same or :valid")
    end

    H_out = H_in + (-Kh + 1) + 2*pad_h
    W_out = W_in + (-Kw + 1) + 2*pad_w
    out = zeros(T, H_out, W_out)

    for y in 1:H_out
        for x in 1:W_out
            s = zero(T)
            for r in 1:Kh
                for c in 1:Kw
                    in_r = y + r - pad_h - 1
                    in_c = x + c - pad_w - 1
                    if 1 <= in_r <= H_in && 1 <= in_c <= W_in
                        s += input[in_r, in_c] * kernel[r,c]
                    end
                end
            end
            out[y,x] = s
        end
    end
    return out
end


###############################
# SAMPLE KERNEL
###############################
function fspecial(sz::Integer)
    if sz == 3
        return Float32[1 2 1;
                       2 4 2;
                       1 2 1] / 16
    else
        k = ones(Float32, sz, sz)
        return k ./ sum(k)
    end
end


###############################
# SELF-TEST (runs if script)
###############################
if abspath(PROGRAM_FILE) == @__FILE__
    println("Running GPU vs CPU convolution test...")

    using Random
    Random.seed!(12345)

    A = rand(Float32, 64, 64)
    K = fspecial(5)

    cpu = cpu_conv2d(A, K, pad=:same)
    gpu = gpu_conv2d(A, K, pad=:same)

    diff = maximum(abs.(cpu .- gpu))
    println("Max difference = ", diff)
    @assert diff < 1e-4

    println("SUCCESS: GPU and CPU match!")
end