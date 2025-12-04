using CUDA
using Adapt # Required for adapting PyTorch tensors to CuArray
using Printf

# Flattened minimal kernel
function conv2d_gpu_kernel!(x, b, y, N, H, W, C_out)
    idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    total = H * W * C_out * N
    if idx <= total
        # Just copy + bias
        # This is where your actual kernel logic will go.
        # Ensure you use setindex! (y[idx] = ...) which is now correctly compiled
        # because the input is a proper CuArray wrapper.
        y[idx] = 1.0f0 + b[(idx-1) % C_out + 1] # Simple bias check
    end
    return
end

# x_d, w_d, b_d are now PyTorch GPU Tensors wrapped by PyCall
function conv2d_gpu(x_d, w_d, b_d)
    x_jl = adapt(CuArray, x_d)
    w_jl = adapt(CuArray, w_d)
    b_jl = adapt(CuVector, b_d)

    # --- Debugging Step: Check sizes ---
    x_size = size(x_jl)
    w_size = size(w_jl)
    
    @printf("Julia Input (x) size: %s\n", string(x_size))
    @printf("Julia Weight (w) size: %s\n", string(w_size))
    # --- End Debugging Step ---

    # Ensure w_jl is 4D for the next line, or check the length of w_size
    if length(w_size) != 4
         # Handle error or use a default size if appropriate
         error("Weight tensor is not 4D! Received size: $(w_size)")
    end

    # Use a more explicit way to get dimensions
    H, W, C_in, N = size(x_jl)
    # Corrected extraction for the 4D weight tensor (kh, kw, C_in, C_out)
    C_out = w_size[4] 

    y_jl = CUDA.zeros(Float32, H, W, C_out, N)

    total = length(y_jl)
    threads = 256
    blocks = cld(total, threads)
    
    start_time = time()
    @cuda threads=threads blocks=blocks conv2d_gpu_kernel!(x_jl, b_jl, y_jl, N, H, W, C_out)
    CUDA.synchronize()
    elapsed_time_ms = (time() - start_time) * 1000

    return y_jl, elapsed_time_ms
end