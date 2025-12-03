using CUDA

function conv2d_gpu_kernel(x, w, b, y)
    H, W, C_in, N = size(x)
    kh, kw, _, C_out = size(w)

    @cuda threads=(16,16,1) blocks=(ceil(Int,H/16), ceil(Int,W/16), C_out) -> begin
        i = (blockIdx().x-1)*blockDim().x + threadIdx().x
        j = (blockIdx().y-1)*blockDim().y + threadIdx().y
        c_out = blockIdx().z

        if i <= H && j <= W
            for n in 1:N
                acc = b[c_out]
                for ci in 1:C_in
                    for ki in 1:kh
                        for kj in 1:kw
                            if i+ki-1 <= H && j+kj-1 <= W
                                acc += x[i+ki-1,j+kj-1,ci,n] * w[ki,kj,ci,c_out]
                            end
                        end
                    end
                end
                y[i,j,c_out,n] = acc
            end
        end
    end
end

function conv2d_gpu(x_h, w_h, b_h)
    # Allocate GPU arrays
    x_d = CuArray(x_h)
    w_d = CuArray(w_h)
    b_d = CuArray(b_h)
    y_d = CuArray(zeros(Float32, size(x_h,1), size(x_h,2), size(w_h,4), size(x_h,4)))

    conv2d_gpu_kernel(x_d, w_d, b_d, y_d)

    return y_d  # still on GPU
end

