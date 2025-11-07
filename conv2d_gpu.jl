using CUDA

function conv2d_gpu(input::CuArray{Float32,4}, weight::CuArray{Float32,4}, bias::CuArray{Float32,1})
    # input: (batch, channels, height, width)
    # weight: (out_channels, in_channels, kh, kw)
    
    batch, in_ch, h, w = size(input)
    out_ch, _, kh, kw = size(weight)
    out_h, out_w = h - kh + 1, w - kw + 1
    
    output = CuArray{Float32}(undef, batch, out_ch, out_h, out_w)
    
    @cuda threads=256 begin
        for n in 1:batch
            for oc in 1:out_ch
                for ic in 1:in_ch
                    # simple nested loop convolution
                    for i in 1:out_h
                        for j in 1:out_w
                            output[n,oc,i,j] += sum(input[n,ic,i:i+kh-1,j:j+kw-1] .* weight[oc,ic,:,:])
                        end
                    end
                end
                output[n,oc,:,:] .+= bias[oc]
            end
        end
    end
    
    return output
end

