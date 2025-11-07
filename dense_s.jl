# dense_serial.jl

function dense_serial(A::Array{Float32,2}, W::Array{Float32,2}, b::Array{Float32,1})
    # A: input batch (batch_size × input_dim)
    # W: weights (input_dim × output_dim)
    # b: bias (output_dim)
    
    batch_size, input_dim = size(A)
    output_dim = size(W, 2)
    
    # Pre-allocate output
    output = zeros(Float32, batch_size, output_dim)
    
    # Serial nested loops
    for i in 1:batch_size          # loop over examples in batch
        for j in 1:output_dim      # loop over output neurons
            sum_val = 0.0f0
            for k in 1:input_dim  # loop over input features
                sum_val += A[i,k] * W[k,j]
            end
            output[i,j] = sum_val + b[j]
        end
    end
    
    return output
end

