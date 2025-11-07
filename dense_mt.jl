function dense_forward(A::Array{Float32,2}, W::Array{Float32,2}, b::Array{Float32,1})
    # A: input batch (batch_size × input_dim)
    # W: weight matrix (input_dim × output_dim)
    # b: bias (output_dim)
    
    # Pre-allocate output
    output = zeros(Float32, size(A,1), size(W,2))
    
    Threads.@threads for i in 1:size(A,1)   # parallelize over batch
        for j in 1:size(W,2)
            output[i,j] = sum(A[i,:] .* W[:,j]) + b[j]
        end
    end
    
    return output
end

