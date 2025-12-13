using Flux
using Flux: DataLoader
using CUDA
using MLDatasets
using BenchmarkTools
using Statistics
using NNlib     # GPU-backed pooling + conv

include("dense_serial.jl")
using .DenseOps

include("conv2d_serial_gpu.jl")
include("conv2d_multithread_gpu.jl")

############################################################
# Custom CPU Dense (Serial vs Threaded Speedup)
############################################################

struct MyDense
    W::Matrix{Float32}
    b::Vector{Float32}
end

Flux.@functor MyDense

function (m::MyDense)(x::AbstractArray)
    A = permutedims(x, (2,1))  # (batch, features)

    t_serial = @elapsed out_serial = DenseOps.dense_serial(A, m.W, m.b)
    t_threaded = @elapsed out_threaded = DenseOps.dense_threaded(A, m.W, m.b)

    speedup = t_serial / t_threaded

    println("→ CPU Dense: serial=$(round(t_serial*1000, digits=2)) ms, ",
            "threaded=$(round(t_threaded*1000, digits=2)) ms, ",
            "speedup=$(round(speedup, digits=2))×")

    return permutedims(out_threaded, (2,1))
end

MyDense(in_dim::Int, out_dim::Int) = MyDense(
    rand(Float32, in_dim, out_dim),
    rand(Float32, out_dim)
)

############################################################
# TimedConvBlock — ONLY conv + ReLU (no pooling!)
############################################################

struct TimedConvBlock
    weights::Vector{CuArray{Float32,4}}
    biases::Vector{CuArray{Float32,1}}
end

function (m::TimedConvBlock)(x)
    n = length(m.weights)
    total_speedup = 0.0

    for i in 1:n
        w = m.weights[i]
        b = m.biases[i]

        # SERIAL
        t_serial = @elapsed y_s = conv2d_serial_gpu(x, w, b)
        y_s = nothing
        GC.gc()
        #CUDA.reclaim()

        # THREADED
        t_threaded = @elapsed y_t = conv2d_multithread_gpu(x, w, b)

        sp = t_serial / t_threaded
        total_speedup += sp

        println("→ Conv layer $i: serial=$(round(t_serial*1000,digits=2)) ms, ",
                "threaded=$(round(t_threaded*1000,digits=2)) ms, ",
                "speedup=$(round(sp,digits=2))×")
       

        # ReLU on GPU
        #x = relu.(y_t)
        #CUDA.@sync relu.(y_t)
        x = y_t

        x = permutedims(x, (3,1,2,4))


        
    end

    #println("→ Conv block avg speedup: $(round(total_speedup/n, digits=2))×")
    return x
end

############################################################
# VGG16 Model Structure (Pooling outside blocks)
############################################################

struct VGG16
    block1
    block2
    block3
    block4
    block5
    classifier::Chain
end

function (m::VGG16)(x)
    pool = MaxPool((2,2))

    
    println("→ Forward Pass: GPU input")
    x = cu(x)  # move input to GPU
    

    println("→ Block 1")
    x = m.block1(x)
    x = pool(x)

    println("→ Block 2")
    x = m.block2(x)
    x = pool(x)

    println("→ Block 3")
    x = m.block3(x)
    x = pool(x)

    println("→ Block 4")
    x = m.block4(x)
    x = pool(x)

    println("→ Block 5")
    x = m.block5(x)
    x = pool(x)

    # Move activations to CPU before pooling
    println("→ Move activations to CPU for pooling and Dense")
    x = cpu(x)

    # Apply pooling once at the end
    #println("→ Pooling on CPU")
    #x = NNlib.maxpool(x, (2,2))

    # Flatten and run classifier
    println("→ Flatten + Dense classifier")
    #x = Flux.flatten(x)
    # Flatten per sample
    x = reshape(x, :, size(x, 4))  # (features, batch_size)
    return m.classifier(x)
end



############################################################
# Initialize VGG16 Conv Blocks (GPU)
############################################################

block1 = TimedConvBlock(
    [cu(rand(Float32,3,3,3,64)), cu(rand(Float32,3,3,64,64))],
    [cu(rand(Float32,64)), cu(rand(Float32,64))]
)

block2 = TimedConvBlock(
    [cu(rand(Float32,3,3,64,128)), cu(rand(Float32,3,3,128,128))],
    [cu(rand(Float32,128)), cu(rand(Float32,128))]
)

block3 = TimedConvBlock(
    [cu(rand(Float32,3,3,128,256)),
     cu(rand(Float32,3,3,256,256)),
     cu(rand(Float32,3,3,256,256))],
    [cu(rand(Float32,256)), cu(rand(Float32,256)), cu(rand(Float32,256))]
)

block4 = TimedConvBlock(
    [cu(rand(Float32,3,3,256,512)),
     cu(rand(Float32,3,3,512,512)),
     cu(rand(Float32,3,3,512,512))],
    [cu(rand(Float32,512)), cu(rand(Float32,512)), cu(rand(Float32,512))]
)

block5 = TimedConvBlock(
    [cu(rand(Float32,3,3,512,512)),
     cu(rand(Float32,3,3,512,512)),
     cu(rand(Float32,3,3,512,512))],
    [cu(rand(Float32,512)), cu(rand(Float32,512)), cu(rand(Float32,512))]
)

############################################################
# Classifier (CPU)
############################################################

# classifier = Chain(
#     MyDense(131072, 4096), relu,
#     MyDense(4096, 4096), relu,
#     MyDense(4096, 10)
# )
classifier = Chain(
    Dense(512, 4096), relu,
    Dense(4096, 4096), relu,
    Dense(4096, 10)
)

model = VGG16(block1, block2, block3, block4, block5, classifier)

############################################################
# CIFAR-10 Loader
############################################################

train_x, train_y = CIFAR10.traindata(Float32)
train_x ./= 255.0

train_loader = DataLoader((train_x, train_y), batchsize=8, shuffle=true)

############################################################
# Inference Loop
############################################################

println("=== Running Inference ===")

for (i, (x_batch, y_batch)) in enumerate(train_loader)
    println("\n----------------------")
    println("Batch $i")
    println("----------------------")

    y_pred = model(x_batch)

    println("Batch $i complete. Output size: ", size(y_pred))

    if i ≥ 5
        println("Stopping after 5 batches.")
        break
    end
end

println("\n=== DONE ===")

