import time
import torch
import numpy as np

# Setup PyJulia with compiled_modules=False to avoid static linking issues
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main

# Load your Julia kernel script
Main.include("conv2d_gpu.jl")

# Set sizes
batch = 8
in_channels = 3
out_channels = 16
height = 64
width = 64
kernel_size = 3

# Random input
x_np = np.random.randn(batch, in_channels, height, width).astype(np.float32)
w_np = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32)
b_np = np.random.randn(out_channels).astype(np.float32)

# Move arrays to Julia
Main.x_h = x_np
Main.w_h = w_np
Main.b_h = b_np

# Run Julia GPU conv
start = time.time()
y_julia = Main.conv2d_gpu(Main.x_h, Main.w_h, Main.b_h)
end = time.time()
julia_time = end - start
print(f"Julia handwritten GPU conv time: {julia_time:.6f} s")

# Convert numpy arrays to PyTorch tensors on GPU
x_torch = torch.tensor(x_np, device="cuda")
w_torch = torch.tensor(w_np, device="cuda")
b_torch = torch.tensor(b_np, device="cuda")

# PyTorch conv
start = time.time()
y_torch = torch.nn.functional.conv2d(x_torch, w_torch, bias=b_torch)
torch.cuda.synchronize()
end = time.time()
torch_time = end - start
print(f"PyTorch GPU conv time: {torch_time:.6f} s")

# Speedup
speedup = torch_time / julia_time
print(f"Speedup (PyTorch / Julia handwritten GPU): {speedup:.2f}x")

