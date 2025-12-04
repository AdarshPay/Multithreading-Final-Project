import numpy as np
import torch
import torch.nn.functional as F
import time

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main

# Load your handwritten Julia CPU kernel
Main.include("conv2d.jl")

# Test parameters
H, W = 128, 128
C_in, C_out = 3, 8
kh, kw = 3, 3
N = 8  # batch size
iters = 5

# Random input
x_np = np.random.rand(H, W, C_in, N).astype(np.float32)
w_np = np.random.rand(kh, kw, C_in, C_out).astype(np.float32)
b_np = np.random.rand(C_out).astype(np.float32)

# Warm-up Julia
Main.conv2d(x_np, w_np, b_np)

# Time Julia conv
start = time.time()
for _ in range(iters):
    y_jl = Main.conv2d(x_np, w_np, b_np)
jl_time = (time.time() - start) / iters

# Prepare PyTorch - transposing the julia (column major) to proper tensor shapes for PyTorch comparison
x_torch = torch.tensor(np.transpose(x_np, (3, 2, 0, 1)))  # NCHW
w_torch = torch.tensor(np.transpose(w_np, (3, 2, 0, 1)))  # out_ch, in_ch, kh, kw
b_torch = torch.tensor(b_np)

# Warm-up PyTorch
# F.conv2d(x_torch, w_torch, bias=b_torch, stride=1, padding=0)
F.relu(F.conv2d(x_torch, w_torch, bias=b_torch, stride=1, padding=0))

# Time PyTorch conv
start = time.time()
with torch.no_grad():
    for _ in range(iters):
        y_torch = F.relu(F.conv2d(x_torch, w_torch, bias=b_torch, stride=1, padding=0))
torch_time = (time.time() - start) / iters

# Transpose PyTorch back to HWCN
y_torch_np = np.transpose(y_torch.detach().numpy(), (2, 3, 1, 0))

# Verify correctness
max_diff = np.max(np.abs(y_jl - y_torch_np))
are_close = np.allclose(y_jl, y_torch_np, atol=1e-5)

# Report
print(f"Julia handwritten conv2d average time: {jl_time*1000:.2f} ms")
print(f"PyTorch conv2d average time: {torch_time*1000:.2f} ms")
print(f"Speedup (PyTorch / Julia): {torch_time / jl_time:.2f}x")
print("Max difference:", max_diff)
print("Are outputs close:", are_close)

