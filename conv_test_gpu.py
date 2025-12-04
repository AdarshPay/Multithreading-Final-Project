import numpy as np
import torch
import torch.nn.functional as F
import time

# Julia imports MUST happen after torch if possible
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main

# Load the corrected Julia kernel file
Main.include("conv2d_gpu.jl")

# --- 1. Test Parameters and CPU Data ---
H, W = 128, 128
C_in, C_out = 3, 8
kh, kw = 3, 3
N = 8  # batch size
iters = 5

# NumPy data (float32)
x_np = np.random.rand(H, W, C_in, N).astype(np.float32) # HWCN
w_np = np.random.rand(kh, kw, C_in, C_out).astype(np.float32) # KHKW_CINCOUT
b_np = np.random.rand(C_out).astype(np.float32)

# --- 2. PyTorch GPU Prep & Tensor Passing (Julia Style) ---
device = torch.device("cuda")

# 1. Julia Input (x_jl_style_d): Create HWCN tensor directly from NumPy
x_jl_style_d = torch.from_numpy(x_np).float().to(device) 

# 2. Julia Weight (w_jl_style_d): Create KHKW_CINCOUT tensor directly from NumPy
w_jl_style_d = torch.from_numpy(w_np).float().to(device)

# 3. Bias (b_jl_style_d): Create 1D tensor
b_jl_style_d = torch.from_numpy(b_np).float().to(device)

# --- 3. Julia Kernel Execution ---

# Warm-up Julia
y_jl_d, _ = Main.conv2d_gpu(x_jl_style_d, w_jl_style_d, b_jl_style_d) 
torch.cuda.synchronize()

# Time Julia conv
total_jl_time = 0
for _ in range(iters):
    y_jl_d, jl_time = Main.conv2d_gpu(x_jl_style_d, w_jl_style_d, b_jl_style_d)
    total_jl_time += jl_time
torch.cuda.synchronize()
avg_jl_time = total_jl_time / iters

# --- 4. PyTorch Comparison Setup (NCHW Style) ---

# Create NCHW tensors for PyTorch's native F.conv2d operation
x_torch_NCHW = torch.from_numpy(np.transpose(x_np, (3, 2, 0, 1))).float().to(device)
w_torch_NCHW = torch.from_numpy(np.transpose(w_np, (3, 2, 0, 1))).float().to(device) 
b_torch_GPU = b_jl_style_d # Reuse the GPU bias tensor

# Warm-up PyTorch
torch.cuda.synchronize()
F.relu(F.conv2d(x_torch_NCHW, w_torch_NCHW, bias=b_torch_GPU, stride=1, padding=0))
torch.cuda.synchronize()

# Time PyTorch conv
start = time.time()
with torch.no_grad():
    for _ in range(iters):
        y_torch_NCHW = F.relu(F.conv2d(x_torch_NCHW, w_torch_NCHW, bias=b_torch_GPU, stride=1, padding=0))
torch.cuda.synchronize()
torch_time = (time.time() - start) / iters

# --- 5. Verification ---

# Transpose Julia output tensor (HWCN) back to CPU NumPy array
y_jl_np = y_jl_d.detach().cpu().numpy()

# Transpose PyTorch output (NCHW) back to HWCN for NumPy comparison
y_torch_cpu = y_torch_NCHW.detach().cpu().numpy()
y_torch_np = np.transpose(y_torch_cpu, (2, 3, 1, 0))

# Verify correctness
max_diff = np.max(np.abs(y_jl_np - y_torch_np))
are_close = np.allclose(y_jl_np, y_torch_np, atol=1e-5)

# --- 6. Report ---
print(f"Julia conv2d total time: {total_jl_time:.2f} ms")
print(f"Julia avg time: {avg_jl_time :.2f} ms")
print(f"PyTorch conv2d average time: {torch_time*1000:.2f} ms")
print(f"Speedup (PyTorch / Julia): {(torch_time*1000) / avg_jl_time:.2f}x")
print("Max difference:", max_diff)
print("Are outputs close:", are_close)