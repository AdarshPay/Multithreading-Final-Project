from julia import Main
import torch
import numpy as np

# Example input
A = torch.randn(2, 512).cpu().numpy().astype(np.float32)
W = torch.randn(512, 1024).cpu().numpy().astype(np.float32)
b = torch.randn(1024).cpu().numpy().astype(np.float32)

# Load the Julia kernel
Main.include("dense_serial.jl")
Main.A = A
Main.W = W
Main.b = b

# Call the kernel
output = Main.dense_serial(Main.A, Main.W, Main.b)

# Convert back to PyTorch tensor
output_tensor = torch.from_numpy(np.array(output))
print(output_tensor.shape)  # Should be (2, 1024)

