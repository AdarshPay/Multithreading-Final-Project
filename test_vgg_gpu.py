# test_vgg_gpu.py
import torch
import torchvision.models as models

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load baseline VGG-16
vgg16 = models.vgg16(pretrained=False)  # don't download weights for now
vgg16.to(device)

# Create a dummy input (batch size 2, 3 channels, 224x224)
dummy_input = torch.randn(2, 3, 224, 224, device=device)

# Forward pass
with torch.no_grad():
    output = vgg16(dummy_input)

print(f"Output shape: {output.shape}")  # Should be [2, 1000] for VGG-16

