"""
Layer-level PyTorch implementation of VGG-16
- Explicit layer attributes (conv1_1, conv1_2, pool1, ...)
- Optional batch normalization (VGG-16-BN) via `batch_norm=True`
- Weight initialization matching common VGG practice
- Returns logits (classifier output). To extract intermediate features, inspect attributes or modify forward.

Usage:
    from vgg16_layer_level import VGG16
    model = VGG16(num_classes=1000, batch_norm=False)
    x = torch.randn(1,3,224,224)
    out = model(x)

This file is intended for learning, debugging, and modification. For production and pretrained weights, refer to torchvision.models.vgg.
"""

from julia.api import Julia
jl = Julia(compiled_modules = False)
from julia import Main
Main.include("dense_serial.jl")
Main.include("conv2d.jl")

# Set the current Julia kernel to use dense_serial
current_julia_kernel = Main.DenseOps.dense_serial
julia_conv2d = Main.conv2d


import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

cpu = torch.device("cpu")
gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU", torch.cuda.is_available())

transform = transforms.Compose([
    transforms.Resize(224),  # VGG16 expects 224x224
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)

train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

class VGG16(nn.Module):
    def __init__(self, num_classes: int = 1000, batch_norm: bool = False):
        super().__init__()
        self.batch_norm = batch_norm

        # Block 1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Optional batchnorm layers (if requested)
        if batch_norm:
            # Create BatchNorm layers matching conv outputs
            self.bn1_1 = nn.BatchNorm2d(64)
            self.bn1_2 = nn.BatchNorm2d(64)
            self.bn2_1 = nn.BatchNorm2d(128)
            self.bn2_2 = nn.BatchNorm2d(128)
            self.bn3_1 = nn.BatchNorm2d(256)
            self.bn3_2 = nn.BatchNorm2d(256)
            self.bn3_3 = nn.BatchNorm2d(256)
            self.bn4_1 = nn.BatchNorm2d(512)
            self.bn4_2 = nn.BatchNorm2d(512)
            self.bn4_3 = nn.BatchNorm2d(512)
            self.bn5_1 = nn.BatchNorm2d(512)
            self.bn5_2 = nn.BatchNorm2d(512)
            self.bn5_3 = nn.BatchNorm2d(512)

        # Classifier (three FC layers as in original paper)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _julia_linear_forward(self, layer, x_tensor):
        # Prepare data in to numpy format
        x_np = x_tensor.detach().cpu().numpy().astype('float32')
        W_np = layer.weight.data.cpu().numpy().astype('float32').T.copy()
        b_np = layer.bias.data.cpu().numpy().astype('float32')

        #------------------------------------------------------------------------
        # Prints for debugging, checking to make sure shapes of tensores align
        print(f"Layer Debug")
        print(f"Input X shape:  {x_np.shape} (Batch x In_Feat)")
        print(f"Weight W shape: {W_np.shape} (In_Feat x Out_Feat)")
        print(f"Bias b shape:   {b_np.shape}")
        
        # Check compatibility
        if x_np.shape[1] != W_np.shape[0]:
            print(f"MISMATCH DETECTED! Input cols ({x_np.shape[1]}) != Weight rows ({W_np.shape[0]})")
        else:
            print(f"Shapes align.")
        #------------------------------------------------------------------------
        # use the selected julia kernel for the forward pass (serial or multithreaded)
        out_np = current_julia_kernel(x_np, W_np, b_np)
        
        return torch.from_numpy(out_np).to(x_tensor.device)

    def _julia_conv2d_forward(self, layer: nn.Conv2d, x_tensor: torch.Tensor):
        """
        Call Julia conv2d(x, w, b) for a Conv2d layer.

        PyTorch: x is (N, C_in, H, W)
        Julia:   x is (H, W, C_in, N)
        PyTorch weights: (C_out, C_in, kh, kw)
        Julia weights:   (kh, kw, C_in, C_out)

        The Julia kernel is 'valid' conv; we handle padding in PyTorch first.
        """
        # Original device (likely GPU)
        orig_device = x_tensor.device

        # Move input to CPU for Julia
        x_cpu = x_tensor.detach().to(cpu)

        # Handle padding from the Conv2d layer using PyTorch (padding=1 for VGG)
        pad_h, pad_w = layer.padding
        if pad_h != 0 or pad_w != 0:
            # F.pad pad order: (left, right, top, bottom)
            x_cpu = F.pad(x_cpu, (pad_w, pad_w, pad_h, pad_h))

        # x_cpu: (N, C_in, H_p, W_p) -> Julia layout: (H_p, W_p, C_in, N)
        x_np = x_cpu.numpy().astype('float32')
        x_np_jl = x_np.transpose(2, 3, 1, 0).copy()  # H, W, C_in, N

        # Weights: PyTorch (C_out, C_in, kh, kw) -> Julia (kh, kw, C_in, C_out)
        W_torch = layer.weight.data.detach().cpu().numpy().astype('float32')
        W_np_jl = W_torch.transpose(2, 3, 1, 0).copy()  # kh, kw, C_in, C_out

        # Bias: (C_out,)
        b_np = layer.bias.data.detach().cpu().numpy().astype('float32')

        # Debug (optional)
        # print(f"[Julia Conv] x {x_np_jl.shape}, w {W_np_jl.shape}, b {b_np.shape}")

        # Call Julia serial conv2d
        y_np_jl = julia_conv2d(x_np_jl, W_np_jl, b_np)
        # Julia y: (H_out, W_out, C_out, N) -> PyTorch: (N, C_out, H_out, W_out)
        y_np = y_np_jl.transpose(3, 2, 0, 1).copy()

        y_tensor = torch.from_numpy(y_np).to(orig_device)
        return y_tensor


    def forward(self, x):
        x = x.to(gpu)
        print("GPU", torch.cuda.is_available())
        print("Input device for conv:", x.device)
        # Block 1
        # x = self.conv1_1(x)
        x = self._julia_conv2d_forward(self.conv1_1, x)
        if self.batch_norm: x = self.bn1_1(x)
        x = self.relu1_1(x)
        # x = self.conv1_2(x)
        x = self._julia_conv2d_forward(self.conv1_2, x)
        if self.batch_norm: x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)
        print("block 1 done")

        # Block 2
        # x = self.conv2_1(x)
        x = self._julia_conv2d_forward(self.conv2_1, x)
        if self.batch_norm: x = self.bn2_1(x)
        x = self.relu2_1(x)
        # x = self.conv2_2(x)
        x = self._julia_conv2d_forward(self.conv2_2, x)
        if self.batch_norm: x = self.bn2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)
        print("block 2 done")

        # Block 3
        # x = self.conv3_1(x)
        x = self._julia_conv2d_forward(self.conv3_1, x)
        if self.batch_norm: x = self.bn3_1(x)
        x = self.relu3_1(x)
        # x = self.conv3_2(x)
        x = self._julia_conv2d_forward(self.conv3_2, x)
        if self.batch_norm: x = self.bn3_2(x)
        x = self.relu3_2(x)
        # x = self.conv3_3(x)
        x = self._julia_conv2d_forward(self.conv3_3, x)
        if self.batch_norm: x = self.bn3_3(x)
        x = self.relu3_3(x)
        x = self.pool3(x)
        print("block 3 done")

        # Block 4
        # x = self.conv4_1(x)
        x = self._julia_conv2d_forward(self.conv4_1, x)
        if self.batch_norm: x = self.bn4_1(x)
        x = self.relu4_1(x)
        # x = self.conv4_2(x)
        x = self._julia_conv2d_forward(self.conv4_2, x)
        if self.batch_norm: x = self.bn4_2(x)
        x = self.relu4_2(x)
        # x = self.conv4_3(x)
        x = self._julia_conv2d_forward(self.conv4_3, x)
        if self.batch_norm: x = self.bn4_3(x)
        x = self.relu4_3(x)
        x = self.pool4(x)
        print("block 4 done")

        # Block 5
        # x = self.conv5_1(x)
        x = self._julia_conv2d_forward(self.conv5_1, x)
        if self.batch_norm: x = self.bn5_1(x)
        x = self.relu5_1(x)
        # x = self.conv5_2(x)
        x = self._julia_conv2d_forward(self.conv5_2, x)
        if self.batch_norm: x = self.bn5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        if self.batch_norm: x = self.bn5_3(x)
        x = self.relu5_3(x)
        x = self.pool5(x)
        print("block 5 done")

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

#        x = self.classifier(x)
        x = x.to(cpu)
        print("Input device for classifier:", x.device)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                x = self._julia_linear_forward(layer, x)
            else:
                x = layer(x)
        return x

    def _initialize_weights(self):
        # Follows common VGG initialization (kaiming normal for conv, normal for fc)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# ---- One-epoch training loop example ----
# Usage:
#   model = VGG16(num_classes=10)
#   train_loader = ...   # your DataLoader
#   optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#   criterion = nn.CrossEntropyLoss()
#   train_one_epoch(model, train_loader, criterion, optimizer, device='cuda')

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    # Switch to evaluate mode (inference only)
    #model.to(device)
    model.eval()
    
    # Limit number of batches for the test
    limit_batches = 20

    print(f"--- Running Inference (No Internal Timer) ---")

    # Disable gradients to avoid errors with Julia
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i >= limit_batches:
                break
            
            images, labels = images.to(device), labels.to(device)

            # Run the Forward Pass (Julia Kernel)
            outputs = model(images)
            
            print(f"Batch {i+1} complete")

    print("--- Done ---")


if __name__ == '__main__':
    # Quick sanity check
    model = VGG16(num_classes=10)
    model.to(gpu)              # conv layers on GPU
    model.classifier.to(cpu)   # classifier layers on CPU
    #model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


    train_one_epoch(model, train_loader, criterion, optimizer, gpu)
