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

# Set the current Julia kernel to use dense_serial
current_julia_kernel = Main.DenseOps.dense_serial

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

    def forward(self, x):
        x = x.to(gpu)
        print("GPU", torch.cuda.is_available())
        print("Input device:", x.device)
        # Block 1
        x = self.conv1_1(x)
        if self.batch_norm: x = self.bn1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        if self.batch_norm: x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2_1(x)
        if self.batch_norm: x = self.bn2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        if self.batch_norm: x = self.bn2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3_1(x)
        if self.batch_norm: x = self.bn3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        if self.batch_norm: x = self.bn3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        if self.batch_norm: x = self.bn3_3(x)
        x = self.relu3_3(x)
        x = self.pool3(x)

        # Block 4
        x = self.conv4_1(x)
        if self.batch_norm: x = self.bn4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        if self.batch_norm: x = self.bn4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        if self.batch_norm: x = self.bn4_3(x)
        x = self.relu4_3(x)
        x = self.pool4(x)

        # Block 5
        x = self.conv5_1(x)
        if self.batch_norm: x = self.bn5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        if self.batch_norm: x = self.bn5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        if self.batch_norm: x = self.bn5_3(x)
        x = self.relu5_3(x)
        x = self.pool5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

#        x = self.classifier(x)
        x = x.to(cpu)
        print("Input device:", x.device)
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
    #model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


    train_one_epoch(model, train_loader, criterion, optimizer, gpu)
