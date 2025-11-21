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

from julia import Main
Main.include("conv2d_gpu.jl")
julia_conv = Main.my_conv  # the Julia function

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU", torch.cude.is_available())


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

    def forward(self, x):
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
        x = self.classifier(x)
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

def train_one_epoch(model, dataloader, criterion, optimizer, device='cpu'):
    model.to(device)
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # statistics
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    print(f"Train Loss: {avg_loss:.4f} | Accuracy: {accuracy*100:.2f}%")


if __name__ == '__main__':
    # Quick sanity check
    model = VGG16(num_classes=10)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    train_one_epoch(model, train_loader, criterion, optimizer, device='cuda')
