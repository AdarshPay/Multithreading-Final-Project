from julia.api import Julia
jl = Julia(compiled_modules = False)
from julia import Main
Main.include("dense_serial.jl")

import os
os.environ["JULIA_NUM_THREADS"] = "8"

# Set the current Julia kernel to use dense_serial
current_julia_kernel = Main.DenseOps.dense_serial

import torch
# --- FIX: Import nn for the VGG16 class ---
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

NUM_BATCHES = 64
# --- FIX: Define TOLERANCE ---
TOLERANCE = 1e-4 

# Set device to CPU explicitly for clarity

transform = transforms.Compose([
    transforms.Resize(224), 
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)

train_loader = DataLoader(trainset, batch_size=512, shuffle=True)

class VGG16(nn.Module):
    def __init__(self, num_classes: int = 10, batch_norm: bool = False):
        super().__init__()
        self.batch_norm = batch_norm

        # ... (Convolutional Blocks defined here) ...
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

        # Classifier
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


    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    # def _capture_io_hook(self, module, input, output):
    #     self.captured_input = input[0].detach() 
    #     self.captured_output = output.detach() 
    #     print("Hook captured I/O for the first linear layer.")
    
    def forward(self, x):
        # ... (Forward logic remains the same, assuming all intermediate blocks are defined) ...
        
        # Block 1
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)
        # Block 2
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)
        # Block 3
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        x = self.pool3(x)
        # Block 4
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        x = self.pool4(x)
        # Block 5
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.relu5_3(x)
        x = self.pool5(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        #x = self.classifier(x)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                x = self._julia_linear_forward(layer, x)
            else:
                x = layer(x)
        return x

    def _julia_linear_forward(self, layer, x_tensor):
        x_np = x_tensor.detach().numpy().astype('float32') 
        W_np = layer.weight.data.numpy().astype('float32').T.copy()
        b_np = layer.bias.data.numpy().astype('float32')
        print(W_np.shape)
        print(b_np.shape)
        
        t_serial, out_serial = Main.DenseOps.time_kernel(Main.DenseOps.dense_serial, x_np, W_np, b_np)

        # THREADED
        t_threaded, out_threaded = Main.DenseOps.time_kernel(Main.DenseOps.dense_threaded, x_np, W_np, b_np)

        print(f"Serial runtime:     {t_serial:.6f} seconds")
        print(f"Threaded runtime:   {t_threaded:.6f} seconds")
        print(f"Speedup:            {t_serial / t_threaded:.2f}x")
        return torch.from_numpy(out_threaded)
    




print("Loading a batch of data...")
data_iterator = iter(train_loader)
images, _ = next(data_iterator)



def train_one_epoch(model, dataloader, criterion, optimizer):
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

            # Run the Forward Pass (Julia Kernel)
            outputs = model(images)
            
            print(f"Batch {i+1} complete")

    print("--- Done ---")

if __name__ == '__main__':
    # Quick sanity check
    model = VGG16(num_classes=10)
    # model.to(gpu)              # conv layers on GPU
    # model.classifier.to(cpu)   # classifier layers on CPU
    #model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


    train_one_epoch(model, train_loader, criterion, optimizer)

# pyt_input_tensor = model.captured_input 
# pyt_output_np = model.captured_output.numpy()

# linear_layer = model.first_linear_layer
# print("Running Julia kernel via wrapper...")
# _julia_linear_forward(linear_layer, pyt_input_tensor)

