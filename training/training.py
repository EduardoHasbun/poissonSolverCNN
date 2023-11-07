#############################################################################################################
#                                                                                                           #
#                           RUN:    python training.py -c train.yml                                         #
#                                                                                                           #
#############################################################################################################


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import yaml
import argparse


os.environ['OPENBLAS_NUM_THREADS'] = '1'


args = argparse.ArgumentParser(description='Training')
args.add_argument('-c', '--cfg', type=str, default=None,
                help='Config filename')
args.add_argument('--case', type=str, default=None, help='Case name')
args = args.parse_args()
with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)

nnx = cfg['globals']['nnx']
nny = cfg['globals']['nny']
epoch = cfg['trainer']['epochs']


# Define the U-Net architecture
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder (contracting path)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder (expansive path)
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3

# Generate example data
num_samples = 100
in_channels = 3
out_channels = 1
input_data = torch.randn(num_samples, in_channels, nnx, nny)
target_data = torch.randint(0, 2, (num_samples, out_channels, nnx, nny), dtype=torch.float32)

# Create DataLoader for your data
dataset = TensorDataset(input_data, target_data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Create U-Net model
model = UNet(in_channels, out_channels)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        targets = nn.functional.interpolate(targets, size=outputs.shape[-2:], mode='bilinear', align_corners=False)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {total_loss / len(dataloader)}")

# Save the trained model
torch.save(model.state_dict(), 'unet_model.pth')
