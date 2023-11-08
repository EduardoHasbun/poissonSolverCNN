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
import torch.nn.functional as F

os.environ['OPENBLAS_NUM_THREADS'] = '1'

args = argparse.ArgumentParser(description='Training')
args.add_argument('-c', '--cfg', type=str, default=None,
                help='Config filename')
args.add_argument('--case', type=str, default=None, help='Case name')
args = args.parse_args()
with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)


#Define Laplacian
def laplacian(field, dx, dy, b=0): 

    # Create laplacian tensor with shape (batch_size, 1, h, w)
    laplacian = torch.zeros_like(field).type(field.type())

    # Check sizes
    assert field.dim() == 4 and laplacian.dim() == 4, 'Dimension mismatch'

    assert field.is_contiguous() and laplacian.is_contiguous(), 'Input is not contiguous'

    laplacian[:, 0, 1:-1, 1:-1] = \
        (1 - b) * ((field[:, 0, 2:, 1:-1] + field[:, 0, :-2, 1:-1] - 2 * field[:, 0, 1:-1, 1:-1]) / dy**2 +
        (field[:, 0, 1:-1, 2:] + field[:, 0, 1:-1, :-2] - 2 * field[:, 0, 1:-1, 1:-1]) / dx**2) + \
        b * (field[:, 0, 2:, 2:] + field[:, 0, 2:, :-2] + field[:, 0, :-2, :-2] + field[:, 0, :-2, 2:] - 4 * field[:, 0, 1:-1, 1:-1]) \
        / (2 * dx**2)

    laplacian[:, 0, 0, 1:-1] = \
            (2 * field[:, 0, 0, 1:-1] - 5 * field[:, 0, 1, 1:-1] + 4 * field[:, 0, 2, 1:-1] - field[:, 0, 3, 1:-1]) / dy**2 + \
            (field[:, 0, 0, 2:] + field[:, 0, 0, :-2] - 2 * field[:, 0, 0, 1:-1]) / dx**2
    laplacian[:, 0, -1, 1:-1] = \
        (2 * field[:, 0, -1, 1:-1] - 5 * field[:, 0, -2, 1:-1] + 4 * field[:, 0, -3, 1:-1] - field[:, 0, -4, 1:-1]) / dy**2 + \
        (field[:, 0, -1, 2:] + field[:, 0, -1, :-2] - 2 * field[:, 0, -1, 1:-1]) / dx**2
    laplacian[:, 0, 1:-1, 0] = \
        (field[:, 0, 2:, 0] + field[:, 0, :-2, 0] - 2 * field[:, 0, 1:-1, 0]) / dy**2 + \
        (2 * field[:, 0, 1:-1, 0] - 5 * field[:, 0, 1:-1, 1] + 4 * field[:, 0, 1:-1, 2] - field[:, 0, 1:-1, 3]) / dx**2
    laplacian[:, 0, 1:-1, -1] = \
        (field[:, 0, 2:, -1] + field[:, 0, :-2, -1] - 2 * field[:, 0, 1:-1, -1]) / dy**2 + \
        (2 * field[:, 0, 1:-1, -1] - 5 * field[:, 0, 1:-1, -2] + 4 * field[:, 0, 1:-1, -3] - field[:, 0, 1:-1, -4]) / dx**2

    laplacian[:, 0, 0, 0] = \
            (2 * field[:, 0, 0, 0] - 5 * field[:, 0, 1, 0] + 4 * field[:, 0, 2, 0] - field[:, 0, 3, 0]) / dy**2 + \
            (2 * field[:, 0, 0, 0] - 5 * field[:, 0, 0, 1] + 4 * field[:, 0, 0, 2] - field[:, 0, 0, 3]) / dx**2
    laplacian[:, 0, 0, -1] = \
            (2 * field[:, 0, 0, -1] - 5 * field[:, 0, 1, -1] + 4 * field[:, 0, 2, -1] - field[:, 0, 3, -1]) / dy**2 + \
            (2 * field[:, 0, 0, -1] - 5 * field[:, 0, 0, -2] + 4 * field[:, 0, 0, -3] - field[:, 0, 0, -4]) / dx**2
    
    laplacian[:, 0, -1, 0] = \
        (2 * field[:, 0, -1, 0] - 5 * field[:, 0, -2, 0] + 4 * field[:, 0, -3, 0] - field[:, 0, -4, 0]) / dy**2 + \
        (2 * field[:, 0, -1, 0] - 5 * field[:, 0, -1, 1] + 4 * field[:, 0, -1, 2] - field[:, 0, -1, 3]) / dx**2
    laplacian[:, 0, -1, -1] = \
        (2 * field[:, 0, -1, -1] - 5 * field[:, 0, -2, -1] + 4 * field[:, 0, -3, -1] - field[:, 0, -4, -1]) / dy**2 + \
        (2 * field[:, 0, 0, -1] - 5 * field[:, 0, 0, -2] + 4 * field[:, 0, 0, -3] - field[:, 0, 0, -4]) / dx**2

    return laplacian


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
            nn.Conv2d(64, out_channels, kernel_size=1)  
        )

    def forward(self, x):
        x1 = self.encoder(x)
        return x1    

#Define Losses Functions
class laplacianLoss(nn.Module):
    def __init__(self, weigth, b=0):
        super().__init__()
        self.weight = weigth
        xmin, xmax, ymin, ymax, nnx, nny = cfg['globals']['xmin'], cfg['globals']['xmax'],\
            cfg['globals']['ymin'], cfg['globals']['ymax'], cfg['globals']['nnx'], cfg['globals']['nny']
        self.Lx = xmax-xmin
        self.Ly = ymax-ymin
        self.dx = self.Lx/nnx
        self.dy = self.Ly/nny
        self.b = b

    def forward(self, output, data=None, target_norm=1., data_norm=1.):
        lapl = laplacian(output * target_norm / data_norm, self.dx, self.dy)
        return self.Lx**2 * self.Ly**2 * F.mse_loss(lapl[:, 0, 1:-1, 1:-1], - data[:, 0, 1:-1, 1:-1]) * self.weight
    
class DirichletLoss(nn.Module):
    def __init__(self, bound_weight):
        super().__init__()
        self.weight = bound_weight
        self.base_weight = self.weight

    def forward(self, output):
        bnd_loss = F.mse_loss(output[:, 0, -1, :], torch.zeros_like(output[:, 0, -1, :]))
        bnd_loss += F.mse_loss(output[:, 0, :, 0], torch.zeros_like(output[:, 0, :, 0]))
        bnd_loss += F.mse_loss(output[:, 0, :, -1], torch.zeros_like(output[:, 0, :, -1]))
        bnd_loss += F.mse_loss(output[:, 0, 0, :], torch.zeros_like(output[:, 0, 0, :]))
        return bnd_loss * self.weight

# Import dataset
data = np.load(cfg['data_loader']['data_dir'])
input_data = torch.from_numpy(data)
num_samples, nnx, nny = input_data.shape
in_channels = 1
out_channels = 1
input_data = input_data.reshape(num_samples, in_channels, nnx, nny)
lapl_weight = cfg['loss']['args']['lapl_weight']
bound_weight = cfg['loss']['args']['bound_weight']
batch_size = cfg['data_loader']['batch_size']

# Create DataLoader 
dataset = TensorDataset(input_data)
dataloader = DataLoader(dataset, batch_size, shuffle=True)

# Create U-Net model
model = UNet(in_channels, out_channels)

# Define loss function and optimizer
laplacian_loss = laplacianLoss(lapl_weight)
dirichlet_loss = DirichletLoss(bound_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = cfg['trainer']['epochs']
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader: 
        inputs = batch[0].float() 
        optimizer.zero_grad()
        outputs = model(inputs)
        lapl_loss = laplacian_loss(outputs, inputs)
        dir_loss = dirichlet_loss(outputs)
        loss = lapl_loss + dir_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {total_loss / len(dataloader)}")


# Save the trained model
torch.save(model.state_dict(), 'unet_model.pth')
