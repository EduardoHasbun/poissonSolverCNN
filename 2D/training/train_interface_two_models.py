import torch
from unet import UNet
import yaml
from torch.utils.data import DataLoader
import numpy as np
from operators import ratio_potrhs, LaplacianLoss, DirichletBoundaryLoss, InterfaceBoundaryLoss
import torch.optim as optim
import os
import argparse
from scipy import ndimage

#Import external parameteres
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('-c', '--cfg', type=str, default=None, help='Config filename')
args = parser.parse_args()
with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)
scales_data = cfg.get('arch', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
kernel_size = cfg['arch']['kernel_sizes']
batch_size = cfg['data_loader']['batch_size']
num_epochs = cfg['trainer']['epochs']
lapl_weight = cfg['loss']['args']['lapl_weight']
bound_weight = cfg['loss']['args']['bound_weight']
lr = cfg['loss']['args']['optimizer_lr']
scales_data = cfg.get('arch', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
kernel_sizes = cfg['arch']['kernel_sizes']
xmin, xmax, ymin, ymax, nnx, nny = cfg['globals']['xmin'], cfg['globals']['xmax'],\
            cfg['globals']['ymin'], cfg['globals']['ymax'], cfg['globals']['nnx'], cfg['globals']['nny']
interface_center = (cfg['globals']['interface_center']['x'], cfg['globals']['interface_center']['y'])
interface_radius = cfg['globals']['interface_radius']
Lx = xmax-xmin
Ly = ymax-ymin
save_dir = os.getcwd()
data_dir_inside = os.path.join(save_dir, '..', 'dataset', 'generated', 'inside.npy')
data_dir_outside = os.path.join(save_dir, '..', 'dataset', 'generated', 'outside.npy')

# Parameters for data
x, y= np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny)
X, Y = np.meshgrid(x,y)
interface_mask = (X - interface_center[0])**2 + (Y - interface_center[1])**2 <= interface_radius**2
interface_boundary = np.zeros_like(interface_mask, dtype=bool)
for i in range(1, interface_mask.shape[0]):
    for j in range(1, interface_mask.shape[1]):
        if interface_mask[i, j] != interface_mask[i - 1, j]:
            interface_boundary[i, j] = True
        elif interface_mask[i, j] != interface_mask[i, j - 1]:
            interface_boundary[i, j] = True

# Define a structuring element for dilation
struct = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]])
boundary_neighbors = ndimage.binary_dilation(interface_boundary, structure=struct).astype(interface_boundary.dtype)
boundary_neighbors[interface_boundary] = 0

# Load Data
dataset_inside = np.load(data_dir_inside)
dataset_outside = np.load(data_dir_outside)
dataloader = DataLoader((dataset_inside, dataset_outside), batch_size=batch_size, shuffle=True)

# Parameters to Nomalize
alpha = 0.1
ratio_max = ratio_potrhs(alpha, Lx, Ly)


# Create models and losses
model_outside = UNet(scales, kernel=kernel_size)
model_inside = UNet(scales, kernel=kernel_size)
model_outside = model_outside.double()
model_inside = model_inside.double()
laplacian_loss = LaplacianLoss(cfg, lapl_weight=lapl_weight)
dirichlet_loss = DirichletBoundaryLoss(bound_weight)
interface_loss = InterfaceBoundaryLoss(bound_weight, interface_boundary)
parameters = list(model_inside.parameters()) + list(model_outside.parameters())
optimizer = optim.Adam(parameters, lr=lr)

#Train loop
for epoch in range (num_epochs):
    total_loss_inside = 0
    total_loss_outside = 0
    for batch_idx, batch in enumerate(dataloader):
        inside, outside = batch[:, np.newaxis, :, :]
        optimizer.zero_grad()
        insdie, outisde = torch.DoubleTensor(inside), torch.DoubleTensor(outside)
        optimizer.zero_grad()
        data_norm_inside = torch.ones((data_dir_inside.size(0), data_dir_inside.size(1), 1, 1)) / ratio_max
        data_norm_outside = torch.ones((data_dir_outside.size(0), data_dir_outside.size(1), 1, 1)) / ratio_max

        # Getting Outputs
        output_inside = model_inside(inside)
        output_outside = model_outside(outside)

        # Loss Inside
        loss_inside = laplacian_loss(output_inside, data = inside, data_norm = data_norm_inside)
        loss_inside += interface_loss(output_inside, output_outside)
        total_loss_inside += loss_inside

        # Loss Outisde
        loss_outside = laplacian_loss(output_outside, data = outisde, data_norm = data_norm_outside)
        loss_outside += dirichlet_loss(output_outside)
        total_loss_outside += loss_outside

        # Backpropagation
        loss_inside.backward()
        loss_outside.backward()
        optimizer.step()
        if batch_idx % 20 ==0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss Inside: {loss_inside.item()}, Loss Outside: {loss_outside.item()}")
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss Inside: {total_loss_inside / len(dataloader)}, Loss Outside: {total_loss_outside / len(dataloader)}")
    torch.save(model_inside.state_dict(), os.path.join(save_dir, 'model_inside.pth'))
    torch.save(model_outside.state_dict(), os.path.join(save_dir, 'model_outside,pth'))

