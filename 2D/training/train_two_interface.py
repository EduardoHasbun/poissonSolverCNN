import torch
from unet_interface import UNet
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
epsilon_inside, epsilon_outside = cfg['globals']['epsilon_inside'], cfg['globals']['epsilon_outside']
Lx, Ly = xmax-xmin, ymax-ymin
dx, dy = Lx / nnx, Ly / nny
save_dir = os.getcwd()
data_dir = os.path.join(save_dir, '..', 'dataset', 'generated', 'domain.npy')
save_dir = os.path.join(save_dir, 'models')


# Parameters to Nomalize
alpha = 0.1
ratio_max = ratio_potrhs(alpha, Lx, Ly)


# Parameters for data
x, y= torch.linspace(xmin, xmax, nnx), torch.linspace(ymin, ymax, nny)
X, Y = torch.meshgrid(x,y)
interface_mask = (X - interface_center[0])**2 + (Y - interface_center[1])**2 <= interface_radius**2
interface_boundary = torch.zeros_like(interface_mask, dtype=bool)
for i in range(1, interface_mask.shape[0]):
    for j in range(1, interface_mask.shape[1]):
        if interface_mask[i, j] != interface_mask[i - 1, j]:
            interface_boundary[i, j] = True
        elif interface_mask[i, j] != interface_mask[i, j - 1]:
            interface_boundary[i, j] = True

# Load Data
dataset = np.load(data_dir) / ratio_max
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Create models and losses
model = UNet(scales, kernel_sizes=kernel_size, input_res=nnx, mask = interface_mask)
model = model.double()
laplacian_loss = LaplacianLoss(cfg, lapl_weight=lapl_weight)
dirichlet_loss = DirichletBoundaryLoss(bound_weight)
interface_loss = InterfaceBoundaryLoss(bound_weight, interface_boundary, interface_mask, interface_center, interface_radius, epsilon_inside, epsilon_outside, dx, dy)
optimizer = optim.Adam(model.parameters(), lr=lr)

#Train loop
for epoch in range (num_epochs):
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        data = batch[:, np.newaxis, :, :]
        optimizer.zero_grad()
        insside = torch.DoubleTensor(data)
        data_norm = torch.ones((data.size(0), data.size(1), 1, 1)) / ratio_max

        
        # Getting Outputs
        subdomain_in, subdomain_out = model(data)

        # Loss
        loss = laplacian_loss(subdomain_in, data = data, data_norm = data_norm)
        loss += laplacian_loss(subdomain_out, data = data, data_norm = data_norm)
        loss += dirichlet_loss(subdomain_out)
        loss += interface_loss(subdomain_in, subdomain_out)

        # Backpropagation
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 20 ==0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {total_loss / len(dataloader)}")
    torch.save(model.state_dict(), os.path.join(save_dir, 'interface_2.pth'))