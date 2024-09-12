import torch
from unet3d_interface import UNet3D
import yaml
from torch.utils.data import DataLoader
import numpy as np
from operators3d import ratio_potrhs, LaplacianLoss, DirichletBoundaryLoss, InterfaceBoundaryLoss
import torch.optim as optim
import os
import argparse

# Import external parameteres
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
interface_weight = cfg['loss']['args']['interface_weight']
lr = cfg['loss']['args']['optimizer_lr']
scales_data = cfg.get('arch', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
kernel_sizes = cfg['arch']['kernel_sizes']
xmin, xmax, ymin, ymax, zmin, zmax, nnx, nny, nnz = cfg['globals']['xmin'], cfg['globals']['xmax'], cfg['globals']['ymin'], cfg['globals']['ymax'],\
            cfg['globals']['zmin'], cfg['globals']['zmax'], cfg['globals']['nnx'], cfg['globals']['nny'], cfg['globals']['nnz']
interface_center = (cfg['globals']['interface_center']['x'], cfg['globals']['interface_center']['y'], cfg['globals']['interface_center']['z'])
interface_radius = cfg['globals']['interface_radius']
epsilon_inside, epsilon_outside = cfg['globals']['epsilon_inside'], cfg['globals']['epsilon_outside']
Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin
dx, dy, dz = Lx / nnx, Ly / nny, Lz / nnz
case_name = cfg['general']['name_case']
save_dir = os.getcwd()
data_dir = os.path.join(save_dir, '..', 'dataset', 'generated', 'domain.npy')
save_dir = os.path.join(save_dir, 'models')
if not os.path.exists(save_dir):
        os.makedirs(save_dir)


# Parameters to Nomalize
alpha = 0.1
ratio_max = ratio_potrhs(alpha, Lx, Ly, Lz)

# Parameters for data 
x, y, z= torch.linspace(xmin, xmax, nnx), torch.linspace(ymin, ymax, nny), torch.linspace(zmin, zmax, nnz)
X, Y, Z = torch.meshgrid(x, y, z)
interface_mask = (X - interface_center[0])**2 + (Y - interface_center[1])**2 + (Z -  interface_center[0])**2<= interface_radius**2
interface_boundary = torch.zeros_like(interface_mask, dtype=bool)
for i in range(1, interface_mask.shape[0]):
    for j in range(1, interface_mask.shape[1]):
        for k in range(1, interface_mask.shape[2]):
            if interface_mask[i, j, k] != interface_mask[i - 1, j, k]:
                interface_boundary[i, j, k] = True
            elif interface_mask[i, j, k] != interface_mask[i, j - 1, k]:
                interface_boundary[i, j, k] = True
            elif interface_mask[i, j, k] != interface_mask[i, j, k - 1]:
                interface_boundary[i, j, k] = True

inner_mask = interface_mask
outer_mask = ~interface_mask | interface_boundary

# Load Data
dataset = np.load(data_dir).astype(np.float32) / ratio_max
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Create model and losses
model = UNet3D(scales, kernel_sizes=kernel_size, input_res = nnx, inner_mask = inner_mask, outer_mask = outer_mask)
model = model.float()
laplacian_loss = LaplacianLoss(cfg, lapl_weight=lapl_weight, e_in = epsilon_inside, e_out = epsilon_outside)
dirichlet_loss = DirichletBoundaryLoss(bound_weight)
interface_loss = InterfaceBoundaryLoss(interface_weight, interface_boundary, interface_center, interface_radius, epsilon_inside, epsilon_outside, dx, dy, dz)
optimizer = optim.Adam(model.parameters(), lr = lr)

# Train loop
for epoch in range (num_epochs):
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        data = batch[:, np.newaxis, :, :]
        optimizer.zero_grad()
        data = torch.FloatTensor(data) 
        data_norm = torch.ones((data.size(0), data.size(1), 1, 1)) / ratio_max

        # Getting Outputs
        subdomain_in, subdomain_out = model(data)

        # Loss
        loss = laplacian_loss(subdomain_in, data = data, data_norm = data_norm, mask = inner_mask)
        loss += laplacian_loss(subdomain_out, data = data, data_norm = data_norm, mask = outer_mask)
        loss += dirichlet_loss(subdomain_out)
        loss += interface_loss(subdomain_in, subdomain_out)

        # Backpropagation
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 20 ==0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {total_loss / len(dataloader)}")
    torch.save(model.state_dict(), os.path.join(save_dir, case_name, '.pth'))

