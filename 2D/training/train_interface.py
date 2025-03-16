import torch
from models import UNetInterface
import yaml
from torch.utils.data import DataLoader
import numpy as np
from operators import ratio_potrhs, LaplacianLossInterface, DirichletBoundaryLoss, InterfaceBoundaryLoss
import torch.optim as optim
import os
import argparse
import matplotlib.pyplot as plt

# Import external parameteres
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('-c', '--cfg', type=str, default=None, help='Config filename')
args = parser.parse_args()
with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)
batch_size = cfg['data_loader']['batch_size']
num_epochs = cfg['trainer']['epochs']
lapl_weight = cfg['loss']['args']['lapl_weight']
bound_weight = cfg['loss']['args']['bound_weight']
interface_weight = cfg['loss']['args']['interface_weight']
lr = cfg['loss']['args']['optimizer_lr']
arch_model = cfg['arch']['model']
arch_type = cfg['arch']['type']
arch_dir = os.path.join('../../Archs/', cfg['arch']['arch_dir'])
with open(arch_dir) as yaml_stream1:
    arch = yaml.safe_load(yaml_stream1)
scales_data = arch.get(arch_type, {}).get('args', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
kernel_size = arch[arch_type]['args']['kernel_sizes']
xmin, xmax, ymin, ymax, nnx, nny = cfg['globals']['xmin'], cfg['globals']['xmax'],\
            cfg['globals']['ymin'], cfg['globals']['ymax'], cfg['globals']['nnx'], cfg['globals']['nny']
interface_center = (cfg['globals']['interface_center']['x'], cfg['globals']['interface_center']['y'])
interface_radius = cfg['globals']['interface_radius']
epsilon_inside, epsilon_outside = cfg['globals']['epsilon_inside'], cfg['globals']['epsilon_outside']
Lx, Ly = xmax-xmin, ymax-ymin
dx, dy = Lx / nnx, Ly / nny
data_dir = cfg['general']['data_dir']
save_dir = os.getcwd()
data_dir = os.path.join(save_dir, '..', data_dir)
save_dir = os.path.join(save_dir, 'trained_models')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
case_name = cfg['general']['name_case']


# Parameters to Nomalize
alpha = 0.1
ratio_max = ratio_potrhs(alpha, Lx, Ly)


# Parameters for data
x, y= torch.linspace(xmin, xmax, nnx), torch.linspace(ymin, ymax, nny)
X, Y = torch.meshgrid(x, y)
interface_mask = (X - interface_center[0])**2 + (Y - interface_center[1])**2 <= interface_radius**2
interface_boundary = torch.zeros_like(interface_mask, dtype=bool)
for i in range(1, interface_mask.shape[0] - 1):
    for j in range(1, interface_mask.shape[1] - 1):
        # Check for boundary change and mark only the outside node
        if interface_mask[i, j] != interface_mask[i - 1, j] or interface_mask[i, j] != interface_mask[i + 1, j]:
            if interface_mask[i, j]:  # If current node is outside the interface
                interface_boundary[i, j] = True
        elif interface_mask[i, j] != interface_mask[i, j - 1] or interface_mask[i, j] != interface_mask[i, j + 1]:
            if interface_mask[i, j]:  # If current node is outside the interface
                interface_boundary[i, j] = True

inner_mask = interface_mask
outer_mask = ~interface_mask | interface_boundary

# Load Data
dataset = np.load(data_dir) * ratio_max
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create models and losses
model = UNetInterface(scales, kernel_sizes=kernel_size, input_res=nnx, inner_mask = inner_mask, outer_mask = outer_mask)
model = model.double()
laplacian_loss = LaplacianLossInterface(cfg, lapl_weight=lapl_weight)
dirichlet_loss = DirichletBoundaryLoss(bound_weight)
interface_loss = InterfaceBoundaryLoss(interface_weight, interface_boundary, interface_center, interface_radius,\
                                        epsilon_inside, epsilon_outside, dx, dy)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Initialize lists to store losses
laplacian_losses = []  # To store Laplacian losses
dirichlet_losses = []  # To store Dirichlet losses
interface_losses = []  # To store Interface losses
total_losses = [] # To store total losses   

print(f"Model used: {cfg['arch']['arch_dir']}, {arch_type}")
# Train loop
for epoch in range (num_epochs):
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        data = batch.unsqueeze(1)
        optimizer.zero_grad()
        data_norm = torch.ones((data.size(0), data.size(1), 1, 1)) / ratio_max

        # Getting Outputs
        subdomain_in, subdomain_out = model(data)

        # Loss
        laplacian_loss_inside = laplacian_loss(subdomain_in, data = data, data_norm = data_norm, mask = inner_mask)
        laplacian_loss_outside = laplacian_loss(subdomain_out, data = data, data_norm = data_norm, mask = outer_mask)
        dirichlet_loss = dirichlet_loss(subdomain_out)
        interface_loss = interface_loss(subdomain_in, subdomain_out, data_norm = data_norm)
        loss = laplacian_loss_inside + laplacian_loss_outside + dirichlet_loss + interface_loss

        # Backpropagation
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 20 ==0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
    # Save epoch losses
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Total Loss: {total_loss / len(dataloader)}")
    if epoch % 20 == 0:
        torch.save(model.state_dict(), os.path.join(save_dir, case_name + f'_epoch_{epoch}' + '.pth'))

# Save losses to a .txt file
loss_file_path = os.path.join(save_dir, f"{case_name}_losses.txt")
with open(loss_file_path, "w") as f:
    f.write("Laplacian Losses:\n")
    f.write(", ".join(map(str, laplacian_losses)) + "\n\n")
    
    f.write("Dirichlet Losses:\n")
    f.write(", ".join(map(str, dirichlet_losses)) + "\n")

print(f"Losses saved to {loss_file_path}")