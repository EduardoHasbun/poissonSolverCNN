import torch
from models import UNet3D, MSNet3D
import yaml
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from operators3d import ratio_potrhs, LaplacianLoss, DirichletBoundaryLoss, InsideLoss
import torch.optim as optim
import os
import argparse

# Import external parameters
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('-c', '--cfg', type=str, default=None, help='Config filename')
args = parser.parse_args()
with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)

# Read config values
scales_data = cfg.get('arch', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
model_type = cfg['arch']['model']
batch_size = cfg['data_loader']['batch_size']
num_epochs = cfg['trainer']['epochs']
lapl_weight = cfg['loss']['args']['lapl_weight']
inside_weight = cfg['loss']['args']['inside_weight']
bound_weight = cfg['loss']['args']['bound_weight']
loss_type = cfg['loss']['type']
lr = cfg['loss']['args']['optimizer_lr']
arch_model = cfg['arch']['model']
arch_type = cfg['arch']['type']
arch_dir = os.path.join('../../', cfg['arch']['arch_dir'])

with open(arch_dir) as yaml_stream1:
    arch = yaml.safe_load(yaml_stream1)
scales_data = arch.get(arch_type, {}).get('args', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
kernel_size = arch[arch_type]['args']['kernel_sizes']

# Grid and domain size
xmin, xmax = cfg['globals']['xmin'], cfg['globals']['xmax']
ymin, ymax = cfg['globals']['ymin'], cfg['globals']['ymax']
zmin, zmax = cfg['globals']['zmin'], cfg['globals']['zmax']
nnx, nny, nnz = cfg['globals']['nnx'], cfg['globals']['nny'], cfg['globals']['nnz']

# Setup directories
case_name = cfg['general']['name_case']
data_dir = os.path.join('..', cfg['general']['data_dir'])  
save_dir = os.path.join(os.getcwd(), 'models')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if loss_type == 'inside':
    target_dir = os.path.join('..', cfg['general']['target_dir'])

# Normalize parameter
alpha = 0.1
ratio_max = ratio_potrhs(alpha, Lx, Ly, Lz)

# Load data
dataset = np.load(data_dir).astype(np.float32)
if loss_type == 'inside':
    target = np.load(target_dir).astype(np.float32)
    data_set = TensorDataset(torch.from_numpy(dataset), torch.from_numpy(target))
    dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
else:
    dataloader = DataLoader(torch.from_numpy(dataset), batch_size=batch_size, shuffle=True)

# Create model
if model_type == 'UNet':
    model = UNet3D(scales, kernel_sizes=kernel_size, input_res=nnx)
    print('Using UNet model \n')
elif model_type == 'MSNet':
    model = MSNet3D(scales=scales, kernel_sizes=kernel_size, input_res=nnx)
    print('Using MSNet model \n')
else:
    raise ValueError("Model type not recognized")

model = model.float()

# Define losses
if loss_type == 'laplacian':
    laplacian_loss = LaplacianLoss(cfg, lapl_weight=lapl_weight)
    print('Using Laplacian Loss \n')
elif loss_type == 'inside':
    inside_loss = InsideLoss(cfg, inside_weight=inside_weight)
    print('Using Inside Loss \n')
dirichlet_loss = DirichletBoundaryLoss(bound_weight)

optimizer = optim.Adam(model.parameters(), lr=lr)

# Track losses
laplacian_losses = []
dirichlet_losses = []
total_losses = []

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        if loss_type == 'inside':
            data, target = batch
            target = target[:, np.newaxis, :, :, :].float()
        else:
            data = batch[:, np.newaxis, :, :, :].float()

        optimizer.zero_grad()
        data = torch.FloatTensor(data)
        data_norm = torch.ones((data.size(0), 1, 1, 1, 1)) / ratio_max
        output = model(data)

        if loss_type == 'laplacian':
            lap_loss = laplacian_loss(output, data=data, data_norm=data_norm)
            loss = lap_loss
            laplacian_losses.append(lap_loss.item())
        elif loss_type == 'inside':
            in_loss = inside_loss(output, target)
            loss = in_loss

        dir_loss = dirichlet_loss(output)
        loss += dir_loss
        dirichlet_losses.append(dir_loss.item())
        total_losses.append(loss.item())

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 20 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {total_loss / len(dataloader)}")

    if epoch % 10 == 0:
        torch.save(model.state_dict(), os.path.join(save_dir, case_name + f'_epoch_{epoch}.pth'))

# Save losses
loss_file_path = os.path.join(save_dir, f"{case_name}_losses.txt")
with open(loss_file_path, "w") as f:
    if loss_type == 'laplacian':
        f.write("Laplacian Losses:\n")
        f.write(", ".join(map(str, laplacian_losses)) + "\n\n")
    f.write("Dirichlet Losses:\n")
    f.write(", ".join(map(str, dirichlet_losses)) + "\n")
    f.write("Total Losses:\n")
    f.write(", ".join(map(str, total_losses)) + "\n")

print(f"Losses saved to {loss_file_path}")
