import torch
from models import UNet3D, MSNet3D
import yaml
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from operators3d import ratio_potrhs, LaplacianLoss, DirichletBoundaryLossFunction
import torch.optim as optim
import os
import argparse

# --- Load config ---
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('-c', '--cfg', type=str, default=None, help='Config filename')
args = parser.parse_args()
with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)

# --- Read config parameters ---
arch_model = cfg['arch']['model']
arch_type = cfg['arch']['type']
arch_dir = os.path.join('..', '..', cfg['arch']['arch_dir'])
with open(arch_dir, 'r') as arch_stream:
    arch = yaml.safe_load(arch_stream)
scales_data = arch.get(arch_type, {}).get('args', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
kernel_size = arch[arch_type]['args']['kernel_sizes']

model_type = cfg['arch']['model']
batch_size = cfg['data_loader']['batch_size']
num_epochs = cfg['trainer']['epochs']
lapl_weight = cfg['loss']['args']['lapl_weight']
bound_weight = cfg['loss']['args']['bound_weight']
lr = cfg['loss']['args']['optimizer_lr']

# --- Domain and resolution ---
xmin, xmax = cfg['globals']['xmin'], cfg['globals']['xmax']
ymin, ymax = cfg['globals']['ymin'], cfg['globals']['ymax']
zmin, zmax = cfg['globals']['zmin'], cfg['globals']['zmax']
nnx, nny, nnz = cfg['globals']['nnx'], cfg['globals']['nny'], cfg['globals']['nnz']
Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin

# --- Directories ---
case_name = cfg['general']['name_case']
data_dir = os.path.join('..', cfg['general']['data_dir'])
save_dir = os.getcwd()
save_dir = os.path.join(save_dir, 'trained_models')
os.makedirs(save_dir, exist_ok=True)

# --- Normalize parameter ---
alpha = 0.1
ratio_max = ratio_potrhs(alpha, Lx, Ly, Lz)

# --- Load dataset ---
dataset = np.load(data_dir).astype(np.float32)
data_set = TensorDataset(torch.from_numpy(dataset))
dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

# --- Model ---
if model_type == 'UNet':
    model = UNet3D(scales, kernel_sizes=kernel_size, input_res=nnx)
    print('Using UNet model\n')
elif model_type == 'MSNet':
    model = MSNet3D(scales=scales, kernel_sizes=kernel_size, input_res=nnx)
    print('Using MSNet model\n')
else:
    raise ValueError("Invalid model type")

model = model.float()

# --- Loss functions ---
laplacian_loss = LaplacianLoss(cfg, lapl_weight=lapl_weight)
dirichlet_loss = DirichletBoundaryLossFunction(bound_weight, xmin, xmax, ymin, ymax, zmin, zmax, nnx, nny, nnz)

optimizer = optim.Adam(model.parameters(), lr=lr)

# --- Track losses ---
laplacian_losses, dirichlet_losses, total_losses = [], [], []

# --- Training loop ---
for epoch in range(num_epochs):
    total_loss_epoch = 0
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        data = batch[0]
        data = data[:, np.newaxis, :, :, :].float()

        data_norm = torch.ones((data.size(0), 1, 1, 1, 1)) / ratio_max
        output = model(data)

        lap_loss = laplacian_loss(output, data=data, data_norm=data_norm)
        loss = lap_loss
        laplacian_losses.append(lap_loss.item())

        dir_loss = dirichlet_loss(output)
        loss += dir_loss
        dirichlet_losses.append(dir_loss.item())
        total_losses.append(loss.item())

        loss.backward()
        optimizer.step()
        total_loss_epoch += loss.item()

        if batch_idx % 20 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {total_loss_epoch / len(dataloader)}")

    # Save model 
    model_path = os.path.join(save_dir, case_name + '.pth')
    torch.save(model.state_dict(), model_path)

# --- Save loss values ---
loss_file = os.path.join(save_dir, f"{case_name}_losses.txt")
with open(loss_file, 'w') as f:
    if loss_type == 'laplacian':
        f.write("Laplacian Losses:\n")
        f.write(", ".join(map(str, laplacian_losses)) + "\n\n")
    f.write("Dirichlet Losses:\n")
    f.write(", ".join(map(str, dirichlet_losses)) + "\n")
    f.write("Total Losses:\n")
    f.write(", ".join(map(str, total_losses)) + "\n")

print(f"Losses saved to {loss_file}")
