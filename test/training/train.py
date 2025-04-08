import torch
import yaml
import numpy as np
import os
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from models import UNet3DCombined
from operators import (
    ratio_potrhs,
    LaplacianLossInterface,
    DirichletBoundaryLoss,
    InterfaceBoundaryLoss
)

# -----------------------------------------------
# Cargar configuración y arquitectura desde YAMLs
# -----------------------------------------------
parser = argparse.ArgumentParser(description='Training 3D Poisson')
parser.add_argument('-c', '--cfg', type=str, default=None, help='Config filename')
args = parser.parse_args()

with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)

batch_size    = cfg['data_loader']['batch_size']
num_epochs    = cfg['trainer']['epochs']
lapl_weight   = cfg['loss']['args']['lapl_weight']
bound_weight  = cfg['loss']['args']['bound_weight']
inter_weight  = cfg['loss']['args']['interface_weight']
lr            = cfg['loss']['args']['optimizer_lr']

arch_model = cfg['arch']['model']
arch_type  = cfg['arch']['type']
arch_dir   = os.path.join('../', cfg['arch']['arch_dir'])

with open(arch_dir) as yaml_arch:
    arch = yaml.safe_load(yaml_arch)

scales_data = arch.get(arch_type, {}).get('args', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
kernel_size = arch[arch_type]['args']['kernel_sizes']

# -----------------------------------------------
# Definición de dominio y mallas 3D
# -----------------------------------------------
xmin, xmax = cfg['globals']['xmin'], cfg['globals']['xmax']
ymin, ymax = cfg['globals']['ymin'], cfg['globals']['ymax']
zmin, zmax = cfg['globals']['zmin'], cfg['globals']['zmax']
nnx, nny, nnz = cfg['globals']['nnx'], cfg['globals']['nny'], cfg['globals']['nnz']
x, y, z = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny), np.linspace(zmin, zmax, nnz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

interface_center = (
    cfg['globals']['interface_center']['x'],
    cfg['globals']['interface_center']['y'],
    cfg['globals']['interface_center']['z']
)
interface_radius = cfg['globals']['interface_radius']
epsilon_inside   = cfg['globals']['epsilon_inside']
epsilon_outside  = cfg['globals']['epsilon_outside']

Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin
dx, dy, dz = Lx / nnx, Ly / nny, Lz / nnz

# -----------------------------------------------
# Paths
# -----------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
data_dir = os.path.join(parent_dir, cfg['general']['data_dir'])
save_dir = os.path.join(script_dir, 'trained_models')
os.makedirs(save_dir, exist_ok=True)
case_name = cfg['general']['name_case']

# -----------------------------------------------
# Device y normalización
# -----------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando device: {device}")

alpha = 0.1
ratio_max = ratio_potrhs(alpha, Lx, Ly, Lz)

# -----------------------------------------------
# Crear malla 3D y máscaras
# -----------------------------------------------
x = torch.linspace(xmin, xmax, nnx, dtype=torch.double)
y = torch.linspace(ymin, ymax, nny, dtype=torch.double)
z = torch.linspace(zmin, zmax, nnz, dtype=torch.double)
X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

interface_mask = (
    (X - interface_center[0])**2 +
    (Y - interface_center[1])**2 +
    (Z - interface_center[2])**2 <= interface_radius**2
)

interface_boundary = torch.zeros_like(interface_mask, dtype=torch.bool)
for i in range(1, nnx - 1):
    for j in range(1, nny - 1):
        for k in range(1, nnz - 1):
            if any([
                interface_mask[i,j,k] != interface_mask[i+dx,j,k]
                for dx in [-1, 1]
            ]) or any([
                interface_mask[i,j,k] != interface_mask[i,j+dy,k]
                for dy in [-1, 1]
            ]) or any([
                interface_mask[i,j,k] != interface_mask[i,j,k+dz]
                for dz in [-1, 1]
            ]):
                if interface_mask[i,j,k]:
                    interface_boundary[i,j,k] = True

inner_mask = interface_mask & ~interface_boundary
outer_mask = ~interface_mask & ~interface_boundary

interface_mask = interface_mask.to(device)
interface_boundary = interface_boundary.to(device)
inner_mask = inner_mask.to(device)
outer_mask = outer_mask.to(device)

# -----------------------------------------------
# Cargar datos y DataLoader
# -----------------------------------------------
data_npz = np.load(data_dir)
rhs_data = data_npz['rhs']      # shape: (N, nnx, nny, nnz)
q_data = data_npz['q']          # shape: (N, max_charges)
xq_data = data_npz['xq']        # shape: (N, max_charges, 3)

# Convert to tensors
rhs_tensor = torch.from_numpy(rhs_data).double()
q_tensor   = torch.from_numpy(q_data).double()
xq_tensor  = torch.from_numpy(xq_data).double()

# Dataset and DataLoader
dataset = TensorDataset(rhs_tensor, q_tensor, xq_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# -----------------------------------------------
# Modelo y funciones de pérdida
# -----------------------------------------------
model = UNet3DCombined(
    scales=scales,
    kernel_sizes=kernel_size,
    input_res=[nnx, nny, nnz]
).double().to(device)

laplacian_loss = LaplacianLossInterface(cfg, lapl_weight, inner_mask, outer_mask, points)
dirichlet_loss = DirichletBoundaryLoss(bound_weight)
interface_loss = InterfaceBoundaryLoss(cfg, inter_weight, interface_boundary, inner_mask, outer_mask,
    interface_center, interface_radius, points, epsilon_inside, epsilon_outside, dx, dy, dz)

optimizer = optim.Adam(model.parameters(), lr=lr)

laplacian_losses, dirichlet_losses, interface_losses, total_losses = [], [], [], []

print(f"Model used: {cfg['arch']['arch_dir']}, {arch_type}")

# -----------------------------------------------
# Bucle de entrenamiento
# -----------------------------------------------
for epoch in range(num_epochs):
    total_loss = 0.0

    for batch_idx, (batch, q_batch, xq_batch) in enumerate(dataloader):
        batch = batch.unsqueeze(1).to(device)  
        mask_tensor = inner_mask.unsqueeze(0).unsqueeze(0).expand_as(batch).double().to(device)
        data_norm = torch.ones((batch.size(0),1,1,1,1), dtype=torch.double, device=device) / ratio_max

        input_tensor = torch.cat([batch, mask_tensor], dim=1)  

        q_batch = q_batch.to(device)     
        xq_batch = xq_batch.to(device)   

        output = model(input_tensor)  

        laplacian = laplacian_loss(output, q_batch, xq_batch, data_norm)
        dirichlet = dirichlet_loss(output)
        interface = interface_loss(output, q_batch, xq_batch, data_norm=data_norm)

        loss = laplacian + dirichlet + interface
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        laplacian_losses.append(laplacian.item())
        dirichlet_losses.append(dirichlet.item())
        interface_losses.append(interface.item())
        total_losses.append(loss.item())

        if batch_idx % 20 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")

    print(f"Epoch [{epoch+1}/{num_epochs}] - Total Loss: {total_loss / len(dataloader):.6f}")

    model_path = os.path.join(save_dir, f"{case_name}.pth")
    torch.save(model.state_dict(), model_path)

# -----------------------------------------------
# Guardar historial de pérdidas
# -----------------------------------------------
loss_file = os.path.join(save_dir, f"{case_name}_losses.txt")
with open(loss_file, "w") as f:
    f.write("Laplacian Losses:\n")
    f.write(", ".join(map(str, laplacian_losses)) + "\n\n")
    f.write("Dirichlet Losses:\n")
    f.write(", ".join(map(str, dirichlet_losses)) + "\n\n")
    f.write("Interface Losses:\n")
    f.write(", ".join(map(str, interface_losses)) + "\n\n")
    f.write("Total Losses:\n")
    f.write(", ".join(map(str, total_losses)) + "\n")

print(f"Pérdidas guardadas en {loss_file}")