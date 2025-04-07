import torch
import yaml
import numpy as np
import os
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pickle

class PoissonChargeDataset(Dataset):
    def __init__(self, rhs_data, q_list, xq_list):
        self.rhs_data = torch.from_numpy(rhs_data).double()
        self.q_list = q_list
        self.xq_list = xq_list

    def __len__(self):
        return len(self.rhs_data)

    def __getitem__(self, idx):
        rhs = self.rhs_data[idx]
        q = torch.from_numpy(self.q_list[idx]).double()
        xq = torch.from_numpy(self.xq_list[idx]).double()
        return rhs, q, xq


def poisson_collate_fn(batch):
    """
    Permite batch_size > 1 para datos con longitud variable (q, xq)
    """
    rhs_batch = torch.stack([item[0] for item in batch])  # (B, nx, ny, nz)
    q_batch = [item[1] for item in batch]  # lista de tensores (q_i)
    xq_batch = [item[2] for item in batch]  # lista de tensores (xq_i)
    return rhs_batch, q_batch, xq_batch


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
with open(data_dir, 'rb') as f:
    data_dict = pickle.load(f)
rhs_data = data_dict['rhs']
q_list = data_dict['q']
xq_list = data_dict['xq']
dataset = PoissonChargeDataset(rhs_data, q_list, xq_list)
dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=poisson_collate_fn)



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
interface_loss = InterfaceBoundaryLoss(inter_weight, interface_boundary, inner_mask, outer_mask,
    interface_center, interface_radius, points, epsilon_inside, epsilon_outside, dx, dy, dz)

optimizer = optim.Adam(model.parameters(), lr=lr)

laplacian_losses, dirichlet_losses, interface_losses, total_losses = [], [], [], []

print(f"Model used: {cfg['arch']['arch_dir']}, {arch_type}")

# -----------------------------------------------
# Bucle de entrenamiento
# -----------------------------------------------
for epoch in range(num_epochs):
    total_loss = 0.0

    for batch_idx, (rhs_batch, q_batch, xq_batch) in enumerate(dataloader):
        rhs_batch = rhs_batch.unsqueeze(1).to(device)  # (B, 1, nx, ny, nz)
        mask_tensor = inner_mask.unsqueeze(0).unsqueeze(0).expand_as(rhs_batch).double().to(device)
        data_norm = torch.ones((rhs_batch.size(0),1,1,1,1), dtype=torch.double, device=device) / ratio_max

        input_tensor = torch.cat([rhs_batch, mask_tensor], dim=1)

        output = model(input_tensor)

        # Calcular las pérdidas acumulando por muestra
        lap_total, dirich_total, interf_total = 0, 0, 0
        for i in range(rhs_batch.size(0)):
            q_i = q_batch[i].to(device)
            xq_i = xq_batch[i].to(device)
            out_i = output[i].unsqueeze(0)  # (1, 1, nx, ny, nz)
            dn_i = data_norm[i].unsqueeze(0)

            lap_i = laplacian_loss(out_i, q_i, xq_i, dn_i)
            dirich_i = dirichlet_loss(out_i)
            interf_i = interface_loss(out_i, q_i, xq_i, data_norm=dn_i)

            lap_total += lap_i
            dirich_total += dirich_i
            interf_total += interf_i

        loss = lap_total + dirich_total + interf_total
        loss.backward()
        optimizer.step()

        laplacian_losses.append(lap_total.item() / batch_size)
        dirichlet_losses.append(dirich_total.item() / batch_size)
        interface_losses.append(interf_total.item() / batch_size)
        total_losses.append(loss.item() / batch_size)


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
    f.write(", ".join(map(str, laplacian)) + "\n\n")
    f.write("Dirichlet Losses:\n")
    f.write(", ".join(map(str, dirichlet_losses)) + "\n\n")
    f.write("Interface Losses:\n")
    f.write(", ".join(map(str, interface_losses)) + "\n\n")
    f.write("Total Losses:\n")
    f.write(", ".join(map(str, total_losses)) + "\n")

print(f"Pérdidas guardadas en {loss_file}")
