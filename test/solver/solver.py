import torch
import torch.nn as nn
import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy import special as sp
import os
import sys

# Custom module paths
sys.path.append('C:/Codigos/poissonSolverCNN/test/training')
import operators as op
from models import UNet3DCombined

# === Load configuration ===
with open('C:/Codigos/poissonSolverCNN/test/solver/solver.yml', 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

# === Directories ===
plots_dir = 'results'
os.makedirs(plots_dir, exist_ok=True)
case_name = cfg['general']['case_name']
model_dir = cfg['general']['model_dir']
arch_dir = os.path.join('..', '..', cfg['arch']['arch_dir'])
arch_type = cfg['arch']['type']

# === Load architecture config ===
with open(arch_dir) as yaml_stream:
    arch = yaml.safe_load(yaml_stream)
arch_model = arch[arch_type]['type']
scales_data = arch[arch_type]['args'].get('scales', {})
scales = [v for k, v in sorted(scales_data.items())]
kernel_size = arch[arch_type]['args']['kernel_sizes']

# === Domain & Grid ===
xmin, xmax, nnx = cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['nnx']
ymin, ymax, nny = cfg['domain']['ymin'], cfg['domain']['ymax'], cfg['domain']['nny']
zmin, zmax, nnz = cfg['domain']['zmin'], cfg['domain']['zmax'], cfg['domain']['nnz']
R = cfg['domain']['R']
interface_center = [0, 0, 0]

x = np.linspace(xmin, xmax, nnx)
y = np.linspace(ymin, ymax, nny)
z = np.linspace(zmin, zmax, nnz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin
points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

# === Interface Masks ===
interface_mask = (X - interface_center[0])**2 + (Y - interface_center[1])**2 + (Z - interface_center[2])**2 <= R**2
labels = np.where(interface_mask, "molecule", "solvent")

interface_boundary = np.zeros_like(interface_mask, dtype=bool)
for i in range(1, nnx - 1):
    for j in range(1, nny - 1):
        for k in range(1, nnz - 1):
            current = interface_mask[i, j, k]
            if any(current != interface_mask[ni, j, k] for ni in [i-1, i+1]) or \
               any(current != interface_mask[i, nj, k] for nj in [j-1, j+1]) or \
               any(current != interface_mask[i, j, nk] for nk in [k-1, k+1]):
                if current:
                    interface_boundary[i, j, k] = True

inner_mask = interface_mask
outer_mask = ~interface_mask | interface_boundary

# === Load Charges and Parameters from 'args' ===
params_arr = np.array(cfg['init']['args'], dtype=np.float64).reshape(-1, 7)
charges = params_arr[:, 0]                     # A
locations = params_arr[:, 1:4]                 # x0, y0, z0
ngauss = len(params_arr)
params = params_arr

# === Constants for Spherical Harmonics ===
E_1 = cfg['spherical_harmonics']['E_1']
E_2 = cfg['spherical_harmonics']['E_2']
kappa = cfg['spherical_harmonics']['kappa']
N = cfg['spherical_harmonics']['N']

# === Utility Functions ===
def G(X, q, xq, epsilon):
    r = np.linalg.norm(X[:, None] - xq[None, :], axis=2)
    return (1 / (epsilon * 4 * np.pi)) * np.sum(q / r, axis=1)

def get_K(x, n):
    K = 0.0
    for s in range(n + 1):
        K += (2**s * sp.factorial(n) * sp.factorial(2 * n - s)) / \
             (sp.factorial(s) * sp.factorial(2 * n) * sp.factorial(n - s)) * x**s
    return K

def Spherical_Harmonics(x, y, z, q, xq, E_1, E_2, kappa, R, labels, points, N):
    rho = np.linalg.norm(points, axis=1)
    zenit = np.arccos(points[:, 2] / rho)
    azim = np.arctan2(points[:, 1], points[:, 0])

    rho_k = np.linalg.norm(xq, axis=1)
    zenit_k = np.arccos(xq[:, 2] / rho_k)
    azim_k = np.arctan2(xq[:, 1], xq[:, 0])

    ix = ((points[:, 0] - x[0]) / (x[1] - x[0])).astype(int)
    iy = ((points[:, 1] - y[0]) / (y[1] - y[0])).astype(int)
    iz = ((points[:, 2] - z[0]) / (z[1] - z[0])).astype(int)

    PHI = np.zeros(len(points), dtype=np.complex128)

    for n in range(N):
        for m in range(-n, n + 1):
            Enm = np.sum(
                q[:, None] * rho_k[:, None]**n * (4 * np.pi / (2 * n + 1)) *
                sp.sph_harm(m, n, -azim_k[:, None], zenit_k[:, None]), axis=0
            )
            Anm = Enm * (1 / (4 * np.pi)) * ((2 * n + 1)) / (
                np.exp(-kappa * R) * ((E_1 - E_2) * n * get_K(kappa * R, n) + E_2 * (2 * n + 1) * get_K(kappa * R, n + 1))
            )
            Bnm = 1 / (R ** (2 * n + 1)) * (
                np.exp(-kappa * R) * get_K(kappa * R, n) * Anm - 1 / (4 * np.pi * E_1) * Enm
            )
            is_molecule = labels[ix, iy, iz] == "molecule"
            is_solvent = labels[ix, iy, iz] == "solvent"
            PHI[is_molecule] += Bnm * rho[is_molecule]**n * sp.sph_harm(m, n, azim[is_molecule], zenit[is_molecule])
            PHI[is_solvent] += Anm * rho[is_solvent]**(-n - 1) * np.exp(-kappa * rho[is_solvent]) * \
                               get_K(kappa * rho[is_solvent], n) * sp.sph_harm(m, n, azim[is_solvent], zenit[is_solvent])

    PHI[is_solvent] -= G(points[is_solvent], q, xq, E_1)
    return np.real(PHI)

def gaussian(x, y, z, A, x0, y0, z0, sigx, sigy, sigz):
    return A * np.exp(-((x - x0) / sigx)**2 - ((y - y0) / sigy)**2 - ((z - z0) / sigz)**2)

def gaussians(x, y, z, params):
    profile = np.zeros_like(x)
    for p in np.array(params).reshape(-1, 7):
        profile += gaussian(x, y, z, *p)
    return profile

def ratio_potrhs(alpha, Lx, Ly, Lz):
    return alpha / (np.pi**2 / 8)**2 / (1 / Lx**2 + 1 / Ly**2 + 1 / Lz**2)

# === Model ===
model = UNet3DCombined(scales=scales, kernel_sizes=kernel_size, input_res=[nnx, nny, nnz]).double()
model.load_state_dict(torch.load(model_dir))
model = model.float()
model.eval()

# === Input & Ground Truth with 2 channels ===
alpha = 0.1
ratio_max = ratio_potrhs(alpha, Lx, Ly, Lz)

# Input: Gaussian profile
input_data = gaussians(X, Y, Z, params) * ratio_max
data_tensor = torch.from_numpy(input_data[np.newaxis, np.newaxis]).float()

# Mask: Interface indicator
mask_tensor = torch.from_numpy(interface_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

# Combine as 2-channel input
input_tensor = torch.cat([data_tensor, mask_tensor], dim=1)

# Analytical Solution
analytical = Spherical_Harmonics(x, y, z, charges, locations, E_1, E_2, kappa, R, labels, points, N).reshape(nnx, nny, nnz)

# Patch singularity at center
mid = (nnx // 2, nny // 2, nnz // 2)
neighbors = [analytical[mid[0] + dx, mid[1] + dy, mid[2] + dz] for dx, dy, dz in
             [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]]
analytical[mid] = np.mean(neighbors)


# === Model Prediction ===
output = model(input_tensor)
output_array = output.detach().numpy()[0, 0, :, :, :] * ratio_max

# === Plotting ===
rel_error = (np.abs(output_array - analytical) / np.abs(analytical)) * 100
abs_error = np.abs(output_array - analytical)
slice_z = nnz // 2

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0,0].imshow(output_array[:, :, slice_z], extent=(xmin, xmax, ymin, ymax), origin='lower')
axs[0,0].set_title("Model Output")

axs[0,1].imshow(analytical[:, :, slice_z], extent=(xmin, xmax, ymin, ymax), origin='lower')
axs[0,1].set_title("Analytical Solution")

axs[1,0].imshow(rel_error[:, :, slice_z], extent=(xmin, xmax, ymin, ymax), origin='lower')
axs[1,0].set_title("Relative Error (%)")

axs[1,1].imshow(abs_error[:, :, slice_z], extent=(xmin, xmax, ymin, ymax), origin='lower')
axs[1,1].set_title("Absolute Error")

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f"{case_name}.png"))

# === Error Metrics ===
print(f"Error Máximo: {np.max(rel_error):.4f}% - Promedio: {np.mean(rel_error):.4f}%")
l2_error = np.linalg.norm((output_array - analytical).ravel(), 2)
l2_ref = np.linalg.norm(analytical.ravel(), 2)
print(f"Norma L2: {l2_error:.4e} - Error Relativo L2: {l2_error / l2_ref:.4%}")

# === Charge-wise Errors ===
for i, (charge, loc) in enumerate(zip(charges, locations)):
    i_x = np.argmin(np.abs(x - loc[0]))
    i_y = np.argmin(np.abs(y - loc[1]))
    i_z = np.argmin(np.abs(z - loc[2]))
    abs_err = np.abs(output_array[i_x, i_y, i_z] - analytical[i_x, i_y, i_z])
    denom = np.abs(analytical[i_x, i_y, i_z])
    rel_err = np.nan if denom < 1e-14 else (abs_err / denom) * 100
    print(f"Charge {i}: Location={loc}, Abs Error={abs_err:.4e}, Rel Error={rel_err:.4f}%")
