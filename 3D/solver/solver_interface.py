import torch
import torch.nn as nn
import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy import special as sp
import os
import sys
sys.path.append('C:/Codigos/poissonSolverCNN/3D/training')
import operators3d as op
from models import UNetInterface

# Load configuration
with open('C:/Codigos/poissonSolverCNN/3D/solver/solver_interface.yml', 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
plots_dir = os.path.join('results')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Get paths and configurations from YAML
case_name = cfg['general']['case_name']
model_dir = cfg['general']['model_dir']
network_type = cfg['arch']['type']
arch_type = cfg['arch']['type']
arch_dir = os.path.join('..', '..', cfg['arch']['arch_dir'])

# Load architecture config
with open(arch_dir) as yaml_stream1:
    arch = yaml.safe_load(yaml_stream1)
arch_model = arch[arch_type]['type']
# Get scales and kernel sizes
scales_data = arch.get(arch_type, {}).get('args', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
kernel_size = arch[arch_type]['args']['kernel_sizes']

# Mesh and Grid
xmin, xmax, nnx = cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['nnx']
ymin, ymax, nny = cfg['domain']['ymin'], cfg['domain']['ymax'], cfg['domain']['nny']
zmin, zmax, nnz = cfg['domain']['zmin'], cfg['domain']['zmax'], cfg['domain']['nnz']
interface_center = (cfg['domain']['interface_center']['x'], cfg['domain']['interface_center']['y'], cfg['domain']['interface_center']['z'])
R = cfg['domain']['R']
x, y, z = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny), np.linspace(zmin, zmax, nnz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
Lx, Ly, Lz = (xmax - xmin), (ymax - ymin), (zmax - zmin)
points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
interface_center = [0, 0, 0]
interface_mask = (X - interface_center[0]) ** 2 + (Y - interface_center[1]) ** 2 + (Z - interface_center[2]) ** 2 <= R ** 2
labels = np.where(interface_mask, "molecule", "solvent")
interface_mask = (X - interface_center[0])**2 + (Y - interface_center[1])**2 + (Z - interface_center[2])**2 <= R**2
interface_boundary = np.zeros_like(interface_mask, dtype=bool)
for i in range(1, interface_mask.shape[0] - 1):
    for j in range(1, interface_mask.shape[1] - 1):
        for k in range (1, interface_mask.shape[2] - 1):
            # Check for boundary change and mark only the outside node
            if interface_mask[i, j, k] != interface_mask[i - 1, j, k] or interface_mask[i, j, k] != interface_mask[i + 1, j, k]:
                if interface_mask[i, j, k]:  # If current node is outside the interface
                    interface_boundary[i, j, k] = True
            elif interface_mask[i, j, k] != interface_mask[i, j - 1, k] or interface_mask[i, j, k] != interface_mask[i, j + 1, k]:
                if interface_mask[i, j, k]:  # If current node is outside the interface
                    interface_boundary[i, j, k] = True
            elif interface_mask[i, j, k] != interface_mask[i, j, k - 1] or interface_mask[i, j, k] != interface_mask[i, j, k+1]:
                if interface_mask[i, j, k]:
                    interface_boundary[i, j, k] = True

inner_mask = interface_mask
outer_mask = ~interface_mask | interface_boundary


# Charges
charges = np.array(cfg['init']['charges'])
locations = np.array(cfg['init']['locations'])
E_1 = cfg['spherical_harmonics']['E_1']
E_2 = cfg['spherical_harmonics']['E_2']
kappa = cfg['spherical_harmonics']['kappa']
N = cfg['spherical_harmonics']['N']

# Functions
def G(X, q, xq, epsilon):
    r_vec_expanded = np.expand_dims(X, axis=1)  # Shape: (n, 1, 3)
    x_qs_expanded = np.expand_dims(xq, axis=0)  # Shape: (1, m, 3)
    r_diff = r_vec_expanded - x_qs_expanded     # Shape: (n, m, 3)
    r = np.sqrt(np.sum(np.square(r_diff), axis=2))  # Shape: (n, m)
    q_over_r = q / r  # Shape: (n, m)
    total_sum = np.sum(q_over_r, axis=1)  # Shape: (n,)
    result = (1 / (epsilon * 4 * np.pi)) * total_sum  # Shape: (n,)
    return result


def Spherical_Harmonics(x, y, z, q, xq, E_1, E_2, kappa, R, labels, points, N):
    # Precompute values for all points
    points = np.array(points)
    rho = np.linalg.norm(points, axis=1)
    zenit = np.arccos(points[:, 2] / rho)
    azim = np.arctan2(points[:, 1], points[:, 0])

    xq = np.array(xq)
    rho_k = np.linalg.norm(xq, axis=1)
    zenit_k = np.arccos(xq[:, 2] / rho_k)
    azim_k = np.arctan2(xq[:, 1], xq[:, 0])
    
    # Precompute the grid indices for labels
    ix = ((points[:, 0] - x[0]) / (x[1] - x[0])).astype(int)
    iy = ((points[:, 1] - y[0]) / (y[1] - y[0])).astype(int)
    iz = ((points[:, 2] - z[0]) / (z[1] - z[0])).astype(int)

    PHI = np.zeros(len(points), dtype=np.complex128)

    # Loop over n and m
    for n in range(N):
        for m in range(-n, n + 1):
            # Compute Enm for all points
            Enm = np.sum(
                q[:, None]
                * rho_k[:, None]**n
                * (4 * np.pi / (2 * n + 1))
                * sp.sph_harm(m, n, -azim_k[:, None], zenit_k[:, None]),
                axis=0
            )
            Anm = Enm * (1 / (4 * np.pi)) * ((2 * n + 1)) / (
                np.exp(-kappa * R) * ((E_1 - E_2) * n * get_K(kappa * R, n) + E_2 * (2 * n + 1) * get_K(kappa * R, n + 1))
            )
            Bnm = 1 / (R ** (2 * n + 1)) * (
                np.exp(-kappa * R) * get_K(kappa * R, n) * Anm - 1 / (4 * np.pi * E_1) * Enm
            )

            # Compute phi based on labels
            is_molecule = labels[ix, iy, iz] == "molecule"
            is_solvent = labels[ix, iy, iz] == "solvent"

            PHI[is_molecule] += (
                Bnm * rho[is_molecule]**n * sp.sph_harm(m, n, azim[is_molecule], zenit[is_molecule])
            )
            PHI[is_solvent] += (
                Anm
                * rho[is_solvent] ** (-n - 1)
                * np.exp(-kappa * rho[is_solvent])
                * get_K(kappa * rho[is_solvent], n)
                * sp.sph_harm(m, n, azim[is_solvent], zenit[is_solvent])
            )

    # Final adjustment for solvent
    is_solvent = labels[ix, iy, iz] == "solvent"
    PHI[is_solvent] -= G(points[is_solvent], q, xq, E_1)

    return np.real(PHI)

def get_K(x, n):
    K = 0.0
    n_fact = sp.factorial(n)
    n_fact2 = sp.factorial(2 * n)
    for s in range(n + 1):
        K += (
            2**s
            * n_fact
            * sp.factorial(2 * n - s)
            / (sp.factorial(s) * n_fact2 * sp.factorial(n - s))
            * x**s
        )
    return K

# Define Ratio
def ratio_potrhs(alpha, Lx, Ly, Lz):
    return alpha / (np.pi**2 / 8)**2 / (1 / Lx**2 + 1 / Ly**2 + 1 / Lz**2)

# Create Model
model = UNetInterface(scales, kernel_sizes=kernel_size, input_res=nnx, inner_mask = inner_mask, outer_mask = outer_mask)
model.load_state_dict(torch.load(model_dir))
model = model.float()
for param in model.parameters():
    param.data = param.data.float()
model.eval() 
alpha = 0.1
ratio = ratio_potrhs(alpha, Lx, Ly, Lz)


# Create input data and solution data
input_data = G(points, charges, locations, E_1)
input_data = input_data.reshape(nnx, nny, nnz)
input_data = input_data[np.newaxis, np.newaxis, :, :, :]
input_data = torch.from_numpy(input_data).float()
analitical_solution = Spherical_Harmonics(x, y, z, charges, locations, E_1, E_2, kappa, R, labels, points, N)
analitical_solution = analitical_solution.reshape(nnx, nny, nnz)
mid_x, mid_y, mid_z = nnx // 2, nny // 2, nnz // 2
neighbors = [
        analitical_solution[mid_x + 1, mid_y, mid_z],
        analitical_solution[mid_x - 1, mid_y, mid_z],
        analitical_solution[mid_x, mid_y + 1, mid_z],
        analitical_solution[mid_x, mid_y - 1, mid_z],
        analitical_solution[mid_x, mid_y, mid_z + 1],
        analitical_solution[mid_x, mid_y, mid_z - 1]
    ]
analitical_solution[mid_x, mid_y, mid_z] = np.mean(neighbors)

#Solver
out_in, out_out = model(input_data)
output = torch.zeros_like(out_in)
output[0, 0, interface_mask] = out_in[0, 0, interface_mask]
output[0, 0, ~interface_mask] = out_out[0, 0, ~interface_mask]
output_array = output.detach().numpy()[0, 0, :, :] * ratio

# Plots
input_plot = input_data.detach().numpy()[0, 0, :, :, :]
input_slice = input_plot[:,:,nnz//2]
ouptut_slice = output_array[:,:,nnz//2]
relative_error_inner = (
    np.abs(output_array - analitical_solution) 
    / np.abs(analitical_solution)
) * 100
fig, axs = plt.subplots(2, 2, figsize=(10, 10)) 
fig.suptitle('3D', fontsize=16) 

# Output
img_output = axs[0,0].imshow(ouptut_slice, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[0,0].set_title('Ouput')
axs[0,0].set_xlabel('X')
axs[0,0].set_ylabel('Y')
cbar_input = plt.colorbar(img_output, ax=axs[0,0], label='Magnitude')

# Analitical Solution
img_resolution = axs[0,1].imshow(analitical_solution[:, :, nnz//2], extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[0,1].set_title('Resolution')
axs[0,1].set_xlabel('X')
axs[0,1].set_ylabel('Y')
plt.colorbar(img_resolution, ax=axs[0,1], label='Magnitude')

# Relative Error
img_error = axs[1,0].imshow(relative_error_inner[:,:,nnz//2], extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
print(f'Error Maximo: {np.max(relative_error_inner)}, Error Promedio: {np.average(relative_error_inner)}')
axs[1,0].set_title('Relative Error')
axs[1,0].set_xlabel('X')
axs[1,0].set_ylabel('Y')
cbar_output = plt.colorbar(img_error, ax=axs[1,0], label='Relative Error %')

# Absolute Error
img_error_abs = axs[1,1].imshow(np.abs(output_array - analitical_solution)[:,:,nnz//2], extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[1,1].set_title('Absolute Error')
axs[1,1].set_xlabel('X')
axs[1,1].set_ylabel('Y')
cbar_output = plt.colorbar(img_error_abs, ax=axs[1,1], label='Absolute Error')


plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig(os.path.join(plots_dir, case_name + '.png'))



# L2 error norm
error = output_array - analitical_solution
print('Error absoluto maximo:', np.max(error))
error_flat = error.ravel()
l2_error = np.linalg.norm(error_flat, 2)
print("Norma L2 (discreta):", l2_error)
l2_analytical = np.linalg.norm(analitical_solution.ravel(), 2)
relative_l2_error = l2_error / l2_analytical
print("Error relativo L2:", relative_l2_error)


# After you have 'output_array' and 'analitical_solution' defined
# and your 'charges' and 'locations' arrays, do:

# Make sure locations is in shape (n_charges, 3)
# charges has shape (n_charges,)
# 'x', 'y', 'z' are your 1D coordinate arrays.

for i, (charge, loc) in enumerate(zip(charges, locations)):
    # Find the indices in x, y, z that are closest to the charge location
    i_x = np.argmin(np.abs(x - loc[0]))
    i_y = np.argmin(np.abs(y - loc[1]))
    i_z = np.argmin(np.abs(z - loc[2]))

    # Compute absolute error at that grid point
    abs_err = np.abs(output_array[i_x, i_y, i_z] - analitical_solution[i_x, i_y, i_z])

    # Compute relative error at that grid point
    # (make sure the analytical solution at this point isn't zero!)
    denom = np.abs(analitical_solution[i_x, i_y, i_z])
    if denom < 1e-14:
        rel_err = np.nan  # or handle differently if needed
    else:
        rel_err = (abs_err / denom) * 100

    # Print or store the error
    print(f"Charge {i}:")
    print(f"  Location        = {loc}")
    print(f"  Grid index      = ({i_x}, {i_y}, {i_z})")
    print(f"  Absolute error  = {abs_err}")
    print(f"  Relative error  = {rel_err:.4f} %\n")
