import torch
import torch.nn as nn
import numpy as np
import yaml
import matplotlib.pyplot as plt
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

# Mesh parameters
xmin, xmax, nnx = cfg['mesh']['xmin'], cfg['mesh']['xmax'], cfg['mesh']['nnx']
ymin, ymax, nny = cfg['mesh']['ymin'], cfg['mesh']['ymax'], cfg['mesh']['nny']
zmin, zmax, nnz = cfg['mesh']['zmin'], cfg['mesh']['zmax'], cfg['mesh']['nnz']
Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin
interface_center = (cfg['mesh']['interface_center']['x'],
                    cfg['mesh']['interface_center']['y'],
                    cfg['mesh']['interface_center']['z'])
interface_radius = cfg['mesh']['interface_radius']

# Parameters for data
x_1d = np.linspace(xmin, xmax, nnx)
y_1d = np.linspace(ymin, ymax, nny)
z_1d = np.linspace(zmin, zmax, nnz)
X_np, Y_np, Z_np = np.meshgrid(x_1d, y_1d, z_1d, indexing="ij")
x = torch.linspace(xmin, xmax, nnx)
y = torch.linspace(ymin, ymax, nny)
z = torch.linspace(zmin, zmax, nnz)
X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
interface_mask = (X - interface_center[0])**2 + (Y - interface_center[1])**2 + (Z - interface_center[2])**2 <= interface_radius**2
interface_boundary = torch.zeros_like(interface_mask, dtype=bool)
for i in range(1, interface_mask.shape[0] - 1):
    for j in range(1, interface_mask.shape[1] - 1):
        for k in range(1, interface_mask.shape[2] - 1):
            # Check for boundary change and mark only the outside node
            if interface_mask[i, j, k] != interface_mask[i - 1, j, k] or interface_mask[i, j, k] != interface_mask[i + 1, j, k]:
                if interface_mask[i, j, k]:  # If current node is inside the interface
                    interface_boundary[i, j, k] = True
            elif interface_mask[i, j, k] != interface_mask[i, j - 1, k] or interface_mask[i, j, k] != interface_mask[i, j + 1, k]:
                if interface_mask[i, j, k]:  # If current node is inside the interface
                    interface_boundary[i, j, k] = True
            elif interface_mask[i, j, k] != interface_mask[i, j, k - 1] or interface_mask[i, j, k] != interface_mask[i, j, k + 1]:
                if interface_mask[i, j, k]:
                    interface_boundary[i, j, k] = True

inner_mask = interface_mask
outer_mask = ~interface_mask | interface_boundary

def charges(X, Y, Z, charge_args):
    """
    Generate the RHS charge distribution.
    Args:
        X, Y, Z: Meshgrid arrays for spatial coordinates.
        charge_args: List of arguments, [charge_value] in this case.
    Returns:
        3D array representing the charge distribution in the domain.
    """
    charge_value = charge_args[0]  # Single charge value
    nx, ny, nz = X.shape
    center_x, center_y, center_z = (nx // 2, ny // 2, nz // 2)
    charge_distribution = np.zeros_like(X)
    charge_distribution[center_x, center_y, center_z] = charge_value
    return charge_distribution

# Define BornIonSolver class
class BornIonSolver:
    def __init__(self, epsilon_1, epsilon_2, kappa, qs, R_mol):
        self.epsilon_1 = epsilon_1
        self.epsilon_2 = epsilon_2
        self.kappa = kappa
        self.qs = qs
        self.R_mol = R_mol

    def analytic_Born_Ion(self, r, R=None, index_q=0):
        if R is None:
            R = self.R_mol
        epsilon_1 = self.epsilon_1
        epsilon_2 = self.epsilon_2
        kappa = self.kappa
        q = self.qs[index_q]

        f_IN = lambda r: (q / (4 * np.pi)) * (-1 / (epsilon_1 * R) + 1 / (epsilon_2 * (1 + kappa * R) * R))
        f_OUT = lambda r: (q / (4 * np.pi)) * (np.exp(-kappa * (r - R)) / (epsilon_2 * (1 + kappa * R) * r) - 1 / (epsilon_1 * r))

        y = np.piecewise(r, [r <= R, r > R], [f_IN, f_OUT])
        return y

# Set parameters
alpha = 0.38
ratio_max = op.ratio_potrhs(alpha, Lx, Ly, Lz)
epsilon_in = cfg['mesh']['epsilon_in']
epsilon_out = cfg['mesh']['epsilon_out']
kappa = cfg['mesh'].get('kappa', 0.0)  # Ensure kappa exists in cfg, default to 0 if not
qs = [cfg['init']['args'][0]]  # Assuming init args has the charge value
R_mol = interface_radius

# Create BornIonSolver instance
born_solver = BornIonSolver(epsilon_1=epsilon_in, epsilon_2=epsilon_out, kappa=kappa, qs=qs, R_mol=R_mol) 

# Compute radial distances
r = np.sqrt((X_np - interface_center[0])**2 + (Y_np - interface_center[1])**2 + (Z_np - interface_center[2])**2)

# Compute analytical solution
solution = born_solver.analytic_Born_Ion(r)

# Prepare input data
input_data = charges(X_np, Y_np, Z_np, cfg['init']['args'])
input_data = input_data[np.newaxis, np.newaxis, :, :]
input_data = torch.from_numpy(input_data).float() * ratio_max

# Create Model
model = UNetInterface(scales, kernel_sizes=kernel_size, input_res=nnx, inner_mask=inner_mask, outer_mask=outer_mask)
model.load_state_dict(torch.load(model_dir))
model = model.float()
model.eval()

# Solver
out_in, out_out = model(input_data)
output = torch.zeros_like(out_in)
output[0, 0, interface_mask] = out_in[0, 0, interface_mask]
output[0, 0, ~interface_mask] = out_out[0, 0, ~interface_mask]
output_array = output.detach().numpy()[0, 0, :, :] * -ratio_max

# Plots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Results', fontsize=16)

# Plot analytical solution
vmin = np.min(solution)
vmax = np.max(solution)
img_input = axs[0, 0].imshow(solution[:, :, nnz // 2], extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
axs[0, 0].set_title('Analytical Solution')
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')
cbar_input = plt.colorbar(img_input, ax=axs[0, 0], label='Magnitude')

# Plot Output
img_output = axs[0, 1].imshow(output_array[:, :, nnz // 2], extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
axs[0, 1].set_title('Output')
axs[0, 1].set_xlabel('X')
axs[0, 1].set_ylabel('Y')
cbar_output = plt.colorbar(img_output, ax=axs[0, 1], label='Magnitude')

# Plot Relative Error
relative_error = np.abs(output_array - solution) / np.max(np.abs(solution)) * 100
img_error = axs[1, 0].imshow(relative_error[:, :, nnz // 2], extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[1, 0].set_title('Relative Error (%)')
axs[1, 0].set_xlabel('X')
axs[1, 0].set_ylabel('Y')
cbar_error = plt.colorbar(img_error, ax=axs[1, 0], label='Error (%)')

# Adjust layout and save the plot
plt.tight_layout(rect=[0, 0, 1, 0.96])
os.makedirs(plots_dir, exist_ok=True)
plt.savefig(os.path.join(plots_dir, f'{case_name}.png'))
plt.show()
