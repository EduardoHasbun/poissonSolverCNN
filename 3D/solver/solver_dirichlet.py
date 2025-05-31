import os
import sys
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score  
sys.path.append('C:/Codigos/poissonSolverCNN/3D/training')
from models import UNet3D
import operators3d as op
from sklearn.metrics import r2_score


sys.path.append('C:/Codigos/poissonSolverCNN/3D/training')

# Load YAML config
with open('C:/Codigos/poissonSolverCNN/3D/solver/solver_dirichlet.yml', 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

# Get paths and configurations from YAML
case_name = cfg['general']['case_name']
model_dir = cfg['general']['model_dir']
arch_type = cfg['arch']['type']
arch_dir = os.path.join('..', '..', cfg['arch']['arch_dir'])

# Load architecture config
with open(arch_dir) as yaml_stream1:
    arch = yaml.safe_load(yaml_stream1)
arch_model = arch[arch_type]['type']
scales_data = arch.get(arch_type, {}).get('args', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
kernel_size = arch[arch_type]['args']['kernel_sizes']

# Set mesh
xmin, xmax, nnx = cfg['mesh']['xmin'], cfg['mesh']['xmax'], cfg['mesh']['nnx']
ymin, ymax, nny = cfg['mesh']['ymin'], cfg['mesh']['ymax'], cfg['mesh']['nny']
zmin, zmax, nnz = cfg['mesh']['zmin'], cfg['mesh']['zmax'], cfg['mesh']['nnz']
Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin
x_1d = np.linspace(xmin, xmax, nnx)
y_1d = np.linspace(ymin, ymax, nny)
z_1d = np.linspace(zmin, zmax, nnz)
X, Y, Z = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')

# Create directories if they don't exist
plots_dir = 'results'
errors_file = os.path.join(plots_dir, 'errors_log.txt')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

def function2solve(x, y, z):
    return -6 * (x + y + z)

def resolution(x, y, z):
    return x**3 + y**3 + z**3

def ratio_potrhs(alpha, Lx, Ly, Lz):
    return alpha / (np.pi**2 / 4)**2 / (1 / Lx**2 + 1 / Ly**2 + 1 / Lz**2)

# Set parameters
alpha = 0.035
ratio_max = op.ratio_potrhs(alpha, Lx, Ly, Lz)

# Create input data and resolution data for the error
input_data = function2solve(X, Y, Z) * ratio_max
input_data = input_data[np.newaxis, np.newaxis, :, :, :]  # shape: (1, 1, nx, ny, nz)
input_data = torch.from_numpy(input_data).float()
resolution_data = resolution(X, Y, Z)

# Create Model
model = UNet3D(scales, kernel_sizes=kernel_size, input_res=nnx)
model.load_state_dict(torch.load(model_dir))
model = model.float()
model.eval()

# Solver
output = model(input_data)
output_array = output.detach().numpy()[0, 0, :, :, :] * ratio_max

# Compute Errors
relative_error = abs(output_array - resolution_data) / np.max(abs(resolution_data)) * 100
max_error = np.max(relative_error)
avg_error = np.average(relative_error)

# Compute R² Variance
def calculate_r2(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    return r2_score(y_true_flat, y_pred_flat)

r2_value = calculate_r2(resolution_data, output_array)

print(f'Max Error: {max_error:.2f}%, Avg Error: {avg_error:.2f}%, R² Variance: {r2_value:.4f}')

# Log case errors
def log_case_error(case_name, max_error, avg_error, r2_value):
    """Logs the case name and errors only if it doesn't already exist in the log file."""
    if not os.path.exists(errors_file):
        with open(errors_file, 'w') as f:
            f.write("Case Name, Max Error (%), Avg Error (%), R^2 Variance\n")

    with open(errors_file, 'r') as f:
        lines = f.readlines()

    # Check if case already exists
    case_exists = any(case_name in line for line in lines)

    if not case_exists:
        with open(errors_file, 'a') as f:
            f.write(f"{case_name}, {max_error:.2f}, {avg_error:.2f}, {r2_value:.4f}\n")

log_case_error(case_name, max_error, avg_error, r2_value)

# Plotting: Show central slices in each direction
mid_x = nnx // 2
mid_y = nny // 2
mid_z = nnz // 2

fig, axs = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle(case_name + " (Central Slices)", fontsize=16)

# X-slice (y-z plane at x=mid_x)
vmin, vmax = np.min(output_array), np.max(output_array)
img_output_x = axs[0, 0].imshow(output_array[mid_x, :, :], extent=(zmin, zmax, ymin, ymax), origin='lower', cmap='viridis', aspect='auto')
axs[0, 0].set_title('NN Output (x-slice)')
plt.colorbar(img_output_x, ax=axs[0, 0])
img_res_x = axs[0, 1].imshow(resolution_data[mid_x, :, :], extent=(zmin, zmax, ymin, ymax), origin='lower', cmap='viridis', aspect='auto')
axs[0, 1].set_title('Analytical (x-slice)')
plt.colorbar(img_res_x, ax=axs[0, 1])
img_err_x = axs[0, 2].imshow(relative_error[mid_x, :, :], extent=(zmin, zmax, ymin, ymax), origin='lower', cmap='viridis', aspect='auto')
axs[0, 2].set_title('Rel. Error (%) (x-slice)')
plt.colorbar(img_err_x, ax=axs[0, 2])

# Y-slice (x-z plane at y=mid_y)
img_output_y = axs[1, 0].imshow(output_array[:, mid_y, :], extent=(zmin, zmax, xmin, xmax), origin='lower', cmap='viridis', aspect='auto')
axs[1, 0].set_title('NN Output (y-slice)')
plt.colorbar(img_output_y, ax=axs[1, 0])
img_res_y = axs[1, 1].imshow(resolution_data[:, mid_y, :], extent=(zmin, zmax, xmin, xmax), origin='lower', cmap='viridis', aspect='auto')
axs[1, 1].set_title('Analytical (y-slice)')
plt.colorbar(img_res_y, ax=axs[1, 1])
img_err_y = axs[1, 2].imshow(relative_error[:, mid_y, :], extent=(zmin, zmax, xmin, xmax), origin='lower', cmap='viridis', aspect='auto')
axs[1, 2].set_title('Rel. Error (%) (y-slice)')
plt.colorbar(img_err_y, ax=axs[1, 2])

# Z-slice (x-y plane at z=mid_z)
img_output_z = axs[2, 0].imshow(output_array[:, :, mid_z], extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis', aspect='auto')
axs[2, 0].set_title('NN Output (z-slice)')
plt.colorbar(img_output_z, ax=axs[2, 0])
img_res_z = axs[2, 1].imshow(resolution_data[:, :, mid_z], extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis', aspect='auto')
axs[2, 1].set_title('Analytical (z-slice)')
plt.colorbar(img_res_z, ax=axs[2, 1])
img_err_z = axs[2, 2].imshow(relative_error[:, :, mid_z], extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis', aspect='auto')
axs[2, 2].set_title('Rel. Error (%) (z-slice)')
plt.colorbar(img_err_z, ax=axs[2, 2])

for i in range(3):
    for j in range(3):
        axs[i, j].set_xlabel('Z' if j < 2 else 'Y')
        axs[i, j].set_ylabel('Y' if i == 0 else ('X' if i == 1 else 'X'))

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig(os.path.join(plots_dir, f'{case_name}_3D_slices.png'))

