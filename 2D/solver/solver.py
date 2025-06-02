import os
import sys
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score  

sys.path.append('C:/Codigos/poissonSolverCNN/2D/training')
from models import UNet, MSNet
import operators as op

# Load YAML config
with open('C:/Codigos/poissonSolverCNN/2D/solver/solver.yml', 'r') as file:
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
params = cfg['init']['args']

# Set mesh
xmin, xmax, nnx = cfg['mesh']['xmin'], cfg['mesh']['xmax'], cfg['mesh']['nnx']
ymin, ymax, nny = cfg['mesh']['ymin'], cfg['mesh']['ymax'], cfg['mesh']['nny']
Lx, Ly = xmax - xmin, ymax - ymin
x_1d = np.linspace(xmin, xmax, nnx)
y_1d = np.linspace(ymin, ymax, nny)
X, Y = np.meshgrid(x_1d, y_1d)

# Create directories if they don't exist
plots_dir = 'results'
errors_file = os.path.join(plots_dir, 'errors_log.txt')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Define Gaussian Functions
def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x)**2 - ((y - y0) / sigma_y)**2)

def gaussians(x, y, params):
    profile = np.zeros_like(x)
    ngauss = int(len(params) / 5)
    params = np.array(params).reshape(ngauss, 5)
    for index in range(ngauss):
        profile += gaussian(x, y, *params[index, :])
    return profile

# Define the analytical solution for Poisson equation for punctual charges
def poisson_punctual_solution(x, y, charges):
    solution = np.zeros_like(x)
    for charge in charges:
        amplitude, x0, y0 = charge
        distance = np.sqrt((x - x0)**2 + (y - y0)**2)
        solution += -amplitude / (4 * np.pi) * np.log(distance + 1e-10)  # Avoid log(0)
    return solution


# Extract punctual charge parameters (amplitude and location) from params
punctual_charges = []
ngauss = int(len(params) / 5)
params_charges = np.array(params).reshape(ngauss, 5)
for index in range(ngauss):
    amplitude, x0, y0, _, _ = params_charges[index]
    punctual_charges.append((amplitude, x0, y0))

# Generate analytical solution for Poisson equation
resolution_data = poisson_punctual_solution(X, Y, punctual_charges)

# Set parameters
alpha = 0.1
ratio_max = op.ratio_potrhs(alpha, Lx, Ly)

# Create input data and resolution data
input_data = gaussians(X, Y, params) * ratio_max
input_data = input_data[np.newaxis, np.newaxis, :, :]
input_data = torch.from_numpy(input_data).float()

# Create Model
if arch_model == 'UNet':
    model = UNet(scales, kernel_sizes=kernel_size, input_res=nnx)
elif arch_model == 'MSNet':
    model = MSNet(scales, kernel_size, input_res=nnx)
model.load_state_dict(torch.load(model_dir))
model = model.float()
model.eval()

# Solver
output = model(input_data)
output_array = output.detach().numpy()[0, 0, :, :] * ratio_max

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

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# fig.suptitle(case_name, fontsize=16)

# Plot Output from NN
vmin, vmax = np.min(output_array), np.max(output_array)
img_output = axs[0].imshow(output_array, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[0].set_title('NN Output')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
plt.colorbar(img_output, ax=axs[0])

# Plot Analytical Solution
img_resolution = axs[1].imshow(resolution_data, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
axs[1].set_title('Analytical Solution')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
plt.colorbar(img_resolution, ax=axs[1])

# Plot Relative Error
img_error = axs[2].imshow(relative_error, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis', vmin=0, vmax=20)
axs[2].set_title('Relative Error (%)')
axs[2].set_xlabel('X')
axs[2].set_ylabel('Y')
plt.colorbar(img_error, ax=axs[2], label='Error %')

# Save and show the plot
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f'{case_name}.png'))
