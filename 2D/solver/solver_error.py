import os
import torch
import torch.nn as nn
import numpy as np
import yaml
import matplotlib.pyplot as plt
from model import UNet, MSNet
from ..training.operators import DirichletBoundaryLossFunction


# Load YAML config
with open('C:\Codigos/poissonSolverCNN/2D/solver/solver.yml', 'r') as file:
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
# Get scales and kernel sizes
scales_data = arch.get(arch_type, {}).get('args', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
kernel_size = arch[arch_type]['args']['kernel_sizes']

# Set mesh
xmin, xmax, nnx = cfg['mesh']['xmin'], cfg['mesh']['xmax'], cfg['mesh']['nnx']
ymin, ymax, nny = cfg['mesh']['ymin'], cfg['mesh']['ymax'], cfg['mesh']['nny']
Lx = xmax - xmin
Ly = ymax - ymin
x_1d = np.linspace(xmin, xmax, nnx)
y_1d = np.linspace(ymin, ymax, nny)
X, Y = np.meshgrid(x_1d, y_1d)

# Create directory to save plots
plots_dir = os.path.join('results')
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

# Define functions
def function2solve(x, y):
    return -6 * (x + y)

def resolution(x, y):
    return x**3 + y**3 

def ratio_potrhs(alpha, Lx, Ly):
    return alpha / (np.pi**2 / 4)**2 / (1 / Lx**2 + 1 / Ly**2)

# Set parameters
alpha = 0.04
ratio_max = ratio_potrhs(alpha, Lx, Ly)

# Create input data and resolution data
input_data = function2solve(X, Y) * ratio_max
input_data = input_data[np.newaxis, np.newaxis, :, :]
input_data = torch.from_numpy(input_data).float()
resolution_data = resolution(X, Y)
input_array = input_data.detach().numpy()[0, 0, :, :]

# Create Model
if arch_model == 'UNet':
    model = UNet(scales, kernel_sizes = kernel_size, input_res = nnx)
elif arch_model == 'MSNet':
    model = MSNet(scales, kernel_size, input_res = nnx)
model.load_state_dict(torch.load(model_dir))
model = model.float()
model.eval()

# Solver
output = model(input_data)
output_array = output.detach().numpy()[0, 0, :, :] * ratio_max

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(10, 5))
fig.suptitle(case_name, fontsize=16)

# Plot Output
img_output = axs[0].imshow(output_array, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[0].set_title('Output')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
plt.colorbar(img_output, ax=axs[0])

# Plot Resolution
img_resolution = axs[1].imshow(resolution_data, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[1].set_title('Resolution')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
plt.colorbar(img_resolution, ax=axs[1])

# Plot Relative Error
relative_error = abs(output_array - resolution_data) / np.max(resolution_data) * 100
print(f'Error Maximo: {np.max(relative_error)}, Error Promedio: {np.average(relative_error)}')
img_error = axs[2].imshow(relative_error, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[2].set_title('Relative Error')
axs[2].set_xlabel('X')
axs[2].set_ylabel('Y')
plt.colorbar(img_error, ax=axs[2], label='Error %')

# Save the plot
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f'{case_name}.png'))
