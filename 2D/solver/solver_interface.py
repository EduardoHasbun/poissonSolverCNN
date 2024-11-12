import torch
import torch.nn as nn
import numpy as np
import yaml
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('C:/Codigos/poissonSolverCNN/2D/training')
import operators as op
from models import UNetInterface




with open('C:\Codigos/poissonSolverCNN/2D/solver/solver_interface.yml', 'r') as file:
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

xmin, xmax, nnx = cfg['mesh']['xmin'], cfg['mesh']['xmax'], cfg['mesh']['nnx']
ymin, ymax, nny  = cfg['mesh']['ymin'], cfg['mesh']['ymax'], cfg['mesh']['nny']
Lx, Ly = xmax - xmin, ymax - ymin
interface_center, interface_radius = (cfg['mesh']['interface_center']['x'], cfg['mesh']['interface_center']['y']), cfg['mesh']['interface_radius']

# Parameters for data
x_1d, y_1d = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny)
X_np, Y_np = np.meshgrid(x_1d, y_1d)
x, y = torch.linspace(xmin, xmax, nnx), torch.linspace(ymin, ymax, nny)
X, Y = torch.meshgrid(x, y, indexing="ij")
interface_mask = (X - interface_center[0])**2 + (Y - interface_center[1])**2 <= interface_radius**2
interface_boundary = torch.zeros_like(interface_mask, dtype=bool)
for i in range(1, interface_mask.shape[0] - 1):
    for j in range(1, interface_mask.shape[1] - 1):
        # Check for boundary change and mark only the outside node
        if interface_mask[i, j] != interface_mask[i - 1, j] or interface_mask[i, j] != interface_mask[i + 1, j]:
            if interface_mask[i, j]:  # If current node is outside the interface
                interface_boundary[i, j] = True
        elif interface_mask[i, j] != interface_mask[i, j - 1] or interface_mask[i, j] != interface_mask[i, j + 1]:
            if interface_mask[i, j]:  # If current node is outside the interface
                interface_boundary[i, j] = True

inner_mask = interface_mask
outer_mask = ~interface_mask | interface_boundary


# Define Gaussians's Functions
def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x)**2
                              - ((y - y0) / sigma_y)**2)
def gaussians(x, y, params):
    profile = np.zeros_like(x)
    ngauss = int(len(params) / 5)
    params = np.array(params).reshape(ngauss, 5)
    for index in range(ngauss):
        profile += gaussian(x, y, *params[index, :])
    return profile

def analytical_solution(x, y, x0, y0, e_in, e_out, R, q):
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    
    # Use np.where to apply conditions element-wise for the entire grid
    r_masked = np.where(r == 0, 1e-10, r)
    solution = np.where(r <= R, 
                    (q / (2 * np.pi * e_in)) * np.log(R / r_masked) - (q / (2 * np.pi * e_out)) * np.log(R), 
                    (q / (2 * np.pi * e_out)) * np.log(r_masked))
    
    solution[r == 0] = q / (2 * np.pi * e_in / 2.5) 

    return solution

# Set parameters
alpha = 0.55
ratio_max = op.ratio_potrhs(alpha, Lx, Ly)


input_data = gaussians(X_np, Y_np, cfg['init']['args'])
input_data = input_data[np.newaxis, np.newaxis, :, :]
input_data = torch.from_numpy(input_data).float() 


solution = analytical_solution(X, Y, interface_center[0], interface_center[1], cfg['mesh']['epsilon_in'], \
                    cfg['mesh']['epsilon_out'], interface_radius, cfg['init']['args'][0])

# Create Model
model = UNetInterface(scales, kernel_sizes=kernel_size, input_res=nnx, inner_mask = inner_mask, outer_mask = outer_mask)
model.load_state_dict(torch.load(model_dir))
model = model.float()
model.eval() 

# Solver
out_in, out_out = model(input_data)
output = torch.zeros_like(out_in)
output[0, 0, interface_mask] = out_in[0, 0, interface_mask]
output[0, 0, ~interface_mask] = out_out[0, 0, ~interface_mask]
output_array = output.detach().numpy()[0, 0, :, :] * ratio_max * 100


# Plots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Interface Unet4 rf300', fontsize=16)

# Plot solution
img_input = axs[0, 0].imshow(solution, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[0, 0].set_title('Analitical Solution')
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')
cbar_input = plt.colorbar(img_input, ax=axs[0, 0], label='Magnitude')

# Plot Output
img_output = axs[0, 1].imshow(output_array, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[0, 1].set_title('Output')
axs[0, 1].set_xlabel('X')
axs[0, 1].set_ylabel('Y')
cbar_output = plt.colorbar(img_output, ax=axs[0, 1], label='Magnitude')

# Plot Relative Error
relative_error = abs(output_array - solution) / np.max(solution) * 100
img_error = axs[1, 0].imshow(relative_error, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[1, 0].set_title('Relatie Error')
axs[1, 0].set_xlabel('X')
axs[1, 0].set_ylabel('Y')
cbar_error = plt.colorbar(img_error, ax=axs[1,0], label='Error')

# Plot One line
line_2 = output_array[0 : nnx, nny//2]
x_line = np.linspace(0, xmax, len(line_2))
axs[1, 1].plot(x_line, line_2)
axs[1, 1].set_title('Line in X')
axs[1, 1].set_xlabel('X')
axs[1, 1].set_ylabel('Y')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])  
os.makedirs(plots_dir, exist_ok=True)
plt.savefig(os.path.join(plots_dir, f'{case_name}.png'))
