import torch
import torch.nn as nn
import numpy as np
import yaml
import matplotlib.pyplot as plt
from unet_interface import UNet
import os


with open('C:\Codigos/poissonSolverCNN/2D/solver/solver.yml', 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
scales_data = cfg.get('arch', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
kernel_size = cfg['arch']['kernel_sizes']
plots_dir = os.path.join('results')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

xmin, xmax, nnx = cfg['mesh']['xmin'], cfg['mesh']['xmax'], cfg['mesh']['nnx']
ymin, ymax, nny  = cfg['mesh']['ymin'], cfg['mesh']['ymax'], cfg['mesh']['nny']
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


input_data = gaussians(X_np, Y_np, cfg['init']['args'])
input_data = input_data[np.newaxis, np.newaxis, :, :]
input_data = torch.from_numpy(input_data).float()

# Create Model
model = UNet(scales, kernel_sizes=kernel_size, input_res=nnx, inner_mask = inner_mask, outer_mask = outer_mask)
model.load_state_dict(torch.load('C:/Codigos/poissonSolverCNN/2D/training/models/interface_18.pth'))
model = model.float()
model.eval() 

# Solver
out_in, out_out = model(input_data)
output = torch.zeros_like(out_in)
output[0, 0, interface_mask] = out_in[0, 0, interface_mask]
output[0, 0, ~interface_mask] = out_out[0, 0, ~interface_mask]
output_array = output.detach().numpy()[0, 0, :, :] 


# Plots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Interface model 1', fontsize=16)

# Plot Input
img_input = axs[0, 0].imshow(input_data[0, 0, :, :], extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[0, 0].set_title('Input')
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')
cbar_input = plt.colorbar(img_input, ax=axs[0, 0], label='Magnitude')

# Plot Output
img_output = axs[0, 1].imshow(output_array, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[0, 1].set_title('Output')
axs[0, 1].set_xlabel('X')
axs[0, 1].set_ylabel('Y')
cbar_output = plt.colorbar(img_output, ax=axs[0, 1], label='Magnitude')

# Plot Reference of the Domain
line = output_array[nnx//2, 0 : nny]
y_line = np.linspace(0, ymax, len(line))
axs[1, 0].plot(y_line, line)
axs[1, 0].set_title('Line in Y')
axs[1, 0].set_xlabel('X')
axs[1, 0].set_ylabel('Y')

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
plt.savefig(os.path.join(plots_dir, 'Interface 18.png'))
