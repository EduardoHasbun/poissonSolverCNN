import torch
import torch.nn as nn
import numpy as np
import yaml
import matplotlib.pyplot as plt
from unet import UNet
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
x_1d = np.linspace(xmin, xmax, nnx)
y_1d = np.linspace(ymin, ymax, nny)
X, Y = np.meshgrid(x_1d, y_1d)


# Interface
interface_mask = (X - interface_center[0])**2 + (Y - interface_center[1])**2 <= interface_radius**2
domain = np.ones((nnx, nny))
domain[~interface_mask] *= 0


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


input_data = gaussians(X, Y, cfg['init']['args'])
input_data[interface_mask] /= 1
input_data[~interface_mask] /= 80
input_data = input_data[np.newaxis, np.newaxis, :, :]
input_data = torch.from_numpy(input_data).float()

# Create Model
model = UNet(scales, kernel_sizes=kernel_size, input_res=nnx)
model.load_state_dict(torch.load('C:/Codigos/poissonSolverCNN/2D/training/models/interface_4.pth'))
model = model.float()
model.eval() 

# Solver
output = model(input_data)
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
img_domain = axs[1, 0].imshow(domain, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[1, 0].set_title('Domain Reference')
axs[1, 0].set_xlabel('X')
axs[1, 0].set_ylabel('Y')
cbar_domain = plt.colorbar(img_domain, ax=axs[1, 0], label='Magnitude')

# Plot One line
line = output_array[0:nnx, nny//2]
x_line = np.linspace(0, xmax, len(line))
axs[1, 1].plot(x_line, line)
axs[1, 1].set_title('Line')
axs[1, 1].set_xlabel('X')
axs[1, 1].set_ylabel('Y')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
os.makedirs(plots_dir, exist_ok=True)
plt.savefig(os.path.join(plots_dir, 'Interface 4 epsilon.png'))
