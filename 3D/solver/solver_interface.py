
import torch
import torch.nn as nn
import numpy as np
import yaml
import matplotlib.pyplot as plt
from unet3d import UNet3D as UNet
from msnet3d import MSNet3D as MSnet
from matplotlib.colors import ListedColormap
import os


with open('C:\Codigos/poissonSolverCNN/3D/solver/solver.yml', 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
scales_data = cfg.get('arch', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
kernel_size = cfg['arch']['kernel_sizes']
network_type = cfg['arch']['type']
plots_dir = os.path.join('results')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

xmin, xmax, nnx = cfg['mesh']['xmin'], cfg['mesh']['xmax'], cfg['mesh']['nnx']
ymin, ymax, nny = cfg['mesh']['ymin'], cfg['mesh']['ymax'], cfg['mesh']['nny']
zmin, zmax, nnz = cfg['mesh']['zmin'], cfg['mesh']['zmax'], cfg['mesh']['nnz']
interface_center, interface_radius = (cfg['mesh']['interface_center']['x'], cfg['mesh']['interface_center']['y'], cfg['mesh']['interface_center']['z']), \
    cfg['mesh']['interface_radius']
x_1d = np.linspace(xmin, xmax, nnx)
y_1d = np.linspace(ymin, ymax, nny)
z_1d = np.linspace(zmin, zmax, nnz)
X, Y, Z = np.meshgrid(x_1d, y_1d, z_1d)
interface_mask = (X - interface_center[0])**2 + (Y - interface_center[1])**2 + (Z -  interface_center[0])**2<= interface_radius**2
domain = np.ones((nnx, nny, nnz))
domain[~interface_mask] *= 0

#Define Gaussians's Functions
def gaussian(x, y, z, amplitude, x0, y0, z0, sigma_x, sigma_y, sigma_z):
    return amplitude * np.exp(-((x - x0) / sigma_x)**2
                              - ((y - y0) / sigma_y)**2
                              - ((z - z0) / sigma_z)**2)
def gaussians(x, y, z, params):
    profile = np.zeros_like(x)
    ngauss = int(len(params) / 7)
    params = np.array(params).reshape(ngauss, 7)
    for index in range(ngauss):
        profile += gaussian(x, y, z, *params[index, :])
    return profile


# input_data = gaussians(X, Y, cfg['init']['args']).astype(np.float32)
input_data = gaussians(X, Y, Z, cfg['init']['args'])
input_data[interface_mask] /= 1
input_data[~interface_mask] /= 80
input_data = input_data[np.newaxis, np.newaxis, :, :, :]
input_data = torch.from_numpy(input_data).float()
input_array = input_data[0, 0, :, :, :]

#Create Model
if network_type == 'UNet':
    model = UNet(scales=scales, kernel_sizes=kernel_size, input_res=nnx)
elif network_type == 'MSNet':
    model = MSnet(scales=scales, kernel_sizes=kernel_size, input_res=nnx)
model.load_state_dict(torch.load('C:/Codigos/poissonSolverCNN/3D/training/models/interface3d_1.pth'))
model = model.float()
for param in model.parameters():
    param.data = param.data.float()
model.eval() 

# Solver
output = model(input_data)
output_array = output.detach().numpy()[0, 0, :, :, :] 

# Slices
input_data_slice = input_data[0, 0, :, :, nnz//2]
output_array_slice = output_array[:, :, nnz//2]
domain_slice = domain[:, :, nnz//2]


# 2D plots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Interface model 1', fontsize=16)

# Plot Input
img_input = axs[0, 0].imshow(input_data_slice, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[0, 0].set_title('Input')
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')
cbar_input = plt.colorbar(img_input, ax=axs[0, 0], label='Magnitude')

# Plot Output
img_output = axs[0, 1].imshow(output_array_slice, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[0, 1].set_title('Output')
axs[0, 1].set_xlabel('X')
axs[0, 1].set_ylabel('Y')
cbar_output = plt.colorbar(img_output, ax=axs[0, 1], label='Magnitude')

# Plot Reference of the Domain
img_domain = axs[1, 0].imshow(domain_slice, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[1, 0].set_title('Domain Reference')
axs[1, 0].set_xlabel('X')
axs[1, 0].set_ylabel('Y')
cbar_domain = plt.colorbar(img_domain, ax=axs[1, 0], label='Magnitude')

# Plot One line
line = output_array[0:nnx, nny//2, nnz//2]
x_line = np.linspace(0, xmax, len(line))
axs[1, 1].plot(x_line, line)
axs[1, 1].set_title('Line')
axs[1, 1].set_xlabel('X')
axs[1, 1].set_ylabel('Y')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
os.makedirs(plots_dir, exist_ok=True)
plt.savefig(os.path.join(plots_dir, 'Interface_3D 1.png'))