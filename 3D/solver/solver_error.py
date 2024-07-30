
import torch
import torch.nn as nn
import numpy as np
import yaml
import matplotlib.pyplot as plt
from unet3d import UNet3D as UNet
from msnet3d import MSNet3D as MSnet
from matplotlib.colors import ListedColormap
import os
# from ..training.operators3d import ratio_potrhs


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
Lx, Ly, Lz = (xmax - xmin), (ymax - ymin), (zmax - zmin)
x_1d = np.linspace(xmin, xmax, nnx)
y_1d = np.linspace(ymin, ymax, nny)
z_1d = np.linspace(zmin, zmax, nnz)
X, Y, Z = np.meshgrid(x_1d, y_1d, z_1d)


# Define Function to test
def function2Solve(x, y, z):
    return -6 * (x + y + z)

# Define Solution of the function
def resolution(x,y,z):
    return x**3 + y**3 + z**3

# Define Ratio
def ratio_potrhs(alpha, Lx, Ly, Lz):
    return alpha / (np.pi**2 / 8)**2 / (1 / Lx**2 + 1 / Ly**2 + 1 / Lz**2)

alpha = 0.5
ratio = ratio_potrhs(alpha, Lx, Ly, Lz)

input_data = function2Solve(X,Y,Z)
input_data = input_data[np.newaxis, np.newaxis, :, :, :]
input_data = torch.from_numpy(input_data).float()
input_array = input_data[0, 0, :, :, :]
analitical_solution = resolution(X,Y,Z)

#Create Modela
if network_type == 'UNet':
    model = UNet(scales=scales, kernel_sizes=kernel_size, input_res=nnx)
elif network_type == 'MSNet':
    model = MSnet(scales=scales, kernel_sizes=kernel_size, input_res=nnx)
model.load_state_dict(torch.load('C:/Codigos/poissonSolverCNN/3D/training/models/test3d_1.pth'))
model = model.float()
for param in model.parameters():
    param.data = param.data.float()
model.eval() 

#Solver
output = model(input_data) * ratio

# Plots
output_array = output.detach().numpy()[0, 0, :, :, :] 
input_plot = input_data.detach().numpy()[0, 0, :, :, :]
input_slice = input_array[:,:,nnz//2]
ouptut_slice = output_array[:,:,nnz//2]
relative_error_inner = np.abs(output_array - analitical_solution) / np.max(analitical_solution)
fig, axs = plt.subplots(1, 3, figsize=(10, 5)) 
fig.suptitle('Power 3', fontsize=16) 

# Output
img_output = axs[0].imshow(ouptut_slice, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[0].set_title('Ouput')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
cbar_input = plt.colorbar(img_output, ax=axs[0], label='Magnitude')

# Analitical Solution
img_resolution = axs[1].imshow(analitical_solution[:, :, nnz//2], extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[1].set_title('Resolution')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
plt.colorbar(img_resolution, ax=axs[1], label='Magnitude')

# Relative Error
img_error = axs[2].imshow(relative_error_inner[:,:,nnz//2], extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[2].set_title('Relative Error')
axs[2].set_xlabel('X')
axs[2].set_ylabel('Y')
cbar_output = plt.colorbar(img_error, ax=axs[2], label='Magnitude')
plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig(os.path.join(plots_dir, f'Test3d 1.png'))