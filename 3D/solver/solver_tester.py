
import torch
import torch.nn as nn
import numpy as np
import yaml
import matplotlib.pyplot as plt
from unet3d import UNet3D as UNet
from msnet3d import MSNet3D as MSnet
from matplotlib.colors import ListedColormap
# from ..training.operators3d import ratio_potrhs


with open('C:\Codigos/poissonSolverCNN/3D/solver/solver.yml', 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
scales_data = cfg.get('arch', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
kernel_size = cfg['arch']['kernel_sizes']
network_type = cfg['arch']['type']

xmin, xmax, nnx = cfg['mesh']['xmin'], cfg['mesh']['xmax'], cfg['mesh']['nnx']
ymin, ymax, nny = cfg['mesh']['ymin'], cfg['mesh']['ymax'], cfg['mesh']['nny']
zmin, zmax, nnz = cfg['mesh']['zmin'], cfg['mesh']['zmax'], cfg['mesh']['nnz']
x_1d = np.linspace(xmin, xmax, nnx)
y_1d = np.linspace(ymin, ymax, nny)
z_1d = np.linspace(zmin, zmax, nnz)
X, Y, Z = np.meshgrid(x_1d, y_1d, z_1d)


# Define Function to test
def function(x, y, z):
    return 6*x + 6*y + 6*z

# Define Solution of the function
def solution(x,y,z):
    return x**3+y**3+z**3


input_data = function(X,Y,Z)
input_data = input_data[np.newaxis, np.newaxis, :, :, :]
input_data = torch.from_numpy(input_data).float()
input_array = input_data[0, 0, :, :, :]
analitical_solution = solution(X,Y,Z)

#Create Model
if network_type == 'UNet':
    model = UNet(scales=scales, kernel_sizes=kernel_size, input_res=nnx)
elif network_type == 'MSNet':
    model = MSnet(scales=scales, kernel_sizes=kernel_size, input_res=nnx)
model.load_state_dict(torch.load('C:/Codigos/poissonSolverCNN/3D/training/laplacian_loss.pth'))
model = model.float()
for param in model.parameters():
    param.data = param.data.float()
model.eval() 

#Solver
output = model(input_data)
output_array = output.detach().numpy()[0, 0, :, :, :] 
input_plot = input_data.detach().numpy()[0, 0, :, :, :]



# Plots
input_slice = input_array[:,:,nnz//2]
ouptut_slice = output_array[:,:,nnz//2]

plt.figure(figsize=(8, 6))
plt.imshow(input_slice, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
plt.colorbar()
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(ouptut_slice, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
plt.colorbar()
plt.show()


# Define the indices for the inner region
inner_indices_x = slice(7, 24)
inner_indices_y = slice(7, 24)
inner_indices_z = slice(7, 24)

# Extract inner regions of the arrays
output_inner = output_array[inner_indices_x, inner_indices_y, inner_indices_z]
analytical_inner = analitical_solution[inner_indices_x, inner_indices_y, inner_indices_z]

# Calculate the relative error
relative_error_inner = np.abs(output_inner - analytical_inner) / np.abs(analytical_inner)

# Plot the relative error
plt.figure(figsize=(8, 6))
plt.imshow(relative_error_inner[:,:,nnz//2], extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
plt.colorbar(label='Relative Error')
plt.title('Relative Error in Inner Region')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()