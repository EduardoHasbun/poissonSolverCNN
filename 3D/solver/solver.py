
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
input_data = input_data[np.newaxis, np.newaxis, :, :, :]
input_data = torch.from_numpy(input_data).float()
input_array = input_data[0, 0, :, :, :]

#Create Model
if network_type == 'UNet':
    model = UNet(scales=scales, kernel_sizes=kernel_size, input_res=nnx)
elif network_type == 'MSNet':
    model = MSnet(scales=scales, kernel_sizes=kernel_size, input_res=nnx)
model.load_state_dict(torch.load('C:/Codigos/poissonSolverCNN/3D/training/one_charge.pth'))
model = model.float()
for param in model.parameters():
    param.data = param.data.float()
model.eval() 

#Solver
output = model(input_data)
output_array = output.detach().numpy()[0, 0, :, :, :] 
input_plot = input_data.detach().numpy()[0, 0, :, :, :]
# print(np.max(output_array))


#Plot 3D
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# x_grid, y_grid, z_grid = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')
# scatter = ax.scatter(x_grid, y_grid, z_grid, c=output_array, cmap='viridis')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# cbar = plt.colorbar(scatter, ax=ax, orientation='vertical')
# cbar.set_label('Color Scale')
# plt.show()



# 2d
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