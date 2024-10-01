import torch
import torch.nn as nn
import numpy as np
import yaml
import matplotlib.pyplot as plt
from unet3d import UNet3D as UNet
from msnet3d import MSNet3D as MSnet
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
x_1d = np.linspace(xmin, xmax, nnx)
y_1d = np.linspace(ymin, ymax, nny)
z_1d = np.linspace(zmin, zmax, nnz)
X, Y, Z = np.meshgrid(x_1d, y_1d, z_1d)


ratio = 0.022

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

def compute_potential(q, X, Y, Z, epsilon_0=1.0):
    # Desplazar las coordenadas para centrar el origen en (2.5, 2.5, 2.5)
    X_centered = X - 2.5
    Y_centered = Y - 2.5
    Z_centered = Z - 2.5
    
    # Calcular la distancia r al centro (2.5, 2.5, 2.5)
    r = np.sqrt(X_centered**2 + Y_centered**2 + Z_centered**2)

    # Evitar la división por cero al calcular el potencial
    with np.errstate(divide='ignore', invalid='ignore'):
        potential = q / (4 * np.pi * epsilon_0 * r)

    # Asignar el valor correcto al nodo central de la malla
    # Asumiendo que (2.5, 2.5, 2.5) es el centro del dominio y coincide con un punto en la malla
    center_index_x = X.shape[0] // 2  # Índice en X
    center_index_y = Y.shape[1] // 2  # Índice en Y
    center_index_z = Z.shape[2] // 2  # Índice en Z

    # Asignar manualmente el valor en el centro donde r = 0
    potential[center_index_x, center_index_y, center_index_z] = q / (4 * np.pi * epsilon_0)

    return potential




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
model.load_state_dict(torch.load('C:/Codigos/poissonSolverCNN/3D/training/models/result3d_charge.pth'))
model = model.float()
for param in model.parameters():
    param.data = param.data.float()
model.eval() 

#Solver
output = model(input_data) / ratio
output_array = output.detach().numpy()[0, 0, :, :, :] 
output_array_slice = output_array[:, :, nnz//2]

# Compute analitical solution
solution = compute_potential(cfg['init']['args'][0], X, Y, Z)
solution_slide = solution[:, :, nnz//2]


# 2D plots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Interface model 1', fontsize=16)

# Plot Input
img_input = axs[0, 0].imshow(output_array_slice, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[0, 0].set_title('Ouptut')
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')
cbar_input = plt.colorbar(img_input, ax=axs[0, 0], label='Magnitude')

# Plot Output
img_output = axs[0, 1].imshow(solution_slide, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[0, 1].set_title('Analitical Solution')
axs[0, 1].set_xlabel('X')
axs[0, 1].set_ylabel('Y')
cbar_output = plt.colorbar(img_output, ax=axs[0, 1], label='Magnitude')

# Plot Y line
line2 = output_array[nnx//2, 0: nny, nnz//2]
y_line = np.linspace(0, ymax, len(line2))
axs[1, 0].plot(y_line, line2)
axs[1, 0].set_title('Y Line')


# Plot Relative Error
relative_error = np.abs(output_array - solution) / np.max(solution) * 100
relative_error[nnx//2, nny//2, nnz//2] = 1
img_output = axs[1, 1].imshow(relative_error[:, :, nnz//2], extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[1, 1].set_title('Realtive Error')
axs[1, 1].set_xlabel('X')
axs[1, 1].set_ylabel('Y')
cbar_output = plt.colorbar(img_output, ax=axs[1, 1], label='Error %')


# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])  
os.makedirs(plots_dir, exist_ok=True)
plt.savefig(os.path.join(plots_dir, 'Result Charge 3D.png'))