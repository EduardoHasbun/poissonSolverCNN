
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

xmin, xmax, nnx = cfg['mesh']['xmin'], cfg['mesh']['xmax'], cfg['mesh']['nnx']
ymin, ymax, nny  = cfg['mesh']['ymin'], cfg['mesh']['ymax'], cfg['mesh']['nny']
x_1d = np.linspace(xmin, xmax, nnx)
y_1d = np.linspace(ymin, ymax, nny)
X, Y = np.meshgrid(x_1d, y_1d)
plots_dir = os.path.join('results')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)


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

def function2solve(x, y):
    return -6 * (x + y)
    # return np.sin(x) + np.sin(y)

def resolution(x, y):
    return x**3 + y**3 
    # return np.sin(x) + np.sin(y)

# Create input data and resolution data for the error
input_data = function2solve(X, Y)
input_data = input_data[np.newaxis, np.newaxis, :, :]
input_data = torch.from_numpy(input_data).float()
resolution_data = resolution(X, Y)

# Create Model
model = UNet(scales, kernel_sizes=kernel_size, input_res=nnx)
model.load_state_dict(torch.load('C:/Codigos/poissonSolverCNN/2D/training/models/model_dirichlet_power3.pth'))
model = model.float()
model.eval() 

# Solver
output = model(input_data)
output_array = output.detach().numpy()[0, 0, :, :] 


# Plots
fig, axs = plt.subplots(1, 3, figsize=(10, 5)) 
fig.suptitle('Power 3', fontsize=16) 

# Plot Input
img_input = axs[0].imshow(output_array, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[0].set_title('Ouput')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
cbar_input = plt.colorbar(img_input, ax=axs[0], label='Magnitude')

# Plot Resolution
img_resolution = axs[1].imshow(resolution_data, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[1].set_title('Resolution')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
cbar_input = plt.colorbar(img_resolution, ax=axs[1], label='Magnitude')

# Plot Output
realitve_error = abs(output_array - resolution_data)/ resolution_data
img_output = axs[2].imshow(realitve_error, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[2].set_title('Relative Error')
axs[2].set_xlabel('X')
axs[2].set_ylabel('Y')
cbar_output = plt.colorbar(img_output, ax=axs[2], label='Magnitude')
plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig(os.path.join(plots_dir, f'Power 3.png'))