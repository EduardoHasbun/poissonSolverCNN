import torch
import torch.nn as nn
import numpy as np
import yaml
import matplotlib.pyplot as plt
from unet import UNet

# Load configuration
with open('C:\Codigos/poissonSolverCNN/2D/solver/solver.yml', 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

# Define mesh parameters
scales_data = cfg.get('arch', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
kernel_size = cfg['arch']['kernel_sizes']
xmin, xmax, nnx = cfg['mesh']['xmin'], cfg['mesh']['xmax'], cfg['mesh']['nnx']
ymin, ymax, nny = cfg['mesh']['ymin'], cfg['mesh']['ymax'], cfg['mesh']['nny']
x_1d = np.linspace(xmin, xmax, nnx)
y_1d = np.linspace(ymin, ymax, nny)
X, Y = np.meshgrid(x_1d, y_1d)

# Define Gaussians's Functions
def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x)**2 - ((y - y0) / sigma_y)**2)

def gaussians(x, y, params):
    profile = np.zeros_like(x)
    ngauss = int(len(params) / 5)
    params = np.array(params).reshape(ngauss, 5)
    for index in range(ngauss):
        profile += gaussian(x, y, *params[index, :])
    return profile

# Generate input data
input_data = gaussians(X, Y, cfg['init']['args'])
input_data = input_data[np.newaxis, np.newaxis, :, :]
input_data = torch.from_numpy(input_data).float()

# Define circular interface parameters
interface_center = (cfg['mesh']['interface_center']['x'], cfg['mesh']['interface_center']['y'])
interface_radius = cfg['mesh']['interface_radius']

# Generate circular mask for interface
interface_mask = (X - interface_center[0])**2 + (Y - interface_center[1])**2 <= interface_radius**2

# Apply circular mask to split input data
input_data_subdomain1 = input_data.clone()
input_data_subdomain2 = input_data.clone()
input_data_subdomain1[:, :, ~interface_mask] = 0
input_data_subdomain2[:, :, interface_mask] = 0


# Load models
model_subdomain1 = UNet(scales=scales, kernel=kernel_size)
model_subdomain1.load_state_dict(torch.load('C:/Codigos/poissonSolverCNN/2D/training/unet_model.pth'))
model_subdomain1.eval()

model_subdomain2 = UNet(scales=scales, kernel=kernel_size)
model_subdomain2.load_state_dict(torch.load('C:/Codigos/poissonSolverCNN/2D/training/unet_model.pth'))
model_subdomain2.eval()

# Solver for subdomain 1
output_subdomain1 = model_subdomain1(input_data_subdomain1)

# Solver for subdomain 2
output_subdomain2 = model_subdomain2(input_data_subdomain2)

# Apply continuity condition at the interface
phi_subdomain1 = output_subdomain1.detach().numpy()[0, 0, :, :]
phi_subdomain2 = output_subdomain2.detach().numpy()[0, 0, :, :]
combined_output_mask = (X - interface_center[0])**2 + (Y - interface_center[1])**2 <= interface_radius**2
combined_output = np.zeros_like(phi_subdomain1)
combined_output[~combined_output_mask] = phi_subdomain1[~combined_output_mask]
combined_output[combined_output_mask] = phi_subdomain2[combined_output_mask]

# Apply boundary condition in the interface


output_array = combined_output.detach().numpy()[0, 0, :, :] 


# Plot results
fig, axs = plt.subplots(1, 2, figsize=(10, 5)) 
img_output = axs[1].imshow(output_array, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[1].set_title('Output')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
cbar_output = plt.colorbar(img_output, ax=axs[1], label='Magnitude')
plt.tight_layout()
plt.show()

