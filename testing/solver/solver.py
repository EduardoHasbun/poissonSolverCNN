
import torch
import torch.nn as nn
import numpy as np
import yaml
import matplotlib.pyplot as plt
from unet import UNet


with open('C:/Codigos/poissonSolverCNN_Gpu/testing/solver\solver.yml', 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
scales_data = cfg.get('arch', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
kernel_size = cfg['arch']['kernel_sizes']

xmin, xmax, nnx = cfg['mesh']['xmin'], cfg['mesh']['xmax'], cfg['mesh']['nnx']
ymin, ymax, nny  = cfg['mesh']['ymin'], cfg['mesh']['ymax'], cfg['mesh']['nny']
x_1d = np.linspace(xmin, xmax, nnx)
y_1d = np.linspace(ymin, ymax, nny)
X, Y = np.meshgrid(x_1d, y_1d)


#Define Gaussians's Functions
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


# input_data = gaussians(X, Y, cfg['init']['args']).astype(np.float32)
input_data = gaussians(X, Y, cfg['init']['args'])
input_data = input_data[np.newaxis, np.newaxis, :, :]
input_data = torch.from_numpy(input_data).float()

#Create Model
model = model = UNet(scales, kernel_sizes=kernel_size, input_res=nnx)
model.load_state_dict(torch.load('C:/Codigos/poissonSolverCNN_Gpu/testing/train/trained_models/unet_model.pth'))
model = model.float()
for param in model.parameters():
    param.data = param.data.float()
model.eval() 

#Solver
output = model(input_data)
output_array = output.detach().numpy()[0, 0, :, :]



# Plots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

# Plot Input
img_input = axs[0].imshow(input_data.numpy()[0, 0, :, :], extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[0].set_title('Input')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
cbar_input = plt.colorbar(img_input, ax=axs[0], label='Magnitude')
cbar_input.formatter.set_scientific(True)
cbar_input.update_ticks()

# Plot Output
output_array = output.detach().numpy()[0, 0, :, :]
img_output = axs[1].imshow(output_array, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
axs[1].set_title('Output')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
cbar_output = plt.colorbar(img_output, ax=axs[1], label='Magnitude')
cbar_output.formatter.set_scientific(True)
cbar_output.update_ticks()


# Adjust layout to prevent clipping of titles
plt.tight_layout()

plt.show()