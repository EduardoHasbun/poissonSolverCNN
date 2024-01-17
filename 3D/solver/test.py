import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt

with open('C:\Codigos/poissonSolverCNN/3D/solver/solver.yml', 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

xmin, xmax, nnx = cfg['mesh']['xmin'], cfg['mesh']['xmax'], cfg['mesh']['nnx']
ymin, ymax, nny = cfg['mesh']['ymin'], cfg['mesh']['ymax'], cfg['mesh']['nny']
zmin, zmax, nnz = cfg['mesh']['zmin'], cfg['mesh']['zmax'], cfg['mesh']['nnz']
x_1d = np.linspace(xmin, xmax, nnx)
y_1d = np.linspace(ymin, ymax, nny)
z_1d = np.linspace(zmin, zmax, nnz)
X, Y, Z = np.meshgrid(x_1d, y_1d, z_1d)

# Define Gaussians's Functions
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

# Create input data without unnecessary reshaping
input_data = gaussians(X, Y, Z, cfg['init']['args'])
field = torch.from_numpy(input_data).float()

print(np.shape(field))

# Visualize slices in different dimensions
plt.figure()
plt.imshow(field[:, :, nnz // 2], origin='lower', extent=[xmin, xmax, ymin, ymax])
plt.title('XY Slice')
plt.colorbar()

plt.figure()
plt.imshow(field[:, nny // 2, :], origin='lower', extent=[xmin, xmax, zmin, zmax])
plt.title('XZ Slice')
plt.colorbar()

plt.figure()
plt.imshow(field[nnx // 2, :, :], origin='lower', extent=[ymin, ymax, zmin, zmax])
plt.title('YZ Slice')
plt.colorbar()

plt.show()
