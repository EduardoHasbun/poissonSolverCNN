import torch
import torch.nn as nn
import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy import special as sp
import os
import sys
sys.path.append('C:/Codigos/poissonSolverCNN/3D/training')
import operators3d as op
from models import UNetInterface

# Load configuration
with open('C:/Codigos/poissonSolverCNN/3D/solver/solver_interface.yml', 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
plots_dir = os.path.join('results')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Get paths and configurations from YAML
case_name = cfg['general']['case_name']
model_dir = cfg['general']['model_dir']
network_type = cfg['arch']['type']
arch_type = cfg['arch']['type']
arch_dir = os.path.join('..', '..', cfg['arch']['arch_dir'])

# Load architecture config
with open(arch_dir) as yaml_stream1:
    arch = yaml.safe_load(yaml_stream1)
arch_model = arch[arch_type]['type']
# Get scales and kernel sizes
scales_data = arch.get(arch_type, {}).get('args', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
kernel_size = arch[arch_type]['args']['kernel_sizes']

# Mesh and Grid
xmin, xmax, nnx = cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['nnx']
ymin, ymax, nny = cfg['domain']['ymin'], cfg['domain']['ymax'], cfg['domain']['nny']
zmin, zmax, nnz = cfg['domain']['zmin'], cfg['domain']['zmax'], cfg['domain']['nnz']
interface_center = (cfg['domain']['interface_center']['x'], cfg['domain']['interface_center']['y'], cfg['domain']['interface_center']['z'])
R = cfg['domain']['R']
x, y, z = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny), np.linspace(zmin, zmax, nnz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
Lx, Ly, Lz = (xmax - xmin), (ymax - ymin), (zmax - zmin)
points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
interface_center = [0, 0, 0]
interface_mask = (X - interface_center[0]) ** 2 + (Y - interface_center[1]) ** 2 + (Z - interface_center[2]) ** 2 <= R ** 2
labels = np.where(interface_mask, "molecule", "solvent")
interface_mask = (X - interface_center[0])**2 + (Y - interface_center[1])**2 + (Z - interface_center[2])**2 <= R**2
interface_boundary = np.zeros_like(interface_mask, dtype=bool)
for i in range(1, interface_mask.shape[0] - 1):
    for j in range(1, interface_mask.shape[1] - 1):
        for k in range (1, interface_mask.shape[2] - 1):
            # Check for boundary change and mark only the outside node
            if interface_mask[i, j, k] != interface_mask[i - 1, j, k] or interface_mask[i, j, k] != interface_mask[i + 1, j, k]:
                if interface_mask[i, j, k]:  # If current node is outside the interface
                    interface_boundary[i, j, k] = True
            elif interface_mask[i, j, k] != interface_mask[i, j - 1, k] or interface_mask[i, j, k] != interface_mask[i, j + 1, k]:
                if interface_mask[i, j, k]:  # If current node is outside the interface
                    interface_boundary[i, j, k] = True
            elif interface_mask[i, j, k] != interface_mask[i, j, k - 1] or interface_mask[i, j, k] != interface_mask[i, j, k+1]:
                if interface_mask[i, j, k]:
                    interface_boundary[i, j, k] = True

inner_mask = interface_mask
outer_mask = ~interface_mask | interface_boundary


# Charges
charges = cfg['init']['charges']
locations = np.array(cfg['init']['locations'])
E_1 = cfg['spherical_harmonics']['E_1']
E_2 = cfg['spherical_harmonics']['E_2']
kappa = cfg['spherical_harmonics']['kappa']
N = cfg['spherical_harmonics']['N']

# Functions
def G(X, q, xq, epsilon):
    r_vec_expanded = np.expand_dims(X, axis=1)
    x_qs_expanded = np.expand_dims(xq, axis=0)
    r_diff = r_vec_expanded - x_qs_expanded
    r = np.sqrt(np.sum(np.square(r_diff), axis=2))
    q_over_r = q / r
    total_sum = np.sum(q_over_r, axis=1)
    result = (1 / (epsilon * 4 * np.pi)) * total_sum
    result = np.expand_dims(result, axis=1)
    return result

def Spherical_Harmonics(x, y, z, q, xq, E_1, E_2, kappa, R, labels, points, N):
    PHI = np.zeros(len(points))
    for K in range(len(points)):
        px, py, pz = points[K]
        ix = int((px - x[0]) / (x[1] - x[0]))
        iy = int((py - y[0]) / (y[1] - y[0]))
        iz = int((pz - z[0]) / (z[1] - z[0]))
        rho = np.sqrt(np.sum(points[K,:] ** 2))
        zenit = np.arccos(points[K, 2] / rho)
        azim = np.arctan2(points[K, 1], points[K, 0])
        phi = 0.0 + 0.0 * 1j
        for n in range(N):
            for m in range(-n, n + 1):
                Enm = 0.0
                for k in range(len(q)):
                    rho_k = np.sqrt(np.sum(xq[k,:] ** 2))
                    zenit_k = np.arccos(xq[k, 2] / rho_k)
                    azim_k = np.arctan2(xq[k, 1], xq[k, 0])
                    Enm += (
                        q[k]
                        * rho_k**n
                        *4*np.pi/(2*n+1)
                        * sp.sph_harm(m, n, -azim_k, zenit_k)
                    )
                Anm = Enm * (1/(4*np.pi)) * ((2*n+1)) / (np.exp(-kappa*R)* ((E_1-E_2)*n*get_K(kappa*R,n)+E_2*(2*n+1)*get_K(kappa*R,n+1)))
                Bnm = 1/(R**(2*n+1))*(np.exp(-kappa*R)*get_K(kappa*R,n)*Anm - 1/(4*np.pi*E_1)*Enm)
                if labels[ix, iy, iz]=='molecule':
                    phi += Bnm * rho**n * sp.sph_harm(m, n, azim, zenit)
                if labels[ix, iy, iz]=='solvent':
                    phi += Anm * rho**(-n-1)* np.exp(-kappa*rho) * get_K(kappa*rho,n) * sp.sph_harm(m, n, azim, zenit)
        if labels[ix, iy, iz] == "solvent":
            phi -= G(np.array([points[K]]), q, xq, E_1)
        PHI[K] = np.real(phi)
    return PHI

def get_K(x, n):
    K = 0.0
    n_fact = sp.factorial(n)
    n_fact2 = sp.factorial(2 * n)
    for s in range(n + 1):
        K += (
            2**s
            * n_fact
            * sp.factorial(2 * n - s)
            / (sp.factorial(s) * n_fact2 * sp.factorial(n - s))
            * x**s
        )
    return K

# Define Ratio
def ratio_potrhs(alpha, Lx, Ly, Lz):
    return alpha / (np.pi**2 / 8)**2 / (1 / Lx**2 + 1 / Ly**2 + 1 / Lz**2)

# Create Model
model = UNetInterface(scales, kernel_sizes=kernel_size, input_res=nnx, inner_mask = inner_mask, outer_mask = outer_mask)
model.load_state_dict(torch.load(model_dir))
model = model.float()
for param in model.parameters():
    param.data = param.data.float()
model.eval() 
alpha = 0.1
ratio = ratio_potrhs(alpha, Lx, Ly, Lz)


# Create input data and solution data
input_data = G(points, charges, locations, E_1)
input_data = input_data.reshape(nnx, nny, nnz)
input_data = input_data[np.newaxis, np.newaxis, :, :, :]
input_data = torch.from_numpy(input_data).float()
analitical_solution = Spherical_Harmonics(x, y, z, charges, locations, E_1, E_2, kappa, R, labels, points, N)
analitical_solution = analitical_solution.reshape(nnx, nny, nnz)

#Solver
out_in, out_out = model(input_data)
output = torch.zeros_like(out_in)
output[0, 0, interface_mask] = out_in[0, 0, interface_mask]
output[0, 0, ~interface_mask] = out_out[0, 0, ~interface_mask]
output_array = output.detach().numpy()[0, 0, :, :] * ratio

# Plots
output_array = output.detach().numpy()[0, 0, :, :, :] 
input_plot = input_data.detach().numpy()[0, 0, :, :, :]
input_slice = input_plot[:,:,nnz//2]
ouptut_slice = output_array[:,:,nnz//2]
relative_error_inner = np.abs(output_array - analitical_solution) / np.abs(np.max(analitical_solution)) * 100
fig, axs = plt.subplots(1, 3, figsize=(10, 5)) 
fig.suptitle('3D', fontsize=16) 

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
print(f'Error Maximo: {np.max(relative_error_inner)}, Error Promedio: {np.average(relative_error_inner)}')
axs[2].set_title('Relative Error')
axs[2].set_xlabel('X')
axs[2].set_ylabel('Y')
cbar_output = plt.colorbar(img_error, ax=axs[2], label='Error %')
plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig(os.path.join(plots_dir, case_name + '.png'))