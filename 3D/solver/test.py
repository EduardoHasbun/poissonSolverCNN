import torch
import torch.nn as nn
import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy import special as sp
from sklearn.metrics import r2_score
import os
import sys
sys.path.append('C:/Codigos/poissonSolverCNN/3D/training')
from models import UNet3D as UNet
from models import MSNet3D as MSNet

# Cargar configuración
with open('C:/Codigos/poissonSolverCNN/3D/solver/solver.yml', 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

# Parámetros del caso
case_name = cfg['general']['case_name']
model_dir = cfg['general']['model_dir']
arch_type = cfg['arch']['type']
arch_dir = os.path.join('..', '..', cfg['arch']['arch_dir'])

# Cargar arquitectura
with open(arch_dir) as arch_file:
    arch = yaml.safe_load(arch_file)
arch_model = arch[arch_type]['type']
scales_data = arch.get(arch_type, {}).get('args', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
kernel_size = arch[arch_type]['args']['kernel_sizes']
params = cfg['init']['args']

# Malla 3D
xmin, xmax, nnx = cfg['mesh']['xmin'], cfg['mesh']['xmax'], cfg['mesh']['nnx']
ymin, ymax, nny = cfg['mesh']['ymin'], cfg['mesh']['ymax'], cfg['mesh']['nny']
zmin, zmax, nnz = cfg['mesh']['zmin'], cfg['mesh']['zmax'], cfg['mesh']['nnz']
Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin
x_1d = np.linspace(xmin, xmax, nnx)
y_1d = np.linspace(ymin, ymax, nny)
z_1d = np.linspace(zmin, zmax, nnz)
Z, Y, X = np.meshgrid(z_1d, y_1d, x_1d, indexing='ij')


# Crear directorios
plots_dir = 'results'
errors_file = os.path.join(plots_dir, 'errors_log.txt')
os.makedirs(plots_dir, exist_ok=True)

# Definir funciones gaussianas
def gaussian(x, y, z, A, x0, y0, z0, sigx, sigy, sigz):
    return A * np.exp(-((x - x0) / sigx)**2 - ((y - y0) / sigy)**2 - ((z - z0) / sigz)**2)

def gaussians(x, y, z, params):
    profile = np.zeros_like(x)
    ngauss = int(len(params) / 7)
    params = np.array(params).reshape(ngauss, 7)
    for p in params:
        profile += gaussian(x, y, z, *p)
    return profile

# Functions
def G(X, q, xq, epsilon):
    r_vec_expanded = np.expand_dims(X, axis=1)  # Shape: (n, 1, 3)
    x_qs_expanded = np.expand_dims(xq, axis=0)  # Shape: (1, m, 3)
    r_diff = r_vec_expanded - x_qs_expanded     # Shape: (n, m, 3)
    r = np.sqrt(np.sum(np.square(r_diff), axis=2))  # Shape: (n, m)
    q_over_r = q / r  # Shape: (n, m)
    total_sum = np.sum(q_over_r, axis=1)  # Shape: (n,)
    result = (1 / (epsilon * 4 * np.pi)) * total_sum  # Shape: (n,)
    return result


def Spherical_Harmonics(x, y, z, q, xq, E_1, E_2, kappa, R, labels, points, N):
    # Precompute values for all points
    points = np.array(points)
    rho = np.linalg.norm(points, axis=1)
    zenit = np.arccos(points[:, 2] / rho)
    azim = np.arctan2(points[:, 1], points[:, 0])

    xq = np.array(xq)
    rho_k = np.linalg.norm(xq, axis=1)
    zenit_k = np.arccos(xq[:, 2] / rho_k)
    azim_k = np.arctan2(xq[:, 1], xq[:, 0])
    
    # Precompute the grid indices for labels
    ix = ((points[:, 0] - x[0]) / (x[1] - x[0])).astype(int)
    iy = ((points[:, 1] - y[0]) / (y[1] - y[0])).astype(int)
    iz = ((points[:, 2] - z[0]) / (z[1] - z[0])).astype(int)

    PHI = np.zeros(len(points), dtype=np.complex128)

    # Loop over n and m
    for n in range(N):
        for m in range(-n, n + 1):
            # Compute Enm for all points
            Enm = np.sum(
                q[:, None]
                * rho_k[:, None]**n
                * (4 * np.pi / (2 * n + 1))
                * sp.sph_harm(m, n, -azim_k[:, None], zenit_k[:, None]),
                axis=0
            )
            Anm = Enm * (1 / (4 * np.pi)) * ((2 * n + 1)) / (
                np.exp(-kappa * R) * ((E_1 - E_2) * n * get_K(kappa * R, n) + E_2 * (2 * n + 1) * get_K(kappa * R, n + 1))
            )
            Bnm = 1 / (R ** (2 * n + 1)) * (
                np.exp(-kappa * R) * get_K(kappa * R, n) * Anm - 1 / (4 * np.pi * E_1) * Enm
            )

            # Compute phi based on labels
            is_molecule = labels[ix, iy, iz] == "molecule"
            is_solvent = labels[ix, iy, iz] == "solvent"

            PHI[is_molecule] += (
                Bnm * rho[is_molecule]**n * sp.sph_harm(m, n, azim[is_molecule], zenit[is_molecule])
            )
            PHI[is_solvent] += (
                Anm
                * rho[is_solvent] ** (-n - 1)
                * np.exp(-kappa * rho[is_solvent])
                * get_K(kappa * rho[is_solvent], n)
                * sp.sph_harm(m, n, azim[is_solvent], zenit[is_solvent])
            )

    # Final adjustment for solvent
    is_solvent = labels[ix, iy, iz] == "solvent"
    # PHI[is_solvent] -= G(points[is_solvent], q, xq, E_1)
    PHI += G(points, q, xq, E_1)

    return np.real(PHI)

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


R = 4.0
E_1 = 1
E_2 = 80
kappa= 0
N= 10
charges = np.array([1.0])
locations = np.array([[0.1, 0.1, 0.0]])



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








# Extraer parámetros de cargas puntuales
ngauss = int(len(params) / 7)
params_arr = np.array(params).reshape(ngauss, 7)
punctual_charges = [(A, x0, y0, z0) for A, x0, y0, z0, _, _, _ in params_arr]

# Solución analítica
x, y, z = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny), np.linspace(zmin, zmax, nnz)
analitical_solution = Spherical_Harmonics(x, y, z, charges, locations, E_1, E_2, kappa, R, labels, points, N)
analitical_solution = analitical_solution.reshape(nnx, nny, nnz)
mid_x, mid_y, mid_z = nnx // 2, nny // 2, nnz // 2
neighbors = [
        analitical_solution[mid_x + 1, mid_y, mid_z],
        analitical_solution[mid_x - 1, mid_y, mid_z],
        analitical_solution[mid_x, mid_y + 1, mid_z],
        analitical_solution[mid_x, mid_y - 1, mid_z],
        analitical_solution[mid_x, mid_y, mid_z + 1],
        analitical_solution[mid_x, mid_y, mid_z - 1]
    ]
analitical_solution[mid_x, mid_y, mid_z] = np.mean(neighbors)

# Cálculo del ratio de escalamiento
def ratio_potrhs(alpha, Lx, Ly, Lz):
    return alpha / (np.pi**2 / 8)**2 / (1 / Lx**2 + 1 / Ly**2 + 1 / Lz**2)

alpha = 0.03
ratio_max = ratio_potrhs(alpha, Lx, Ly, Lz)

# Generar entrada
input_data = gaussians(X, Y, Z, params) * ratio_max
input_data = input_data[np.newaxis, np.newaxis, :, :, :]
input_data = torch.from_numpy(input_data).float()

# Crear modelo
if arch_model == 'UNet':
    model = UNet(scales=scales, kernel_sizes=kernel_size, input_res=nnx)
elif arch_model == 'MSNet':
    model = MSNet(scales=scales, kernel_sizes=kernel_size, input_res=nnx)
model.load_state_dict(torch.load(model_dir))
model = model.float()
model.eval()

# Ejecutar red
output = model(input_data)
output_array = output.detach().numpy()[0, 0, :, :, :] * ratio_max

# Cálculo de errores
eps = 1e-8
relative_error = np.abs(output_array - analitical_solution) / (analitical_solution) * 100
max_error = np.max(relative_error)
avg_error = np.average(relative_error)

# R2
def calculate_r2(y_true, y_pred):
    return r2_score(y_true.flatten(), y_pred.flatten())

# r2_value = calculate_r2(analitical_solution, output_array)

# print(f'Max Error: {max_error:.2f}%, Avg Error: {avg_error:.2f}%, R² Variance: {r2_value:.4f}')

# Guardar errores
def log_case_error(case_name, max_error, avg_error, r2_value):
    if not os.path.exists(errors_file):
        with open(errors_file, 'w') as f:
            f.write("Case Name, Max Error (%), Avg Error (%), R^2 Variance\n")

    with open(errors_file, 'r') as f:
        lines = f.readlines()

    if not any(case_name in line for line in lines):
        with open(errors_file, 'a') as f:
            f.write(f"{case_name}, {max_error:.2f}, {avg_error:.2f}, {r2_value:.4f}\n")

log_case_error(case_name, max_error, avg_error, 1.0)
vmin, vmax = np.min(output_array), np.max(output_array)

# Visualización: Corte en el centro del dominio Z
mid_z = nnz // 2
fig, axs = plt.subplots(3, 3, figsize=(15, 13))


vmin, vmax = np.min(output_array), np.max(output_array)

# Output

from matplotlib.ticker import FuncFormatter


mid_x = output_array.shape[0] // 2
mid_y = output_array.shape[1] // 2
mid_z = output_array.shape[2] // 2

titles = ['Z Slice (XY)', 'Y Slice (XZ)', 'X Slice (YZ)']

for i, (title, data_fn) in enumerate(zip(
    titles,
    [
        lambda arr: arr[:, :, mid_z],  # Z slice → plano XY
        lambda arr: arr[:, mid_y, :],  # Y slice → plano XZ
        lambda arr: arr[mid_x, :, :],  # X slice → plano YZ
    ]
)):
    # NN Output
    img0 = axs[0, i].imshow(data_fn(output_array).T, origin='lower', cmap='viridis',
                            vmin=vmin, vmax=vmax, aspect='auto')
    axs[0, i].set_title(f'NN Output - {title}')
    plt.colorbar(img0, ax=axs[0, i])

    # Analytical Solution
    img1 = axs[1, i].imshow(data_fn(analitical_solution).T, origin='lower', cmap='viridis')
    axs[1, i].set_title(f'Analytical - {title}')
    plt.colorbar(img1, ax=axs[1, i])

    # Relative Error
    img2 = axs[2, i].imshow(
        data_fn(relative_error).T,
        origin='lower',
        cmap='viridis',
        aspect='auto',
        vmin=0,       # Opcional, si quieres que el mínimo sea 0
        vmax=10       # Aquí defines el máximo de la barra de color
    )
    axs[2, i].set_title(f'Error Relativo (%) - {title}')
    plt.colorbar(img2, ax=axs[2, i])

# Etiquetas de ejes
for row in range(3):
    axs[row, 0].set_ylabel(['NN Output', 'Analytical', 'Error Relativo (%)'][row])

for col in range(3):
    axs[2, col].set_xlabel(['X vs Y', 'X vs Z', 'Y vs Z'][col])

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f'{case_name}.png'))
