import os
import sys
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score  

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

def poisson_punctual_solution(x, y, z, charges):
    solution = np.zeros_like(x)
    eps = 1e-8  # Precaución contra división por cero
    for A, x0, y0, z0 in charges:
        distance = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
        distance = np.where(distance < eps, eps, distance)
        solution += A / (4 * np.pi * distance)
    return solution



# Extraer parámetros de cargas puntuales
ngauss = int(len(params) / 7)
params_arr = np.array(params).reshape(ngauss, 7)
punctual_charges = [(A, x0, y0, z0) for A, x0, y0, z0, _, _, _ in params_arr]

# Solución analítica
resolution_data = poisson_punctual_solution(X, Y, Z, punctual_charges)

# Cálculo del ratio de escalamiento
def ratio_potrhs(alpha, Lx, Ly, Lz):
    return alpha / (np.pi**2 / 8)**2 / (1 / Lx**2 + 1 / Ly**2 + 1 / Lz**2)

alpha = 0.13
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
relative_error = np.abs(output_array - resolution_data) / (resolution_data) * 100
max_error = np.max(relative_error)
avg_error = np.average(relative_error)

# R2
def calculate_r2(y_true, y_pred):
    return r2_score(y_true.flatten(), y_pred.flatten())

r2_value = calculate_r2(resolution_data, output_array)

print(f'Max Error: {max_error:.2f}%, Avg Error: {avg_error:.2f}%, R² Variance: {r2_value:.4f}')

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

log_case_error(case_name, max_error, avg_error, r2_value)
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
    img1 = axs[1, i].imshow(data_fn(resolution_data).T, origin='lower', cmap='viridis',
                            vmin=vmin, vmax=vmax, aspect='auto')
    axs[1, i].set_title(f'Analytical - {title}')
    plt.colorbar(img1, ax=axs[1, i])

    # Relative Error
    img2 = axs[2, i].imshow(data_fn(relative_error).T, origin='lower', cmap='viridis',
                            aspect='auto')
    axs[2, i].set_title(f'Error Relativo (%) - {title}')
    plt.colorbar(img2, ax=axs[2, i])

# Etiquetas de ejes
for row in range(3):
    axs[row, 0].set_ylabel(['NN Output', 'Analytical', 'Error Relativo (%)'][row])

for col in range(3):
    axs[2, col].set_xlabel(['X vs Y', 'X vs Z', 'Y vs Z'][col])

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f'{case_name}.png'))
