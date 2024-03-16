import numpy as np
import os
import argparse
import yaml
from multiprocessing import get_context
from scipy.interpolate import RegularGridInterpolator as rgi
import matplotlib.pyplot as plt
from tqdm import tqdm as log_progress

parser = argparse.ArgumentParser(description='Point Charge Dataset')
parser.add_argument('-c', '--cfg', type=str, default=None, help='Config filename')
args = parser.parse_args()

with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)

n_fields = cfg['n_fields']
nits = cfg['n_it']
plotting = cfg['plotting']

if __name__ == '__main__':
    xmin, xmax, nnx = cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['nnx']
    ymin, ymax, nny = cfg['domain']['ymin'], cfg['domain']['ymax'], cfg['domain']['nny']
    zmin, zmax, nnz = cfg['domain']['zmin'], cfg['domain']['zmax'], cfg['domain']['nnz']
    n_res_factor = 10

    x, y, z = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny), np.linspace(zmin, zmax, nnz)

    nnx_lower, nny_lower, nnz_lower = int(nnx / n_res_factor), int(nny / n_res_factor), int(nnz / n_res_factor)
    x_lower, y_lower, z_lower = np.linspace(xmin, xmax, nnx_lower), np.linspace(ymin, ymax, nny_lower), np.linspace(zmin, zmax, nnz_lower)

    points = np.array(np.meshgrid(x, y, z, indexing='ij')).T.reshape(-1, 3)

    def generate_field(nits):
        for i in range(nits):
            data_lower = np.zeros((nnx_lower, nny_lower, nnz_lower))
            # Creating a Gaussian charge in the center
            center_idx = (nnx_lower // 2, nny_lower // 2, nnz_lower // 2)
            sigma = 5.0e-3  # Adjust sigma as needed
            gauss = lambda x, y, z: np.exp(-((x - center_idx[0]) ** 2 + (y - center_idx[1]) ** 2 + (z - center_idx[2]) ** 2) / (2 * sigma ** 2))
            for xi in range(nnx_lower):
                for yi in range(nny_lower):
                    for zi in range(nnz_lower):
                        data_lower[xi, yi, zi] = gauss(xi, yi, zi)
            f = rgi((x_lower, y_lower, z_lower), data_lower, method='cubic')
            yield f(points).reshape((nnx, nny, nnz))

    # Generate fields
    fields = np.empty((n_fields, nnx, nny, nnz))
    for idx, field in log_progress(enumerate(generate_field(n_fields)), total=n_fields, desc="Generating Fields"):
        fields[idx] = field

        if plotting and idx % 10 == 0:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
            ax.scatter(x_grid, y_grid, z_grid, c=fields[idx].ravel(), cmap='viridis')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Field Sample {idx}')
            plt.show()

    # Poisson equation solution (1/r)
    poisson_solution = 1 / np.linalg.norm(points - np.array([0, 0, 0]), axis=1).reshape((nnx, nny, nnz))

    # Save generated data
    plots_dir = os.path.join('generated', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    file_path_fields = os.path.join('generated', 'fields.npy')
    file_path_solution = os.path.join('generated', 'poisson_solution.npy')
    print(np.shape(fields))
    print(np.shape(poisson_solution))
    os.makedirs('generated', exist_ok=True)
    np.save(file_path_fields, fields)
    np.save(file_path_solution, poisson_solution)
