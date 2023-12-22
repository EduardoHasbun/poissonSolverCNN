import numpy as np
import os
from multiprocessing import get_context
import yaml
from scipy.interpolate import RegularGridInterpolator as rgi
import matplotlib.pyplot as plt
import argparse

# Specific arguments
parser = argparse.ArgumentParser(description='RHS random dataset')
parser.add_argument('-c', '--cfg', type=str, default=None, help='Config filename')
args = parser.parse_args()
with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)
nits = cfg['n_it']
plotting = True

if __name__ == '__main__':
    # Parameters for data generation
    xmin, xmax, nnx = cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['nnx']
    ymin, ymax, nny = cfg['domain']['ymin'], cfg['domain']['ymax'], cfg['domain']['nny']
    zmin, zmax, nnz = cfg['domain']['zmin'], cfg['domain']['zmax'], cfg['domain']['nnz']
    n_res_factor = 16

    # Create a 1D grid for each axis
    x, y, z = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny), np.linspace(zmin, zmax, nnz)

    # Factor to divide the grid by to generate the random grid
    nnx_lower, nny_lower, nnz_lower = int(nnx / n_res_factor), int(nny / n_res_factor), int(nnz / n_res_factor)
    x_lower, y_lower, z_lower = np.linspace(xmin, xmax, nnx_lower), np.linspace(ymin, ymax, nny_lower), np.linspace(zmin, zmax, nnz_lower)

    # Create a single array for the points
    points = np.array(np.meshgrid(x, y, z, indexing='ij')).T.reshape(-1, 3)

    def generate_random(nits):
        for i in range(nits):
            data_lower = 2 * np.random.random((nnx_lower, nny_lower, nnz_lower)) - 1
            f = rgi((x_lower, y_lower, z_lower), data_lower, method='cubic')
            yield f(points).reshape((nnx, nny, nnz))

    plots_dir = os.path.join('generated', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Generate random data samples
    random_data_array = np.empty((nits, nnx, nny, nnz))
    for idx, random_data in enumerate(generate_random(nits)):
        random_data_array[idx] = random_data * 1.5e3

        if plotting and idx % 10 == 0:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
            ax.scatter(x_grid, y_grid, z_grid, c=random_data_array[idx].ravel(), cmap='viridis')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Random Data Sample {idx}')
            plt.savefig(os.path.join(plots_dir, f'random_data_plot_{idx}.png'))
            plt.close()


    file_path = os.path.join('generated', 'random_data.npy')
    print(np.shape(random_data_array))
    os.makedirs('generated', exist_ok=True)
    np.save(file_path, random_data_array)
