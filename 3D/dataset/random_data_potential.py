import numpy as np
import os
from multiprocessing import get_context
import yaml
from scipy.interpolate import RegularGridInterpolator as rgi
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm as log_progress
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
# Specific arguments
parser = argparse.ArgumentParser(description='RHS random dataset')
parser.add_argument('-c', '--cfg', type=str, default=None, help='Config filename')
args = parser.parse_args()
with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)
nits = cfg['n_it']
plotting = False

if __name__ == '__main__':
    # Parameters for data generation
    xmin, xmax, nnx = cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['nnx']
    ymin, ymax, nny = cfg['domain']['ymin'], cfg['domain']['ymax'], cfg['domain']['nny']
    zmin, zmax, nnz = cfg['domain']['zmin'], cfg['domain']['zmax'], cfg['domain']['nnz']
    n_res_factor = 10

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

    print('nnx: ', nnx, 'nny: ', nny, "nnz: ", nnz)

    # Generate random data samples
    random_data_array = np.empty((nits, nnx, nny, nnz))
    potential_array = np.empty((nits, nnx, nny, nnz))

    for idx, random_data in log_progress(enumerate(generate_random(nits)), total=nits, desc="Processing"):
        random_data_array[idx] = random_data * 1.5e3

        # Solve Poisson equation: Laplacian(potential) = random_data

        laplacian = np.zeros_like(random_data)
        laplacian[1:-1, 1:-1, 1:-1] = (
            random_data[:-2, 1:-1, 1:-1] + random_data[2:, 1:-1, 1:-1] +
            random_data[1:-1, :-2, 1:-1] + random_data[1:-1, 2:, 1:-1] +
            random_data[1:-1, 1:-1, :-2] + random_data[1:-1, 1:-1, 2:]
        ) / 6.0

        # Create a sparse matrix for the Laplacian
        diag = np.ones(laplacian.size)
        offsets = [-nnx * nny, -nnx, -1, 0, 1, nnx, nnx * nny]
        laplacian_matrix = diags([diag, diag, diag, diag, diag, diag, diag], offsets, shape=(nnx * nny * nnz, nnx * nny * nnz))

        laplacian_reshape = laplacian[1:-1, 1:-1, 1:-1].ravel()

        # Solve for potential using a sparse solver
        potential_reshape = spsolve(laplacian_matrix, laplacian_reshape)
        potential = potential_reshape.reshape((nnx-2, nny-2, nnz-2))

        potential_array[idx] = potential



    file_path_random = os.path.join('generated', 'random_data2.npy')
    file_path_potential = os.path.join('generated', 'potential_data.npy')
    print(np.shape(random_data_array))
    os.makedirs('generated', exist_ok=True)
    np.save(file_path_random, random_data_array)
    np.save(file_path_potential, potential_array)