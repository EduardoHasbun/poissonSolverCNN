import numpy as np
import os
from multiprocessing import Pool, cpu_count
import yaml
from scipy.interpolate import RegularGridInterpolator as rgi
import argparse
from tqdm import tqdm as log_progress
import matplotlib.pyplot as plt

# Specific arguments
parser = argparse.ArgumentParser(description='RHS random dataset')
parser.add_argument('-c', '--cfg', type=str, default=None, help='Config filename')
args = parser.parse_args()
with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)
nits = cfg['n_it']
plotting = False
n_res_factor = cfg['n_res_factor']

def generate_random(i):
    # Parameters for data generation
    xmin, xmax, nnx = cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['nnx']
    ymin, ymax, nny = cfg['domain']['ymin'], cfg['domain']['ymax'], cfg['domain']['nny']
    zmin, zmax, nnz = cfg['domain']['zmin'], cfg['domain']['zmax'], cfg['domain']['nnz']
    n_res_factor = cfg['n_res_factor']

    # Create a 1D grid for each axis
    x, y, z = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny), np.linspace(zmin, zmax, nnz)

    # Factor to divide the grid by to generate the random grid
    nnx_lower, nny_lower, nnz_lower = int(nnx / n_res_factor), int(nny / n_res_factor), int(nnz / n_res_factor)
    x_lower, y_lower, z_lower = np.linspace(xmin, xmax, nnx_lower), np.linspace(ymin, ymax, nny_lower), np.linspace(zmin, zmax, nnz_lower)

    # Create a single array for the points
    points = np.array(np.meshgrid(x, y, z, indexing='ij')).T.reshape(-1, 3)

    data_lower = 2 * np.random.random((nnx_lower, nny_lower, nnz_lower)) - 1
    f = rgi((x_lower, y_lower, z_lower), data_lower, method='cubic')
    return f(points).reshape((nnx, nny, nnz))

if __name__ == '__main__':
    pool = Pool(processes=cpu_count())

    plots_dir = os.path.join('generated', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    print('nnx: ', cfg['domain']['nnx'], 'nny: ', cfg['domain']['nny'], "nnz: ", cfg['domain']['nnz'])

    # Generate random data samples
    random_data_array = np.empty((nits, cfg['domain']['nnx'], cfg['domain']['nny'], cfg['domain']['nnz']))
    for idx, random_data in log_progress(enumerate(pool.imap(generate_random, range(nits))), total=nits, desc="Processing"):
        random_data_array[idx] = random_data

    file_path = os.path.join('generated', 'random.npy')
    print(np.shape(random_data_array))
    os.makedirs('generated', exist_ok=True)
    np.save(file_path, random_data_array)
