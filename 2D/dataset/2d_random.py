import numpy as np
import os
from multiprocessing import Pool, cpu_count
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
plotting = False

def generate_random_data(i, x_lower, y_lower, x, y, nnx, nny):
    n_res_factor = 16
    nnx_lower = int(nnx / n_res_factor)
    nny_lower = int(nny / n_res_factor)
    
    z_lower = 2 * np.random.random((nny_lower, nnx_lower)) - 1
    f = rgi((x_lower, y_lower,), z_lower, method='cubic')
    return f(x, y)

if __name__ == '__main__':
    # Parameters for data generation
    xmin, xmax, nnx = cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['nnx']
    nny, ymin, ymax = cfg['domain']['nny'], cfg['domain']['ymin'], cfg['domain']['ymax']
    
    # Create a grid
    x, y = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny)

    # Factor to divide the grid by to generate the random grid
    nnx_lower = int(nnx / 16)
    nny_lower = int(nny / 16)
    x_lower, y_lower = np.linspace(xmin, xmax, nnx_lower), np.linspace(ymin, ymax, nny_lower)

    plots_dir = os.path.join('generated', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Generate random data samples using multiprocessing
    with Pool(processes=cpu_count()) as pool:
        results = list(pool.starmap(
            generate_random_data,
            [(i, x_lower, y_lower, x, y, nnx, nny) for i in range(nits)]
        ))

    # Convert results to array
    random_data_array = np.array(results)

    for idx, random_data in enumerate(random_data_array):
        if plotting and idx % 100 == 0:
            plt.figure(figsize=(8, 6))
            plt.imshow(random_data, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
            plt.colorbar(label='Random Data')
            plt.title(f'Random Data Sample {idx}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.savefig(os.path.join(plots_dir, f'random_data_plot_{idx}.png'))
            plt.close()

    file_path = os.path.join('generated', 'random_data.npy')
    os.makedirs('generated', exist_ok=True)
    np.save(file_path, random_data_array)
