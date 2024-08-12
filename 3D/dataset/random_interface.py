import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import argparse
from scipy.interpolate import RegularGridInterpolator as rgi
from tqdm import tqdm as log_progress
from multiprocessing import Pool, cpu_count


# Specific arguments
parser = argparse.ArgumentParser(description='RHS random dataset')
parser.add_argument('-c', '--cfg', type=str, default=None, help='Config filename')
args = parser.parse_args()
with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)
nits = cfg['n_it']
ploting = False

# Parameters for data generation
nnx, xmin, xmax = cfg['domain']['nnx'], cfg['domain']['xmin'], cfg['domain']['xmax']
nny, ymin, ymax = cfg['domain']['nny'], cfg['domain']['ymin'], cfg['domain']['ymax']
nnz, zmin, zmax = cfg['domain']['nnz'], cfg['domain']['zmin'], cfg['domain']['zmax']
interface_center = (cfg['domain']['interface_center']['x'], cfg['domain']['interface_center']['y'], cfg['domain']['interface_center']['z'])
interface_radius = cfg['domain']['interface_radius']
n_res_factor = 5


# Create a grid
x, y, z = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny), np.linspace(zmin, zmax, nnz)
X, Y, Z = np.meshgrid(x, y, z)

# Generate circular mask for interface
interface_mask = (X - interface_center[0])**2 + (Y - interface_center[1])**2 + (Z - interface_center[2]) <= interface_radius**2


def generate_random(i):
    # Parameters for data generation
    nnx, xmin, xmax = cfg['domain']['nnx'], cfg['domain']['xmin'], cfg['domain']['xmax']
    nny, ymin, ymax = cfg['domain']['nny'], cfg['domain']['ymin'], cfg['domain']['ymax']
    nnz, zmin, zmax = cfg['domain']['nnz'], cfg['domain']['zmin'], cfg['domain']['zmax']
    n_res_factor = 5

    # Create a grid
    x, y, z = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny), np.linspace(zmin, zmax, nnz)
    X, Y, Z = np.meshgrid(x, y, z)

    # Factor to divide the grid by to generate the random grid
    nnx_lower = int(nnx / n_res_factor)
    nny_lower = int(nny / n_res_factor)
    nnz_lower = int(nnz / n_res_factor)
    x_lower, y_lower, z_lower = np.linspace(xmin, xmax, nnx_lower), np.linspace(ymin, ymax, nny_lower), np.linspace(zmin, zmax, nnz_lower)
    
    # Create a single array for the points
    points = np.array(np.meshgrid(x, y, z, indexing='ij')).T.reshape(-1, 3)

    data_lower_inside = 2 * np.random.random((nnx_lower, nny_lower, nnz_lower)) - 1
    data_lower_outside = 2 * np.random.random((nnx_lower, nny_lower, nnz_lower)) - 1
    f_inside = rgi((x_lower, y_lower,), data_lower_inside, method='cubic')
    f_outside = rgi((x_lower, y_lower,), data_lower_outside, method='cubic')
    return f_inside(points).reshape((nnx, nny, nnz)), f_outside(points).reshape((nnx, nny, nnz))


if __name__ == '__main__':
    pool = Pool(processes=cpu_count())

    plots_dir = os.path.join('generated', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    print('nnx: ', cfg['domain']['nnx'], 'nny: ', cfg['domain']['nny'], 'nnz: ', cfg['domain']['nnz'],
           'xmax: ', cfg['domain']['xmax'], 'ymax: ', cfg['domain']['ymax'], 'zmax', cfg['domain']['zmax'])

    # Generate random data samples
    data_array = np.empty((nits, cfg['domain']['nnx'], cfg['domain']['nny'], cfg['domain']['nnz']))
    inside_domain_array = np.empty((nits, cfg['domain']['nnx'], cfg['domain']['nny'], cfg['domain']['nnz']))
    outside_domain_array = np.empty((nits, cfg['domain']['nnx'], cfg['domain']['nny'], cfg['domain']['nnz']))
    for idx, (inside_domain, outside_domain) in log_progress(enumerate(pool.imap(generate_random, range(nits))), total=nits, desc="Processing"):

        inside_domain[~interface_mask] = 0
        outside_domain[interface_mask] = 0

        data_array[idx, interface_mask] = inside_domain[interface_mask] 
        data_array[idx, ~interface_mask] = outside_domain[~interface_mask] 
        # inside_domain_array[idx] = inside_domain 
        # outside_domain_array[idx] = outside_domain 

        plt.imshow(data_array[idx])
        plt.show()
        
        if ploting and idx%100==0:
            plt.figure(figsize=(8, 6))
            plt.imshow(data_array[idx], extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
            plt.colorbar(label='Random Data')
            plt.title(f'Random Data Sample {idx}')
            plt.xlabel('X')
            plt.ylabel('Y') 
            plt.savefig(os.path.join(plots_dir, f'random_data_plot_{idx}.png'))
            plt.close()

    file_path_domain = os.path.join('generated', 'domain.npy')
    file_path_inside = os.path.join('generated', 'inside.npy')
    file_path_outside = os.path.join('generated', 'outside.npy')
    os.makedirs('generated', exist_ok=True)
    np.save(file_path_domain, data_array)
    np.save(file_path_inside, inside_domain_array)
    np.save(file_path_outside, outside_domain_array)