import numpy as np
import os
from multiprocessing import Pool, cpu_count
import yaml
import matplotlib.pyplot as plt
import argparse
from scipy.interpolate import RegularGridInterpolator as rgi
from tqdm import tqdm as tqdm


# Specific arguments
parser = argparse.ArgumentParser(description='RHS random dataset')
parser.add_argument('-c', '--cfg', type=str, default=None, help='Config filename')
args = parser.parse_args()
with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)
nits = cfg['n_it']
ploting = False


if __name__ == '__main__':
    pool = Pool(processes=cpu_count())
    # Parameters for data generation
    xmin, xmax, nnx = cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['nnx']
    nny, ymin, ymax = cfg['domain']['nny'], cfg['domain']['ymin'], cfg['domain']['ymax']
    interface_center = (cfg['domain']['interface_center']['x'], cfg['domain']['interface_center']['y'])
    interface_radius = cfg['domain']['interface_radius']
    n_res_factor = 16

    # Create a grid
    x, y= np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny)
    X, Y = np.meshgrid(x,y)

    # Generate circular mask for interface
    interface_mask = (X - interface_center[0])**2 + (Y - interface_center[1])**2 <= interface_radius**2

    # Factor to divide the grid by to generate the random grid
    nnx_lower = int(nnx / n_res_factor)
    nny_lower = int(nny / n_res_factor)
    x_lower, y_lower = np.linspace(xmin, xmax, nnx_lower), np.linspace(ymin, ymax, nny_lower)
    points = np.array(np.meshgrid(x, y, indexing='ij')).T.reshape(-1, 2)

    def generate_random_data(nits):
        for i in range(nits):
            z_lower = 2 * np.random.random((nnx_lower, nny_lower)) - 1
            f = rgi((x_lower, y_lower,), z_lower, method='cubic')
            yield f(points).reshape((nnx, nny))

    plots_dir = os.path.join('generated', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Generate random data samples
    random_data_array = np.empty((nits, nnx, nny))
    inside_domain_array = np.empty((nits, nnx, nny))
    outside_domain_array = np.empty((nits, nnx, nny))
    for idx, random_data in tqdm(enumerate(pool.map(generate_random_data, range(nits))), total = nits):
        if idx == 0:
            inside_domain = np.zeros_like(random_data)
        else:
            inside_domain = random_data_old.copy()
        outside_domain = random_data.copy()

        inside_domain[~interface_mask] = 0
        outside_domain[interface_mask] = 0

        random_data_array[idx, interface_mask] = inside_domain[interface_mask]
        random_data_array[idx, ~interface_mask] = outside_domain[~interface_mask]
        inside_domain_array[idx] = inside_domain
        outside_domain_array[idx] = outside_domain

        random_data_old = random_data
        
        if ploting and idx%100==0:
            plt.figure(figsize=(8, 6))
            plt.imshow(random_data, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
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
    np.save(file_path_domain, random_data_array)
    np.save(file_path_inside, inside_domain_array)
    np.save(file_path_outside, outside_domain_array)
    

