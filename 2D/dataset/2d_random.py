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
xmin, xmax, nnx = cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['nnx']
nny, ymin, ymax = cfg['domain']['nny'], cfg['domain']['ymin'], cfg['domain']['ymax']
e_in, e_out = cfg['domain']['epsilon_in'], cfg['domain']['epsilon_out']
interface_center = (cfg['domain']['interface_center']['x'], cfg['domain']['interface_center']['y'])
interface_radius = cfg['domain']['interface_radius']
n_res_factor = 20
# Create a grid

x, y= np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny)
X, Y = np.meshgrid(x,y)

def generate_random(i):
    # Parameters for data generation
    xmin, xmax, nnx = cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['nnx']
    nny, ymin, ymax = cfg['domain']['nny'], cfg['domain']['ymin'], cfg['domain']['ymax']
    n_res_factor = 20

    # Create a grid
    x, y= np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny)

    # Factor to divide the grid by to generate the random grid
    nnx_lower = int(nnx / n_res_factor)
    nny_lower = int(nny / n_res_factor)
    x_lower, y_lower = np.linspace(xmin, xmax, nnx_lower), np.linspace(ymin, ymax, nny_lower)
    points = np.array(np.meshgrid(x, y, indexing='ij')).T.reshape(-1, 2)

    z_lower = 2 * np.random.random((nnx_lower, nny_lower)) - 1
    f= rgi((x_lower, y_lower,), z_lower, method='cubic')
    return f(points).reshape((nnx, nny))


if __name__ == '__main__':
    pool = Pool(processes=cpu_count())

    plots_dir = os.path.join('generated', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    print('nnx: ', cfg['domain']['nnx'], 'nny: ', cfg['domain']['nny'], 'xmax: ', cfg['domain']['xmax'], 'ymax: ', cfg['domain'][ymax])

    # Generate random data samples
    data_array = np.empty((nits, cfg['domain']['nnx'], cfg['domain']['nny']))
    inside_domain_array = np.empty((nits, cfg['domain']['nnx'], cfg['domain']['nny']))
    outside_domain_array = np.empty((nits, cfg['domain']['nnx'], cfg['domain']['nny']))
    for idx, data in log_progress(enumerate(pool.imap(generate_random, range(nits))), total=nits, desc="Processing"):

        data_array[idx] = data
        if ploting and idx%100==0:
            plt.figure(figsize=(8, 6))
            plt.imshow(data_array[idx], extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
            plt.colorbar(label='Random Data')
            plt.title(f'Random Data Sample {idx}')
            plt.xlabel('X')
            plt.ylabel('Y') 
            plt.savefig(os.path.join(plots_dir, f'random_data_plot_{idx}.png'))
            plt.close()

    file_path_domain = os.path.join('generated', 'random_data_normal.npy')
    os.makedirs('generated', exist_ok=True)
    np.save(file_path_domain, data_array)
