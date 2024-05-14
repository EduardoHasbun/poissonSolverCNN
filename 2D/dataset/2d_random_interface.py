import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import argparse
from scipy.interpolate import RegularGridInterpolator as rgi
from tqdm import tqdm as log_progress
from multiprocessing import Pool, cpu_count

class RandomDataGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.random_inside = []

    def generate_random_data(self, i):
        # Parameters for data generation
        xmin, xmax, nnx = self.cfg['domain']['xmin'], self.cfg['domain']['xmax'], self.cfg['domain']['nnx']
        nny, ymin, ymax = self.cfg['domain']['nny'], self.cfg['domain']['ymin'], self.cfg['domain']['ymax']
        interface_center = (self.cfg['domain']['interface_center']['x'], self.cfg['domain']['interface_center']['y'])
        interface_radius = self.cfg['domain']['interface_radius']
        n_res_factor = 16

        # Create a grid
        x, y = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny)
        X, Y = np.meshgrid(x, y)

        # Generate circular mask for interface
        interface_mask = (X - interface_center[0]) ** 2 + (Y - interface_center[1]) ** 2 <= interface_radius ** 2

        # Factor to divide the grid by to generate the random grid
        nnx_lower = int(nnx / n_res_factor)
        nny_lower = int(nny / n_res_factor)
        x_lower, y_lower = np.linspace(xmin, xmax, nnx_lower), np.linspace(ymin, ymax, nny_lower)
        points = np.array(np.meshgrid(x, y, indexing='ij')).T.reshape(-1, 2)

        z_lower = 2 * np.random.random((nnx_lower, nny_lower)) - 1
        f = rgi((x_lower, y_lower,), z_lower, method='cubic')
        random_data = f(points).reshape((nnx, nny))
        self.random_inside.append(random_data)
    
        inside_domain = self.random_inside[i-1]
        outside_domain = random_data.copy()

        inside_domain[~interface_mask] = 0
        outside_domain[interface_mask] = 0
        random_data[interface_mask] = inside_domain[interface_mask]
        random_data[~interface_mask] = outside_domain[~interface_mask]

        return random_data, inside_domain, outside_domain

    def generate_random_data_parallel(self, nits):
        pool = Pool(processes=cpu_count())
        results = list(pool.imap(self.generate_random_data, range(nits)))
        pool.close()
        pool.join()
        return results

# Specific arguments
parser = argparse.ArgumentParser(description='RHS random dataset')
parser.add_argument('-c', '--cfg', type=str, default=None, help='Config filename')
args = parser.parse_args()
with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)

generator = RandomDataGenerator(cfg)

if __name__ == '__main__':
    plots_dir = os.path.join('generated', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    nits = cfg['n_it']
    plotting = False

    # Generate random data samples
    random_data_array = np.empty((nits, cfg['domain']['nnx'], cfg['domain']['nny']))
    inside_array = np.empty((nits, cfg['domain']['nnx'], cfg['domain']['nny']))
    outside_array = np.empty((nits, cfg['domain']['nnx'], cfg['domain']['nny']))

    for idx in log_progress(range(nits), desc="Processing"):
        random_data, inside_domain, outside_domain= generator.generate_random_data(idx)
        random_data_array[idx] = random_data
        inside_array[idx] = inside_domain
        outside_array[idx] = outside_domain
        plt.figure(figsize=(8, 6))
        plt.imshow(random_data)
        plt.show()


        if plotting and idx % 100 == 0:
            plt.figure(figsize=(8, 6))
            plt.imshow(random_data, extent=(cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['ymin'], cfg['domain']['ymax']), origin='lower', cmap='viridis')
            plt.colorbar(label='Random Data')
            plt.title(f'Random Data Sample {idx}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.savefig(os.path.join(plots_dir, f'random_data_plot_{idx}.png'))
            plt.close()

    file_path_domain = os.path.join('generated', 'domain.npy')
    file_path_inside = os.path.join('generated', 'inside.npy')
    file_path_outisde = os.path.join('generated', 'outside.npy')
    os.makedirs('generated', exist_ok=True)
    np.save(file_path_domain, random_data_array)
    np.save(file_path_inside, inside_array)
    np.save(file_path_outisde, outside_domain)
