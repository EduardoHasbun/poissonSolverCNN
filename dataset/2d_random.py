#############################################################################################################
#                                                                                                           #
#                             RUN:    python 2d_random.py -c train.yml                                      #
#                                                                                                           #
#############################################################################################################

import numpy as np
import os
from multiprocessing import get_context
import argparse
import yaml
from scipy import interpolate
import matplotlib.pyplot as plt

os.environ['OPENBLAS_NUM_THREADS'] = '1'

args = argparse.ArgumentParser(description='RHS random dataset')
args.add_argument('-c', '--cfg', type=str, default=None,
                help='Config filename')
args.add_argument('--case', type=str, default=None, help='Case name')

# Specific arguments
args.add_argument('-nr', '--n_res_factor', default=16, type=int,
                    help='grid of npts/nres on which the random set is taken')
args = args.parse_args()

with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)

nits = cfg['n_it']

if __name__ == '__main__':
    # Parameters for data generation
    xmin, xmax, nnx = cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['nnx']
    nny, ymin, ymax = cfg['domain']['nny'], cfg['domain']['ymin'], cfg['domain']['ymax']
    n_res_factor = args.n_res_factor
    # Create a grid
    x, y= np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny)

    # Factor to divide the grid by to generate the random grid
    nnx_lower = int(nnx / n_res_factor)
    nny_lower = int(nny / n_res_factor)
    x_lower, y_lower = np.linspace(xmin, xmax, nnx_lower), np.linspace(ymin, ymax, nny_lower)

    def generate_random_data(nits):
        """ Generate random data samples """
        for i in range(nits):
            z_lower = 2 * np.random.random((nny_lower, nnx_lower)) - 1
            f = interpolate.interp2d(x_lower, y_lower, z_lower, kind='cubic')
            yield f(x, y)

    # Create a directory for saving data (if needed)
    data_dir = cfg['output_dir']
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    plots_dir = os.path.join(data_dir, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Generate random data samples
    random_data_array = np.empty((nits, nnx, nny))
    for idx, random_data in enumerate(generate_random_data(nits)):
        random_data_array[idx] = random_data

        if idx%100==0:
            plt.figure(figsize=(8, 6))
            plt.imshow(random_data, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
            plt.colorbar(label='Random Data')
            plt.title(f'Random Data Sample {idx}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.savefig(os.path.join(plots_dir, f'random_data_plot_{idx}.png'))
            plt.close()

    # Save the 3D numpy array as a single .npy file
    data_filename = os.path.join(data_dir, 'random_data.npy')
    np.save(data_filename, random_data_array)

