import numpy as np
import os
import argparse
import yaml
from multiprocessing import get_context
from scipy.interpolate import RegularGridInterpolator as rgi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importing this for 3D plotting
from tqdm import tqdm as log_progress

parser = argparse.ArgumentParser(description='Point Charge Dataset')
parser.add_argument('-c', '--cfg', type=str, default=None, help='Config filename')
args = parser.parse_args()

with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)

n_fields = cfg['n_it']
nits = cfg['n_it']
plotting = False

if __name__ == '__main__':
    xmin, xmax, nnx = cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['nnx']
    ymin, ymax, nny = cfg['domain']['ymin'], cfg['domain']['ymax'], cfg['domain']['nny']
    zmin, zmax, nnz = cfg['domain']['zmin'], cfg['domain']['zmax'], cfg['domain']['nnz']
    if not os.path.exists('generated'):
        os.makedirs('generated')

    x, y, z = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny), np.linspace(zmin, zmax, nnz)

    amplitude = 1.8e+5
    points = np.array(np.meshgrid(x, y, z, indexing='ij')).T.reshape(-1, 3)

    def generate_field(nits):
        for i in range(nits):
            data = np.zeros((nnx, nny, nnz))
            # Creating a Gaussian charge in the center
            amplitude = 1.8e+11  # Adjust as needed
            sigma = 5.0e-3 # Adjust as needed
            gauss = lambda x, y, z: amplitude * np.exp(-((x * (xmax - xmin) / nnx - (xmax - xmin) / 2) ** 2 +
                                             (y * (ymax - ymin) / nny - (ymax - ymin) / 2) ** 2 +
                                             (z * (zmax - zmin) / nnz - (zmax - zmin) / 2) ** 2) / (2 * sigma ** 2))
            for xi in range(nnx):
                for yi in range(nny):
                    for zi in range(nnz):
                        data[xi, yi, zi] = gauss(xi, yi, zi)
            f = rgi((x, y, z), data, method='cubic')
            yield f(points).reshape((nnx, nny, nnz))

# Generate fields
fields = np.empty((n_fields, nnx, nny, nnz))
potentials = np.empty((n_fields, nnx, nny, nnz))  # Array to store potentials
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

    # Save fields and potentials
    file_path_fields = os.path.join('generated', 'fields.npy')
    file_path_potentials = os.path.join('generated', 'potentials.npy')
    np.save(file_path_fields, fields)
    np.save(file_path_potentials, potentials)

    # Save generated data
    plots_dir = os.path.join('generated', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    file_path_fields = os.path.join('generated', 'fields.npy')
    print(np.shape(fields))
    print(np.shape(potentials))
    os.makedirs('generated', exist_ok=True)
    np.save(file_path_fields, fields)


    # # 3D plot of potential
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_grid, y_grid, z_grid, c=potentials.ravel(), cmap='viridis')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('Potential Distribution')
    # plt.show()

    # # 2D plot of potential in the middle of the domain
    # plt.figure()
    # plt.imshow(potentials[:, :, nnz // 2], cmap='viridis', origin='lower', extent=[xmin, xmax, ymin, ymax])
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Potential Distribution in the Middle of the Domain')
    # plt.colorbar(label='Potential')
    # plt.show()
