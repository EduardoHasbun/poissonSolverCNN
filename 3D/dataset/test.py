import numpy as np
import os
import argparse
import yaml
from multiprocessing import get_context
from scipy.interpolate import RegularGridInterpolator as rgi
import matplotlib.pyplot as plt
from tqdm import tqdm as log_progress

parser = argparse.ArgumentParser(description='Point Charge Dataset')
parser.add_argument('-c', '--cfg', type=str, default=None, help='Config filename')
args = parser.parse_args()

with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)

n_fields = cfg['n_it']
nits = cfg['n_it']
plotting = True

if __name__ == '__main__':
    xmin, xmax, nnx = cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['nnx']
    ymin, ymax, nny = cfg['domain']['ymin'], cfg['domain']['ymax'], cfg['domain']['nny']
    zmin, zmax, nnz = cfg['domain']['zmin'], cfg['domain']['zmax'], cfg['domain']['nnz']
    if not os.path.exists('generated'):
        os.makedirs('generated')

    x, y, z = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny), np.linspace(zmin, zmax, nnz)

    points = np.array(np.meshgrid(x, y, z, indexing='ij')).T.reshape(-1, 3)

    # Generate the field outside the loop
    data = np.zeros((nnx, nny, nnz))
    amplitude = 1.8e+3  # Adjust as needed
    sigma = 1.0e-3  # Adjust as needed
    gauss = lambda x, y, z: amplitude * np.exp(-((x * (xmax - xmin) / nnx - (xmax - xmin) / 2) ** 2 +
                                                 (y * (ymax - ymin) / nny - (ymax - ymin) / 2) ** 2 +
                                                 (z * (zmax - zmin) / nnz - (zmax - zmin) / 2) ** 2) / (2 * sigma ** 2))
    for xi in range(nnx):
        for yi in range(nny):
            for zi in range(nnz):
                data[xi, yi, zi] = gauss(xi, yi, zi)
    f = rgi((x, y, z), data, method='cubic')

    # Generate fields and potentials
    fields = np.empty((n_fields, nnx, nny, nnz))
    potentials = np.empty((n_fields, nnx, nny, nnz))

    for idx, _ in log_progress(enumerate(range(n_fields)), total=n_fields, desc="Generating Fields"):
        fields[idx] = f(points).reshape((nnx, nny, nnz))
        potentials[idx] = np.zeros((nnx, nny, nnz))  # Initialize potentials for each field
        for xi in range(nnx):
            for yi in range(nny):
                for zi in range(nnz):
                    distance = np.sqrt(((xi * (xmax - xmin) / nnx) - (xmax - xmin) / 2) ** 2 +
                                    ((yi * (ymax - ymin) / nny) - (ymax - ymin) / 2) ** 2 +
                                    ((zi * (zmax - zmin) / nnz) - (zmax - zmin) / 2) ** 2)
                    if distance == 0:
                        potentials[idx, xi, yi, zi] = amplitude
                    else:
                        potentials[idx, xi, yi, zi] = 1 / distance


    # Save fields and potentials
    file_path_fields = os.path.join('generated', 'fields.npy')
    file_path_potentials = os.path.join('generated', 'potentials.npy')
    np.save(file_path_fields, fields)
    np.save(file_path_potentials, potentials)

    # Plotting
    if plotting:
        plt.figure()
        plt.imshow(fields[0, :, :, nnz//2], cmap='viridis', origin='lower', extent=[xmin, xmax, ymin, ymax])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Field ')
        plt.colorbar(label='Field')
        plt.show()

        plt.figure()
        plt.imshow(potentials[0, :, :, nnz // 2], cmap='viridis', origin='lower', extent=[xmin, xmax, ymin, ymax])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Potential Distribution in the Middle of the Domain')
        plt.colorbar(label='Potential')
        plt.show()
