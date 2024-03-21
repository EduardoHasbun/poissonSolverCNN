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

    # Define the positions and magnitudes of the charges
    charges = [
        {'position': [(xmax - xmin) * 0.25, (ymax - ymin) * 0.25, (zmax - zmin) * 0.5], 'magnitude': 1.8e+3, 'sigma': 1.0e-3},
        {'position': [(xmax - xmin) * 0.75, (ymax - ymin) * 0.75, (zmax - zmin) * 0.5], 'magnitude': 1.8e+3, 'sigma': 1.0e-3}
    ]

    def gauss(x, y, z, charge):
        amplitude = charge['magnitude']
        sigma = charge['sigma']
        return amplitude * np.exp(-((x - charge['position'][0]) ** 2 +
                                    (y - charge['position'][1]) ** 2 +
                                    (z - charge['position'][2]) ** 2) / (2 * sigma ** 2))

    # Generate the field outside the loop
    data = np.zeros((nnx, nny, nnz))
    for charge in charges:
        for xi in range(nnx):
            for yi in range(nny):
                for zi in range(nnz):
                    data[xi, yi, zi] += gauss(xi * (xmax - xmin) / nnx,
                                               yi * (ymax - ymin) / nny,
                                               zi * (zmax - zmin) / nnz,
                                               charge)

    f = rgi((x, y, z), data, method='cubic')

    # Generate fields and potentials
    fields = np.empty((n_fields, nnx, nny, nnz))
    potentials = np.empty((n_fields, nnx, nny, nnz))

    for idx, _ in log_progress(enumerate(range(n_fields)), total=n_fields, desc="Generating Fields"):
        fields[idx] = f(points).reshape((nnx, nny, nnz))
        potentials[idx] = np.zeros((nnx, nny, nnz))  # Initialize potentials for each field
        for charge in charges:
            for xi in range(nnx):
                for yi in range(nny):
                    for zi in range(nnz):
                        distance = np.sqrt(((xi * (xmax - xmin) / nnx) - charge['position'][0]) ** 2 +
                                           ((yi * (ymax - ymin) / nny) - charge['position'][1]) ** 2 +
                                           ((zi * (zmax - zmin) / nnz) - charge['position'][2]) ** 2)
                        if distance ==0:
                            potentials[idx, xi, yi, zi] += charge['magnitude'] /((xmax - xmin) / nnx)
                        else:
                            potentials[idx, xi, yi, zi] += charge['magnitude'] / distance

    # Save fields and potentials
    file_name_fields = (f"fields_2_charges_"
    f"X1_{charges[0]['position'][0]}_Y1_{charges[0]['position'][1]}_Z1_{charges[0]['position'][2]}_"
    f"X2_{charges[1]['position'][0]}_Y2_{charges[1]['position'][1]}_Z2_{charges[1]['position'][2]}.npy")
   
    file_name_potentials = (f"potentials_2_charges_"
    f"X1_{charges[0]['position'][0]}_Y1_{charges[0]['position'][1]}_Z1_{charges[0]['position'][2]}_"
    f"X2_{charges[1]['position'][0]}_Y2_{charges[1]['position'][1]}_Z2_{charges[1]['position'][2]}.npy")
 
    file_path_fields = os.path.join('generated', file_name_fields)
    file_path_potentials = os.path.join('generated', file_name_potentials)
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
