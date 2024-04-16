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
n_charges = cfg['n_charges']
plotting = False


if __name__ == '__main__':
    xmin, xmax, nnx = cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['nnx']
    ymin, ymax, nny = cfg['domain']['ymin'], cfg['domain']['ymax'], cfg['domain']['nny']
    zmin, zmax, nnz = cfg['domain']['zmin'], cfg['domain']['zmax'], cfg['domain']['nnz']
    if not os.path.exists('generated'):
        os.makedirs('generated')

    x, y, z = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny), np.linspace(zmin, zmax, nnz)

    points = np.array(np.meshgrid(x, y, z, indexing='ij')).T.reshape(-1, 3)

    def gauss(z, y, x, charge):
        amplitude = charge['magnitude']
        sigma = charge['sigma']
        return amplitude * np.exp(-((x - charge['position'][0]) ** 2 +
                                    (y - charge['position'][1]) ** 2 +
                                    (z - charge['position'][2]) ** 2) / (2 * sigma ** 2))
    
    def generate_random_charge(xmin, xmax, ymin, ymax, zmin, zmax):
        random_position = [np.random.uniform((xmax - xmin)*0.4, (xmax - xmin)*0.6),
                           np.random.uniform((ymax - ymin)*0.4, (ymax - ymin)*0.6),
                           np.random.uniform((zmax - zmin)*0.4, (zmax - zmin)*0.6)]
        random_magnitude = np.random.uniform(5e0, 1e+1) 
        return {'position': random_position, 'magnitude': random_magnitude, 'sigma': 1.0e-3}


    # Initialize arrays
    fields = np.empty((n_fields, nnx, nny, nnz))
    potentials = np.empty((n_fields, nnx, nny, nnz))
    positions_data = []

    for idx in log_progress(range(n_fields), total=n_fields, desc="Generating Fields"):
        # Generate multiple random charges for this field
        charges = [generate_random_charge(xmin, xmax, ymin, ymax, zmin, zmax) for _ in range(n_charges)] 

        # Initialize data for field and potential
        data = np.zeros((nnx, nny, nnz))
        potential_data = np.zeros((nnx, nny, nnz))
        
        # Loop over each charge and update the field and potential
        for charge in charges:
            positions_data.append((charge['position'][0], charge['position'][1], charge['position'][2]))
            for xi in range(nnx):
                for yi in range(nny):
                    for zi in range(nnz):
                        x_coord = xi * (xmax - xmin) / nnx
                        y_coord = yi * (ymax - ymin) / nny
                        z_coord = zi * (zmax - zmin) / nnz
                        
                        # Update the field
                        data[zi, yi, xi] += gauss(z_coord, y_coord, x_coord, charge)
                        
                        # Update the potential, considering the distance to avoid division by zero
                        distance = np.sqrt((x_coord - charge['position'][0]) ** 2 +
                                           (y_coord - charge['position'][1]) ** 2 +
                                           (z_coord - charge['position'][2]) ** 2)
                        if distance != 0:  # Avoid division by zero
                            potential_data[xi, yi, zi] += charge['magnitude'] / distance

        # Interpolate the field data for smoother visualization
        f = rgi((x, y, z), data, method='cubic')
        fields[idx] = f(points).reshape((nnx, nny, nnz))
        potentials[idx] = potential_data
    
    file_path_fields = os.path.join('generated', 'fields.npy')
    file_path_potentials = os.path.join('generated', 'potentials.npy')
    np.save(file_path_fields, fields)
    np.save(file_path_potentials, potentials)

    if plotting:
        plt.figure()
        plt.imshow(fields[0, :, :, nnz//2], cmap='viridis', origin='lower', extent=[xmin, xmax, ymin, ymax])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Field center {positions_data[0]}')
        plt.colorbar(label='Field')
        plt.show()
        
        plt.figure()
        plt.imshow(potentials[0, :, :, nnz // 2], cmap='viridis', origin='lower', extent=[xmin, xmax, ymin, ymax])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Potential {positions_data[0]}')
        plt.colorbar(label='Potential')
        plt.show()



        plt.figure()
        plt.imshow(fields[1, :, :, nnz//2], cmap='viridis', origin='lower', extent=[xmin, xmax, ymin, ymax])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Field {positions_data[1]}')
        plt.colorbar(label='Field')
        plt.show()

        plt.figure()
        plt.imshow(potentials[1, :, :, nnz // 2], cmap='viridis', origin='lower', extent=[xmin, xmax, ymin, ymax])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Potential {positions_data[0]}')
        plt.colorbar(label='Potential')
        plt.show()



        plt.figure()
        plt.imshow(fields[2, :, :, nnz//2], cmap='viridis', origin='lower', extent=[xmin, xmax, ymin, ymax])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Field {positions_data[2]}')
        plt.colorbar(label='Field')
        plt.show()

        plt.figure()
        plt.imshow(potentials[2, :, :, nnz // 2], cmap='viridis', origin='lower', extent=[xmin, xmax, ymin, ymax])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Potential {positions_data[2]}')
        plt.colorbar(label='Potential')
        plt.show()



        plt.figure()
        plt.imshow(fields[3, :, :, nnz//2], cmap='viridis', origin='lower', extent=[xmin, xmax, ymin, ymax])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Field {positions_data[3]}')
        plt.colorbar(label='Field')
        plt.show()

        plt.figure()
        plt.imshow(potentials[3, :, :, nnz // 2], cmap='viridis', origin='lower', extent=[xmin, xmax, ymin, ymax])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Potential {positions_data[3]}')
        plt.colorbar(label='Potential')
        plt.show()