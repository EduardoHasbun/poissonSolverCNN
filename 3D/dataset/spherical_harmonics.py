import numpy as np
import os
from multiprocessing import Pool, cpu_count
import yaml
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy import special
from tqdm import tqdm as log_progress
import argparse
import matplotlib.pyplot as plt  # For plotting

# Specific arguments
parser = argparse.ArgumentParser(description='Dataset for Poisson Equation with two regions')
parser.add_argument('-c', '--cfg', type=str, default=None, help='Config filename')
args = parser.parse_args()
with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)

nits = cfg['n_it']
plotting = True # Enable plotting

# Analytical solution for spherical potentials
def an_spherical(q, xq, E_1, E_2, E_0, R, N):
    """
    Computes the analytical solution for the potential in a sphere with Nq charges inside.
    Based on Kirkwood (1934).
    """
    PHI = np.zeros(len(q))
    for K in range(len(q)):
        rho = np.sqrt(np.sum(xq[K]**2))
        zenit = np.arccos(xq[K, 2] / rho)
        azim = np.arctan2(xq[K, 1], xq[K, 0])

        phi = 0. + 0. * 1j
        for n in range(N):
            for m in range(-n, n + 1):
                sph1 = special.sph_harm(m, n, zenit, azim)
                cons1 = rho**n / (E_1 * E_0 * R**(2 * n + 1)) * (E_1 - E_2) * (
                    n + 1) / (E_1 * n + E_2 * (n + 1))
                cons2 = 4 * np.pi / (2 * n + 1)

                for k in range(len(q)):
                    rho_k = np.sqrt(np.sum(xq[k]**2))
                    zenit_k = np.arccos(xq[k, 2] / rho_k)
                    azim_k = np.arctan2(xq[k, 1], xq[k, 0])
                    sph2 = np.conj(special.sph_harm(m, n, zenit_k, azim_k))
                    phi += cons1 * cons2 * q[K] * rho_k**n * sph1 * sph2

        PHI[K] = np.real(phi) / (4 * np.pi)

    return PHI

# Generate random charge positions and potentials
def generate_single_charge_potential(i):
    # Domain configuration
    xmin, xmax, nnx = cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['nnx']
    ymin, ymax, nny = cfg['domain']['ymin'], cfg['domain']['ymax'], cfg['domain']['nny']
    zmin, zmax, nnz = cfg['domain']['zmin'], cfg['domain']['zmax'], cfg['domain']['nnz']

    # Spherical region parameters
    R = cfg['sphere']['radius']  # Interface radius
    E_1 = cfg['sphere']['epsilon_inside']  # Dielectric constant inside
    E_2 = cfg['sphere']['epsilon_outside']  # Dielectric constant outside
    E_0 = cfg['sphere']['epsilon_0']  # Vacuum permittivity

    # Place a single charge at the center of the domain
    charge_value = 1.0  # Charge magnitude
    xq = np.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)])  # Charge at the center

    # Create a regular grid
    x = np.linspace(xmin, xmax, nnx)
    y = np.linspace(ymin, ymax, nny)
    z = np.linspace(zmin, zmax, nnz)
    grid = np.meshgrid(x, y, z, indexing='ij')

    # Compute the potential at each grid point
    potential_grid = np.zeros((nnx, nny, nnz))
    for i in range(nnx):
        for j in range(nny):
            for k in range(nnz):
                # Current grid point
                point = np.array([x[i], y[j], z[k]])
                
                # Compute distance to the charge
                rho = np.linalg.norm(point - xq)

                if rho > 0:  # Avoid division by zero
                    # Determine region (inside or outside the sphere)
                    if rho <= R:
                        epsilon = E_1  # Inside the sphere
                    else:
                        epsilon = E_2  # Outside the sphere

                    # Compute potential
                    potential_grid[i, j, k] = charge_value / (4 * np.pi * epsilon * E_0 * rho)

    # Create RHS (charge distribution)
    rhs = np.zeros((nnx, nny, nnz))
    charge_idx = (
        int((nnx - 1) / 2),
        int((nny - 1) / 2),
        int((nnz - 1) / 2),
    )
    rhs[charge_idx] = charge_value  # Place the charge at the center of the grid

    return rhs, potential_grid




if __name__ == '__main__':
    pool = Pool(processes=cpu_count())

    output_dir = os.path.join('generated')
    os.makedirs(output_dir, exist_ok=True)

    print('nnx:', cfg['domain']['nnx'], 'nny:', cfg['domain']['nny'], 'nnz:', cfg['domain']['nnz'])

    # Generate dataset
    rhs_array = []
    potential_array = []
    for idx, data in log_progress(enumerate(pool.imap(generate_single_charge_potential, range(nits))), total=nits, desc="Processing"):
        rhs, potential = data
        rhs_array.append(rhs)
        potential_array.append(potential)

    # Save RHS and potentials
    rhs_array = np.array(rhs_array)
    potential_array = np.array(potential_array)

    np.save(os.path.join(output_dir, 'rhs.npy'), rhs_array)
    np.save(os.path.join(output_dir, 'potentials.npy'), potential_array)
    print("RHS saved at:", os.path.join(output_dir, 'rhs.npy'))
    print("Potentials saved at:", os.path.join(output_dir, 'potentials.npy'))

    # Plot results for the first sample
    if plotting:
        sample_rhs = rhs_array[0]
        sample_potential = potential_array[0]

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("RHS (Charge Distribution)")
        plt.imshow(sample_rhs[:, :, sample_rhs.shape[2] // 2], origin='lower', extent=[cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['ymin'], cfg['domain']['ymax']])
        plt.colorbar(label='Charge Density')
        plt.subplot(1, 2, 2)
        plt.title("Potential Field")
        plt.imshow(sample_potential[:, :, sample_potential.shape[2] // 2], origin='lower', extent=[cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['ymin'], cfg['domain']['ymax']])
        plt.colorbar(label='Potential')
        plt.show()
