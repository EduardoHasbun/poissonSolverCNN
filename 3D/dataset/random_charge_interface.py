import numpy as np
import os
from multiprocessing import Pool, cpu_count
import yaml
from scipy import special as sp
import argparse
from tqdm import tqdm as log_progress
import matplotlib.pyplot as plt
import time

def spherical_to_cartesian(radius, zenith, azimuth):
    """Convert spherical coordinates to Cartesian."""
    x = radius * np.sin(zenith) * np.cos(azimuth)
    y = radius * np.sin(zenith) * np.sin(azimuth)
    z = radius * np.cos(zenith)
    return np.array([x, y, z])

def G(X, q, xq, epsilon):
    r_vec_expanded = np.expand_dims(X, axis=1)  # Shape: (n, 1, 3)
    x_qs_expanded = np.expand_dims(xq, axis=0)  # Shape: (1, m, 3)
    r_diff = r_vec_expanded - x_qs_expanded     # Shape: (n, m, 3)
    r = np.sqrt(np.sum(np.square(r_diff), axis=2))  # Shape: (n, m)
    q_over_r = q / r  # Shape: (n, m)
    total_sum = np.sum(q_over_r, axis=1)  # Shape: (n,)
    result = (1 / (epsilon * 4 * np.pi)) * total_sum  # Shape: (n,)
    return result


def Spherical_Harmonics(x, y, z, q, xq, E_1, E_2, kappa, R, labels, points, N):
    # Precompute values for all points
    points = np.array(points)
    rho = np.linalg.norm(points, axis=1)
    zenit = np.arccos(points[:, 2] / rho)
    azim = np.arctan2(points[:, 1], points[:, 0])

    xq = np.array(xq)
    rho_k = np.linalg.norm(xq, axis=1)
    zenit_k = np.arccos(xq[:, 2] / rho_k)
    azim_k = np.arctan2(xq[:, 1], xq[:, 0])
    
    # Precompute the grid indices for labels
    ix = ((points[:, 0] - x[0]) / (x[1] - x[0])).astype(int)
    iy = ((points[:, 1] - y[0]) / (y[1] - y[0])).astype(int)
    iz = ((points[:, 2] - z[0]) / (z[1] - z[0])).astype(int)

    PHI = np.zeros(len(points), dtype=np.complex128)

    # Loop over n and m
    for n in range(N):
        for m in range(-n, n + 1):
            # Compute Enm for all points
            Enm = np.sum(
                q[:, None]
                * rho_k[:, None]**n
                * (4 * np.pi / (2 * n + 1))
                * sp.sph_harm(m, n, -azim_k[:, None], zenit_k[:, None]),
                axis=0
            )
            Anm = Enm * (1 / (4 * np.pi)) * ((2 * n + 1)) / (
                np.exp(-kappa * R) * ((E_1 - E_2) * n * get_K(kappa * R, n) + E_2 * (2 * n + 1) * get_K(kappa * R, n + 1))
            )
            Bnm = 1 / (R ** (2 * n + 1)) * (
                np.exp(-kappa * R) * get_K(kappa * R, n) * Anm - 1 / (4 * np.pi * E_1) * Enm
            )

            # Compute phi based on labels
            is_molecule = labels[ix, iy, iz] == "molecule"
            is_solvent = labels[ix, iy, iz] == "solvent"

            PHI[is_molecule] += (
                Bnm * rho[is_molecule]**n * sp.sph_harm(m, n, azim[is_molecule], zenit[is_molecule])
            )
            PHI[is_solvent] += (
                Anm
                * rho[is_solvent] ** (-n - 1)
                * np.exp(-kappa * rho[is_solvent])
                * get_K(kappa * rho[is_solvent], n)
                * sp.sph_harm(m, n, azim[is_solvent], zenit[is_solvent])
            )

    # Final adjustment for solvent
    is_solvent = labels[ix, iy, iz] == "solvent"
    PHI[is_solvent] -= G(points[is_solvent], q, xq, E_1)

    return np.real(PHI)

def get_K(x, n):
    K = 0.0
    n_fact = sp.factorial(n)
    n_fact2 = sp.factorial(2 * n)
    for s in range(n + 1):
        K += (
            2**s
            * n_fact
            * sp.factorial(2 * n - s)
            / (sp.factorial(s) * n_fact2 * sp.factorial(n - s))
            * x**s
        )
    return K

def generate_random(args):
    cfg, i = args
    xmin, xmax, nnx = cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['nnx']
    ymin, ymax, nny = cfg['domain']['ymin'], cfg['domain']['ymax'], cfg['domain']['nny']
    zmin, zmax, nnz = cfg['domain']['zmin'], cfg['domain']['zmax'], cfg['domain']['nnz']
    R = cfg['spherical_harmonics']['R']
    max_radius = cfg['spherical_harmonics']['max_radius'] * R

    x, y, z = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny), np.linspace(zmin, zmax, nnz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    interface_center = [0, 0, 0]
    interface_mask = (X - interface_center[0]) ** 2 + (Y - interface_center[1]) ** 2 + (Z - interface_center[2]) ** 2 <= R**2
    labels = np.where(interface_mask, "molecule", "solvent")

    min_charges, max_charges = cfg['spherical_harmonics']['min_charges'], cfg['spherical_harmonics']['max_charges']
    num_charges = np.random.randint(min_charges, max_charges + 1)
    charge_range = cfg['spherical_harmonics']['charge_range']
    q = np.random.uniform(charge_range[0], charge_range[1], num_charges)

    radii = np.random.uniform(0, max_radius, num_charges)
    zenith_angles = np.random.uniform(0, np.pi, num_charges)
    azimuth_angles = np.random.uniform(0, 2 * np.pi, num_charges)
    xq = np.array([spherical_to_cartesian(r, zen, az) for r, zen, az in zip(radii, zenith_angles, azimuth_angles)])

    E_1 = cfg['spherical_harmonics']['E_1']
    E_2 = cfg['spherical_harmonics']['E_2']
    kappa = cfg['spherical_harmonics']['kappa']
    N = cfg['spherical_harmonics']['N']

    field = Spherical_Harmonics(x, y, z, q, xq, E_1, E_2, kappa, R, labels, points, N).reshape((nnx, nny, nnz))
    mid_x, mid_y, mid_z = nnx // 2, nny // 2, nnz // 2
    neighbors = [
        field[mid_x + 1, mid_y, mid_z],
        field[mid_x - 1, mid_y, mid_z],
        field[mid_x, mid_y + 1, mid_z],
        field[mid_x, mid_y - 1, mid_z],
        field[mid_x, mid_y, mid_z + 1],
        field[mid_x, mid_y, mid_z - 1]
    ]
    field[mid_x, mid_y, mid_z] = np.mean(neighbors)
    rhs = G(points, q, xq, E_1).reshape((nnx, nny, nnz))
    return rhs, field

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dataset for spherical harmonics')
    parser.add_argument('-c', '--cfg', type=str, default=None, help='Config filename')
    args = parser.parse_args()
    with open(args.cfg, 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)
    nits = cfg['n_it']
    plotting = cfg['plotting']

    start_time = time.time()
    pool = Pool(processes=cpu_count())

    plots_dir = os.path.join('generated', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    rhs_data_array = np.empty((nits, cfg['domain']['nnx'], cfg['domain']['nny'], cfg['domain']['nnz']))
    potentials_data_array = np.empty((nits, cfg['domain']['nnx'], cfg['domain']['nny'], cfg['domain']['nnz']))

    args_list = [(cfg, i) for i in range(nits)]
    for idx, (rhs_data, potentials_data) in log_progress(enumerate(pool.imap(generate_random, args_list)), total=nits, desc="Processing"):
        rhs_data_array[idx] = rhs_data
        potentials_data_array[idx] = potentials_data

    rhs_file_path = os.path.join('generated', 'rhs_data.npy')
    potentials_file_path = os.path.join('generated', 'potentials_data.npy')

    os.makedirs('generated', exist_ok=True)
    np.save(rhs_file_path, rhs_data_array)
    np.save(potentials_file_path, potentials_data_array)


    if plotting:
        for idx in range(nits):
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(rhs_data_array[idx, cfg['domain']['nnx'] // 2], cmap='jet')
            ax[0].set_title('RHS')
            ax[1].imshow(potentials_data_array[idx, cfg['domain']['nnx'] // 2], cmap='jet')
            ax[1].set_title('Potentials')
            plt.savefig(os.path.join(plots_dir, f'plot_{idx}.png'))
            plt.close()