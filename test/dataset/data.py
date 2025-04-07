import numpy as np
import os
from multiprocessing import Pool, cpu_count
import yaml
from scipy import special as sp
import argparse
from tqdm import tqdm as log_progress
import matplotlib.pyplot as plt
import time


def rhs_punctual_charges(points, q, xq, sigma=0.02):
    rhs = np.zeros(points.shape[0])

    for i in range(len(q)):
        r0 = xq[i]
        dist2 = np.sum((points - r0)**2, axis=1)
        prefactor = -q[i] / ((2 * np.pi * sigma**2)**1.5)
        rhs += prefactor * np.exp(-dist2 / (2 * sigma**2))

    return rhs

def spherical_to_cartesian(radius, zenith, azimuth):
    """Convert spherical coordinates to Cartesian."""
    x = radius * np.sin(zenith) * np.cos(azimuth)
    y = radius * np.sin(zenith) * np.sin(azimuth)
    z = radius * np.cos(zenith)
    return np.array([x, y, z])


def generate_random(args):
    cfg, i = args
    xmin, xmax, nnx = cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['nnx']
    ymin, ymax, nny = cfg['domain']['ymin'], cfg['domain']['ymax'], cfg['domain']['nny']
    zmin, zmax, nnz = cfg['domain']['zmin'], cfg['domain']['zmax'], cfg['domain']['nnz']
    R = cfg['gausseans']['R']
    max_radius = cfg['gausseans']['max_radius'] * R
    sigma = cfg['gausseans']['sigma']

    x, y, z = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny), np.linspace(zmin, zmax, nnz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    min_charges, max_charges = cfg['gausseans']['min_charges'], cfg['gausseans']['max_charges']
    num_charges = np.random.randint(min_charges, max_charges + 1)
    charge_range = cfg['gausseans']['charge_range']
    q = np.random.uniform(charge_range[0], charge_range[1], num_charges)

    radii = np.random.uniform(0, max_radius, num_charges)
    zenith_angles = np.random.uniform(0, np.pi, num_charges)
    azimuth_angles = np.random.uniform(0, 2 * np.pi, num_charges)
    xq = np.array([spherical_to_cartesian(r, zen, az) for r, zen, az in zip(radii, zenith_angles, azimuth_angles)])

    rhs = rhs_punctual_charges(points, q, xq).reshape((nnx, nny, nnz))
    return rhs, q, xq


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dataset')
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
    # Instead of separate arrays for charges and positions, 
    # we can store them in lists, then convert to an object array if needed.
    q_list = []
    xq_list = []

    args_list = [(cfg, i) for i in range(nits)]
    for idx, (rhs_data, q, xq) in log_progress(enumerate(pool.imap(generate_random, args_list)), 
                                               total=nits, desc="Processing"):
        rhs_data_array[idx] = rhs_data
        q_list.append(q)
        xq_list.append(xq)

    # Convert lists to arrays of dtype=object if the lengths of q or xq vary:
    q_array = np.array(q_list, dtype=object)
    xq_array = np.array(xq_list, dtype=object)

    os.makedirs('generated', exist_ok=True)

    # Save 
    np.savez_compressed(os.path.join('generated', 'dataset.npz'),
                        rhs=rhs_data_array,
                        q=q_array,
                        xq=xq_array)

    # Optional plotting
    if plotting:
        for idx in range(nits):
            fig, axes = plt.subplots(1, 1, figsize=(10, 5))
            ax = axes
            ax.imshow(rhs_data_array[idx, cfg['domain']['nnx'] // 2], cmap='jet')
            ax.set_title('RHS')
            plt.savefig(os.path.join(plots_dir, f'plot_{idx}.png'))
            plt.close()
