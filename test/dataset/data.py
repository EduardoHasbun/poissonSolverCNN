import numpy as np
import os
from multiprocessing import Pool, cpu_count
import yaml
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

    # Ensure different RNG state per process
    np.random.seed(os.getpid() + i)

    xmin, xmax, nnx = cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['nnx']
    ymin, ymax, nny = cfg['domain']['ymin'], cfg['domain']['ymax'], cfg['domain']['nny']
    zmin, zmax, nnz = cfg['domain']['zmin'], cfg['domain']['zmax'], cfg['domain']['nnz']
    R = cfg['gausseans']['R']
    max_radius = cfg['gausseans']['max_radius'] * R
    sigma = float(cfg['gausseans']['sigma'])

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
    xq = np.array([spherical_to_cartesian(r, zen, az) 
                   for r, zen, az in zip(radii, zenith_angles, azimuth_angles)])

    rhs = rhs_punctual_charges(points, q, xq, sigma=sigma).reshape((nnx, nny, nnz))
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

    # Directory for storing plots
    plots_dir = os.path.join('generated', 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # 1) Pre-allocate RHS data
    rhs_data_array = np.empty((nits, 
                               cfg['domain']['nnx'], 
                               cfg['domain']['nny'], 
                               cfg['domain']['nnz']))

    # 2) Pre-allocate arrays for charges
    #    We'll assume a known max_charges from the config
    max_charges = cfg['gausseans']['max_charges']  
    # q_array shape:  (nits, max_charges)
    # xq_array shape: (nits, max_charges, 3)
    q_array = np.zeros((nits, max_charges), dtype=np.float64)
    xq_array = np.zeros((nits, max_charges, 3), dtype=np.float64)

    args_list = [(cfg, i) for i in range(nits)]
    for idx, (rhs_data, q, xq) in log_progress(
                enumerate(pool.imap(generate_random, args_list)),
                total=nits, desc="Processing"
    ):
        # Fill the main RHS data
        rhs_data_array[idx] = rhs_data

        # Zero-pad the charges
        num_charges = len(q)
        q_array[idx, :num_charges] = q
        xq_array[idx, :num_charges, :] = xq

        # If you like, you can leave the extra indexes at zero
        # if num_charges < max_charges. They remain zeros by default.

    # 3) Save the dataset
    os.makedirs('generated', exist_ok=True)
    np.savez_compressed(
        os.path.join('generated', 'dataset.npz'),
        rhs=rhs_data_array,
        q=q_array,
        xq=xq_array
    )

    # (Optional) Quick visualization
    if plotting:
        for idx in range(nits):
            fig, axes = plt.subplots(1, 1, figsize=(10, 5))
            ax = axes
            # Show a middle slice for demonstration
            mid_slice = cfg['domain']['nnx'] // 2
            ax.imshow(rhs_data_array[idx, mid_slice], cmap='jet')
            ax.set_title(f'RHS Sample {idx}')
            plt.savefig(os.path.join(plots_dir, f'plot_{idx}.png'))
            plt.close()

    # Optionally store charge info in a text file
    with open(os.path.join('generated', 'charges_data.txt'), 'w') as f:
        for i in range(nits):
            # Determine how many charges were actually used (non-zero)
            # For simplicity, let's consider an absolute tolerance to detect zero vs. real
            non_zero_mask = ~(np.isclose(q_array[i], 0.0))
            actual_charges = np.count_nonzero(non_zero_mask)

            f.write(f"Sample {i}:\n")
            f.write("q = " + ", ".join(map(str, q_array[i, :actual_charges])) + "\n")
            # xq lines
            for j in range(actual_charges):
                f.write("xq[%d] = %s\n" % (j, xq_array[i, j].tolist()))
            f.write("\n")

    print("Data generation complete! Elapsed time: %.2f s" % (time.time() - start_time))
