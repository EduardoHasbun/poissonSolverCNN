import numpy as np
import os
from multiprocessing import Pool, cpu_count
import yaml
from scipy import special as sp
import argparse
from tqdm import tqdm as log_progress
import matplotlib.pyplot as plt
import time

def G(X, q, xq, epsilon):
    r_vec_expanded = np.expand_dims(X, axis=1)
    x_qs_expanded = np.expand_dims(xq, axis=0)
    r_diff = r_vec_expanded - x_qs_expanded
    r = np.sqrt(np.sum(np.square(r_diff), axis=2))
    q_over_r = q / r
    total_sum = np.sum(q_over_r, axis=1)
    result = (1 / (epsilon * 4 * np.pi)) * total_sum
    result = np.expand_dims(result, axis=1)
    return result

def Spherical_Harmonics(x, y, z, q, xq, E_1, E_2, kappa, R, labels, points, N):
    PHI = np.zeros(len(points))
    for K in range(len(points)):
        px, py, pz = points[K]
        ix = int((px - x[0]) / (x[1] - x[0]))
        iy = int((py - y[0]) / (y[1] - y[0]))
        iz = int((pz - z[0]) / (z[1] - z[0]))
        rho = np.sqrt(np.sum(points[K,:] ** 2))
        zenit = np.arccos(points[K, 2] / rho)
        azim = np.arctan2(points[K, 1], points[K, 0])
        phi = 0.0 + 0.0 * 1j
        for n in range(N):
            for m in range(-n, n + 1):
                Enm = 0.0
                for k in range(len(q)):
                    rho_k = np.sqrt(np.sum(xq[k,:] ** 2))
                    zenit_k = np.arccos(xq[k, 2] / rho_k)
                    azim_k = np.arctan2(xq[k, 1], xq[k, 0])
                    Enm += (
                        q[k]
                        * rho_k**n
                        *4*np.pi/(2*n+1)
                        * sp.sph_harm(m, n, -azim_k, zenit_k)
                    )
                Anm = Enm * (1/(4*np.pi)) * ((2*n+1)) / (np.exp(-kappa*R)* ((E_1-E_2)*n*get_K(kappa*R,n)+E_2*(2*n+1)*get_K(kappa*R,n+1)))
                Bnm = 1/(R**(2*n+1))*(np.exp(-kappa*R)*get_K(kappa*R,n)*Anm - 1/(4*np.pi*E_1)*Enm)
                if labels[ix, iy, iz]=='molecule':
                    phi += Bnm * rho**n * sp.sph_harm(m, n, azim, zenit)
                if labels[ix, iy, iz]=='solvent':
                    phi += Anm * rho**(-n-1)* np.exp(-kappa*rho) * get_K(kappa*rho,n) * sp.sph_harm(m, n, azim, zenit)
        if labels[ix, iy, iz] == "solvent":
            phi -= G(np.array([points[K]]), q, xq, E_1)
        PHI[K] = np.real(phi)
    return PHI

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
    # Parameters for data generation
    xmin, xmax, nnx = cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['nnx']
    ymin, ymax, nny = cfg['domain']['ymin'], cfg['domain']['ymax'], cfg['domain']['nny']
    zmin, zmax, nnz = cfg['domain']['zmin'], cfg['domain']['zmax'], cfg['domain']['nnz']
    R = cfg['spherical_harmonics']['R']

    # Create the grid
    x, y, z = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny), np.linspace(zmin, zmax, nnz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    interface_center = [0, 0, 0]
    interface_mask = (X - interface_center[0]) ** 2 + (Y - interface_center[1]) ** 2 + (Z - interface_center[2]) ** 2 <= R ** 2
    labels = np.where(interface_mask, "molecule", "solvent")


    # Generate random charges and locations
    min_charges, max_charges = cfg['spherical_harmonics']['min_charges'], cfg['spherical_harmonics']['max_charges']
    num_charges = np.random.randint(min_charges, max_charges + 1)
    charge_range = cfg['spherical_harmonics']['charge_range']
    # q = np.random.uniform(charge_range[0], charge_range[1], num_charges)
    q = ([1.0])
    location_range = cfg['spherical_harmonics']['location_range']
    # xq = np.random.uniform(location_range[0], location_range[1], (num_charges, 3))
    xq = np.array([[0.2, 0.1, 0.1]])
    # print(f'Random charges (q): {q}')
    # print(f'Random locations (xq): {xq}')


    # Parameters for Spherical Harmonics
    E_1 = cfg['spherical_harmonics']['E_1']
    E_2 = cfg['spherical_harmonics']['E_2']
    kappa = cfg['spherical_harmonics']['kappa']
    N = cfg['spherical_harmonics']['N']

    # Calculate the spherical harmonics field
    field = Spherical_Harmonics(x, y, z, q, xq, E_1, E_2, kappa, R, labels, points, N)
    
    # RHS for punctual charges
    rhs = G(points, q, xq, E_1).reshape((nnx, nny, nnz))
    
    return rhs, field.reshape((nnx, nny, nnz))

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


    # Generate random data samples
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
        for i in range(nits):
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # Changed to 1 row, 3 columns
            ax[0].imshow(rhs_data_array[i, :, :, cfg['domain']['nnz'] // 2])
            ax[0].set_title('RHS')
            ax[1].imshow(potentials_data_array[i, :, :, cfg['domain']['nnz'] // 2])
            ax[1].set_title('Potentials')
            ax[2].plot(potentials_data_array[i, :, cfg['domain']['nny'] // 2, cfg['domain']['nnz'] // 2], label='Potentials x-line')  # Plotting x line for Potentials
            ax[2].set_title('X Line')
            ax[2].legend()
            plt.savefig(os.path.join(plots_dir, f'plot_{i}.png'))
            plt.close()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to compute the script: {elapsed_time:.2f} seconds")