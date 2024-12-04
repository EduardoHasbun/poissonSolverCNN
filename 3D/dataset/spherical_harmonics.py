import numpy as np
import os
from multiprocessing import Pool, cpu_count
import yaml
from tqdm import tqdm as log_progress
import argparse
import matplotlib.pyplot as plt

# Argument parser for config file
parser = argparse.ArgumentParser(description='Dataset for Born Ion Analytical Solution')
parser.add_argument('-c', '--cfg', type=str, default=None, help='Config filename')
args = parser.parse_args()

# Load configuration
with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)

# Parameters
nits = cfg['n_iterations']
plotting = cfg['plotting']
output_dir = cfg['output_dir']
os.makedirs(output_dir, exist_ok=True)

# Born Ion Solver class
class BornIonSolver:
    def __init__(self, epsilon_1, epsilon_2, kappa, qs, R_mol):
        self.epsilon_1 = epsilon_1
        self.epsilon_2 = epsilon_2
        self.kappa = kappa
        self.qs = qs
        self.R_mol = R_mol

    def analytic_Born_Ion(self, r, R=None, index_q=0):
        if R is None:
            R = self.R_mol
        epsilon_1 = self.epsilon_1
        epsilon_2 = self.epsilon_2
        kappa = self.kappa
        q = self.qs[index_q]

        f_IN = lambda r: (q / (4 * np.pi)) * (-1 / (epsilon_1 * R) + 1 / (epsilon_2 * (1 + kappa * R) * R))
        f_OUT = lambda r: (q / (4 * np.pi)) * (np.exp(-kappa * (r - R)) / (epsilon_2 * (1 + kappa * R) * r) - 1 / (epsilon_1 * r))

        y = np.piecewise(r, [r <= R, r > R], [f_IN, f_OUT])
        return y

# Generate dataset for a single charge
def generate_single_charge(i):
    # Domain parameters
    xmin, xmax, nnx = cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['nnx']
    ymin, ymax, nny = cfg['domain']['ymin'], cfg['domain']['ymax'], cfg['domain']['nny']
    zmin, zmax, nnz = cfg['domain']['zmin'], cfg['domain']['zmax'], cfg['domain']['nnz']

    # Solver parameters
    solver = BornIonSolver(
        epsilon_1=cfg['solver']['epsilon_1'],
        epsilon_2=cfg['solver']['epsilon_2'],
        kappa=cfg['solver']['kappa'],
        qs=cfg['solver']['charges'],
        R_mol=cfg['solver']['R_mol']
    )

    # Regular grid
    x = np.linspace(xmin, xmax, nnx)
    y = np.linspace(ymin, ymax, nny)
    z = np.linspace(zmin, zmax, nnz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2)

    # Compute potential field
    potential = solver.analytic_Born_Ion(R.flatten()).reshape(nnx, nny, nnz)

    # Create RHS (charge distribution)
    rhs = np.zeros((nnx, nny, nnz))
    charge_idx = (nnx // 2, nny // 2, nnz // 2)
    rhs[charge_idx] = cfg['solver']['charges'][0]

    return rhs, potential

if __name__ == '__main__':
    # Parallel generation of datasets
    pool = Pool(processes=cpu_count())
    rhs_array = []
    potential_array = []
    for idx, data in log_progress(enumerate(pool.imap(generate_single_charge, range(nits))), total=nits, desc="Processing"):
        rhs, potential = data
        rhs_array.append(rhs)
        potential_array.append(potential)

    # Save datasets
    rhs_array = np.array(rhs_array)
    potential_array = np.array(potential_array)
    np.save(os.path.join(output_dir, 'rhs.npy'), rhs_array)
    np.save(os.path.join(output_dir, 'potentials.npy'), potential_array)
    print("RHS and potentials saved.")

    # Plotting
    if plotting:
        sample_potential = potential_array[0]
        plt.figure(figsize=(8, 6))
        plt.imshow(
            sample_potential[:, :, sample_potential.shape[2] // 2],
            extent=[cfg['domain']['xmin'], cfg['domain']['xmax'], cfg['domain']['ymin'], cfg['domain']['ymax']],
            origin='lower',
            cmap='viridis'
        )
        plt.colorbar(label='Potential')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Central Slice of the Potential Field')
        plt.show()
