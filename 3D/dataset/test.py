import numpy as np
from scipy import special as sp
import matplotlib.pyplot as plt


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


def G(X, q, xq, epsilon):
    r_vec_expanded = np.expand_dims(X, axis=1)  # Shape: (n, 1, 3)
    x_qs_expanded = np.expand_dims(xq, axis=0)  # Shape: (1, m, 3)
    r_diff = r_vec_expanded - x_qs_expanded     # Shape: (n, m, 3)
    r = np.sqrt(np.sum(np.square(r_diff), axis=2))  # Shape: (n, m)
    q_over_r = q / r  # Shape: (n, m)
    total_sum = np.sum(q_over_r, axis=1)  # Shape: (n,)
    result = (1 / (epsilon * 4 * np.pi)) * total_sum  # Shape: (n,)
    return result



def Spherical_Harmonics_loop(x, y, z, q, xq, E_1, E_2, kappa, R, labels, points, N):
    PHI = np.zeros(len(points))
    for K in range(len(points)):
        px, py, pz = points[K]
        ix = int((px - x[0]) / (x[1] - x[0]))
        iy = int((py - y[0]) / (y[1] - y[0]))
        iz = int((pz - z[0]) / (z[1] - z[0]))
        rho = np.sqrt(np.sum(points[K, :] ** 2))
        zenit = np.arccos(points[K, 2] / rho)
        azim = np.arctan2(points[K, 1], points[K, 0])
        phi = 0.0 + 0.0j
        for n in range(N):
            for m in range(-n, n + 1):
                Enm = 0.0
                for k in range(len(q)):
                    rho_k = np.sqrt(np.sum(xq[k, :] ** 2))
                    zenit_k = np.arccos(xq[k, 2] / rho_k)
                    azim_k = np.arctan2(xq[k, 1], xq[k, 0])
                    Enm += (
                        q[k]
                        * rho_k**n
                        * 4 * np.pi / (2 * n + 1)
                        * sp.sph_harm(m, n, -azim_k, zenit_k)
                    )
                Anm = Enm * (1 / (4 * np.pi)) * ((2 * n + 1)) / (
                    np.exp(-kappa * R) * ((E_1 - E_2) * n * get_K(kappa * R, n) + E_2 * (2 * n + 1) * get_K(kappa * R, n + 1))
                )
                Bnm = 1 / (R ** (2 * n + 1)) * (
                    np.exp(-kappa * R) * get_K(kappa * R, n) * Anm - 1 / (4 * np.pi * E_1) * Enm
                )
                if labels[ix, iy, iz] == "molecule":
                    phi += Bnm * rho**n * sp.sph_harm(m, n, azim, zenit)
                if labels[ix, iy, iz] == "solvent":
                    phi += (
                        Anm
                        * rho ** (-n - 1)
                        * np.exp(-kappa * rho)
                        * get_K(kappa * rho, n)
                        * sp.sph_harm(m, n, azim, zenit)
                    )
        if labels[ix, iy, iz] == "solvent":
            phi -= G(np.array([points[K]]), q, xq, E_1)
        PHI[K] = np.real(phi)
    return PHI


x = np.linspace(-2.5, 2.5, 51)
y = np.linspace(-2.5, 2.5, 51)
z = np.linspace(-2.5, 2.5, 51)
q = np.array([-1.0, 1.0])
xq = np.array([[0.3, 0.1, 0], [-0.2, 0, 0]])
E_1 = 1.0
E_2 = 80.0
kappa = 0.1
R = 1
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
interface_center = [0, 0, 0]
interface_mask = (X - interface_center[0]) ** 2 + (Y - interface_center[1]) ** 2 + (Z - interface_center[2]) ** 2 <= R**2
labels = np.where(interface_mask, "molecule", "solvent")
N = 7


field = Spherical_Harmonics(x, y, z, q, xq, E_1, E_2, kappa, R, labels, points, N)
field_loop = Spherical_Harmonics_loop(x, y, z, q, xq, E_1, E_2, kappa, R, labels, points, N)
field = field.reshape((51, 51, 51))
field_loop = field_loop.reshape((51, 51, 51))
mid_x, mid_y, mid_z = 51 // 2, 51 // 2, 51 // 2
neighbors = [
    field[mid_x + 1, mid_y, mid_z],
    field[mid_x - 1, mid_y, mid_z],
    field[mid_x, mid_y + 1, mid_z],
    field[mid_x, mid_y - 1, mid_z],
    field[mid_x, mid_y, mid_z + 1],
    field[mid_x, mid_y, mid_z - 1]
]
field[mid_x, mid_y, mid_z] = np.mean(neighbors)
field_loop[mid_x, mid_y, mid_z] = np.mean(neighbors)
fig, axs = plt.subplots(1, 3, figsize=(10, 5)) 
loop = axs[0].imshow(field_loop[:, :, 25], cmap='viridis')
vectorized = axs[1].imshow(field[:, :, 25], cmap='viridis')     
error = axs[2].imshow(field_loop[:, :, 25] - field[:, :, 25], cmap='viridis')
cbar = fig.colorbar(error, ax=axs[2], orientation='horizontal')
print(np.max(field_loop - field), np.min(field_loop - field))
plt.show()
