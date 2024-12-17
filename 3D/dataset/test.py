import numpy as np
from scipy import special as sp
import matplotlib.pyplot as plt


def Spherical_Harmonics(x, y, z, points, q, xq, E_1, E_2, kappa, R, labels, N = 20):

    PHI = np.zeros(len(points))

    for K in range(len(points)):
        px, py, pz = points[K]
        ix = int((px - x[0]) / (x[1] - x[0]))  # Convert x to grid index
        iy = int((py - y[0]) / (y[1] - y[0]))  # Convert y to grid index
        iz = int((pz - z[0]) / (z[1] - z[0]))  # Convert z to grid index
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


# General Parameters
q = np.array([[1.0]])
xq = np.array([[0.45, 0.0e+00, 0.0e+00]])
E_1 = 1
E_2 = 80
kappa = 0.125
R = 0.5
N = 5

# Define the grid size, mesh and labels
grid_size = 50
x = np.linspace(-1, 1, grid_size)
y = np.linspace(-1, 1, grid_size)
z = np.linspace(-1, 1, grid_size)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
interface_center = [0, 0, 0]
interface_mask = (X - interface_center[0]) ** 2 + (Y - interface_center[1]) ** 2 + (Z - interface_center[2]) ** 2 <= R ** 2
labels = np.where(interface_mask, "molecule", "solvent")

# Call the function
field = Spherical_Harmonics(x, y, z, points, q, xq, E_1, E_2, kappa, R, labels, N)

# Plot the results
plt.imshow(field.reshape((grid_size, grid_size, grid_size))[:, :, grid_size // 2], extent=(-1, 1, -1, 1))
plt.savefig('imshow.png')

plt.plot(field.reshape((grid_size, grid_size, grid_size))[:, grid_size // 2, grid_size // 2])
plt.savefig('line.png')



