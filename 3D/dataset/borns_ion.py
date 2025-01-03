import numpy as np
from scipy import special as sp
import matplotlib.pyplot as plt

def Spherical_Harmonics(x, q, xq, E_1, E_2, kappa, R, flag = 'molecule', N=5):
    PHI = np.zeros(len(x))

    for K in range(len(x)):
        rho = np.sqrt(np.sum(x[K,:] ** 2))
        zenit = np.arccos(x[K, 2] / rho)
        azim = np.arctan2(x[K, 1], x[K, 0])

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
                
                if flag=='molecule':
                    phi += Bnm * rho**n * sp.sph_harm(m, n, azim, zenit)
                if flag=='solvent':
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

# -------------------------------
# Example usage: Single point charge at the origin in 3D
if __name__ == "__main__":
    # Define a single point charge at the origin
    q = np.array([1.0])    # single positive charge
    xq = np.array([[0.0, 0.0, 0.0]])  # at origin

    # Dielectric parameters and other constants
    E_1 = 1.0  # Permittivity inside (e.g. water)
    E_2 = 80.0   # Permittivity outside
    kappa = 0.0 # No screening
    R = 1.0     # Some radius parameter for boundary conditions
    flag = 'molecule' # Evaluate as if inside the molecule region

    # Create a 3D grid: We'll take a cube from -1 to 1 in x, y, z
    Nx = 50
    Ny = 50
    Nz = 50
    x_vals = np.linspace(-3.0, 3.0, Nx)
    y_vals = np.linspace(-3.0, 3.0, Ny)
    z_vals = np.linspace(-3.0, 3.0, Nz)

    # Create all 3D points
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')  # shape: (Nx, Ny, Nz)
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Compute the potential in 3D
    PHI = Spherical_Harmonics(points, q, xq, E_1, E_2, kappa, R, flag = flag)
    PHI_3D = PHI.reshape((Nx, Ny, Nz))


    # Find index closest to zero
    ix0 = Nx//2
    iy0 = Ny//2
    iz0 = Nz//2

    fig, axes = plt.subplots(1, 3, figsize=(15,5))

    # Slice in xy-plane at z=0
    c_xy = axes[0].contourf(x_vals, y_vals, PHI_3D[:,:,iz0].T, 50, cmap='RdBu_r')
    axes[0].set_title('z=0 plane')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')
    plt.colorbar(c_xy, ax=axes[0])

    # Slice in xz-plane at y=0
    c_xz = axes[1].contourf(x_vals, z_vals, PHI_3D[:,iy0,:].T, 50, cmap='RdBu_r')
    axes[1].set_title('y=0 plane')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('z')
    axes[1].set_aspect('equal')
    plt.colorbar(c_xz, ax=axes[1])

    # Slice in yz-plane at x=0
    c_yz = axes[2].contourf(y_vals, z_vals, PHI_3D[ix0,:,:].T, 50, cmap='RdBu_r')
    axes[2].set_title('x=0 plane')
    axes[2].set_xlabel('y')
    axes[2].set_ylabel('z')
    axes[2].set_aspect('equal')
    plt.colorbar(c_yz, ax=axes[2])

    plt.tight_layout()
    plt.show()
