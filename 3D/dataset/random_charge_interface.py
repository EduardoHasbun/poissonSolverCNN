import numpy as np
from scipy import special as sp


class Solution_utils():
    def __init__(self, qs, x_qs, epsilon_1, epsilon_2, kappa):
        self.qs = qs
        self.x_qs = x_qs
        self.epsilon_1 = epsilon_1
        self.epsilon_2 = epsilon_2
        self.kappa = kappa
        self.mesh = type('Mesh', (), {'R_max_dist': 10.0})  


    def Spherical_Harmonics(self, x, flag, R=None, N=25):

        q = self.qs
        xq = self.x_qs
        E_1 = self.epsilon_1
        E_2 = self.epsilon_2
        kappa = self.kappa
        if R is None:
            R = self.mesh.R_max_dist

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

                    Anm = Enm * (1/(4*np.pi)) * ((2*n+1)) / (np.exp(-kappa*R)* ((E_1-E_2)*n*self.get_K(kappa*R,n)+E_2*(2*n+1)*self.get_K(kappa*R,n+1)))
                    Bnm = 1/(R**(2*n+1))*(np.exp(-kappa*R)*self.get_K(kappa*R,n)*Anm - 1/(4*np.pi*E_1)*Enm)
                    
                    if flag=='molecule':
                        phi += Bnm * rho**n * sp.sph_harm(m, n, azim, zenit)
                    if flag=='solvent':
                        phi += Anm * rho**(-n-1)* np.exp(-kappa*rho) * self.get_K(kappa*rho,n) * sp.sph_harm(m, n, azim, zenit)

            PHI[K] = np.real(phi)
        
        return PHI
    @staticmethod
    def get_K(x, n):
        K = 0.0
        n_fact = sp.factorial(n)
        n_fact2 = sp.factorial(2 * n)
        for s in range(n + 1):
            K += (
                2 ** s
                * n_fact
                * sp.factorial(2 * n - s)
                / (sp.factorial(s) * n_fact2 * sp.factorial(n - s))
                * x ** s
            )
        return K


qs = [1.0, -1.0]
x_qs = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
epsilon_1 = 80.0
epsilon_2 = 1.0
kappa = 0.1

solution = Solution_utils(qs, x_qs, epsilon_1, epsilon_2, kappa)


grid_size = 60
x = np.linspace(-5, 5, grid_size)
y = np.linspace(-5, 5, grid_size)
z = np.linspace(-5, 5, grid_size)
X, Y, Z = np.meshgrid(x, y, z)
points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))


field = solution.Spherical_Harmonics(points, flag='molecule', N = 20)


field_reshaped = field.reshape(grid_size, grid_size, grid_size)

import matplotlib.pyplot as plt

plt.imshow(field_reshaped[:, :, grid_size//2])
plt.title("3D Spherical Harmonics Field")
plt.savefig('pll.png')



