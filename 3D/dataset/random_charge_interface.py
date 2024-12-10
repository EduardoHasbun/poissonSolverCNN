import numpy as np
from numpy import pi
from scipy import special as sp
import matplotlib.pyplot as plt



def Spherical_Harmonics(x, q, xq, E_1, E_2, kappa, flag, R=None, N=20):
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



q = np.array([1.0]) 
xq = np.array([[0.0, 0.0, 0.0]])  
E_1, E_2, E_0 = 1.0, 80.0, 1.0  
R = 1.0  
N = 10  
kappa = 0.1
flag = 'molecule'

x = np.linspace(-2.5, 2.5, 50)
y = np.linspace(-2.5, 2.5, 50)
z = np.linspace(-2.5, 2.5, 50)
X, Y, Z = np.meshgrid(x, y, z)

points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

potentials = Spherical_Harmonics(points, q, xq, E_1, E_2, kappa, flag, R, N)

plt.imshow(potentials[:, :, 25])
plt.savefig('pl.npg')