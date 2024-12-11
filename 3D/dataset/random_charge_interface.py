import numpy as np
import tensorflow as tf
from scipy import special as sp
from functools import wraps

class Solution_utils(): 

    qe = 1.60217663e-19
    eps0 = 8.8541878128e-12     
    kb = 1.380649e-23              
    Na = 6.02214076e23
    ang_to_m = 1e-10
    cal2j = 4.184

    pi = np.pi

    apbs_created = False
    pbj_created = False
    pbj_mesh_density = 5
    pbj_mesh_generator = 'msms'

    def phi_known(self,function,field,X,flag,*args,**kwargs):
        if function == 'Spherical_Harmonics':
            phi_values = self.Spherical_Harmonics(X, flag,*args,**kwargs)
            if flag=='solvent':
                phi_values -= self.G(X)[:,0]
        elif function == 'G_Yukawa':
            phi_values = self.G_Yukawa(X)[:,0] - self.G(X)[:,0]
        elif function == 'analytic_Born_Ion':
            phi_values = self.analytic_Born_Ion(np.linalg.norm(X, axis=1) ,*args,**kwargs)
        elif function == 'PBJ':
            phi_values = self.pbj_solution(X, flag)
            if flag=='solvent':
                phi_values -= self.G(X)[:,0]
        elif function == 'APBS':
            phi_values = self.apbs_solution(X)
        
        if field == 'react':
            return np.array(phi_values)
        elif field == 'phi':
            return np.array(phi_values + self.G(X)[:,0])


    def adim_function(function):
        @wraps(function)
        def wrapper(self,*kargs,**kwargs):
            ans = function(self,*kargs,**kwargs)
            return ans*self.beta**-1
        return wrapper

    @adim_function
    def G(self,X):
        r_vec_expanded = tf.expand_dims(X, axis=1)
        x_qs_expanded = tf.expand_dims(self.x_qs, axis=0)
        r_diff = r_vec_expanded - x_qs_expanded
        r = tf.sqrt(tf.reduce_sum(tf.square(r_diff), axis=2))
        q_over_r = self.qs / r
        total_sum = tf.reduce_sum(q_over_r, axis=1)
        result = (1 / (self.epsilon_1 * 4 * self.pi)) * total_sum
        result = tf.expand_dims(result, axis=1)
        return result
    
    @adim_function
    def dG_n(self,X,n):
        r_vec_expanded = tf.expand_dims(X, axis=1)
        x_qs_expanded = tf.expand_dims(self.x_qs, axis=0)
        r_diff = r_vec_expanded - x_qs_expanded
        r = tf.sqrt(tf.reduce_sum(tf.square(r_diff), axis=2))
        dg_dr = self.qs / (r**3) * (-1 / (self.epsilon_1 * 4 * self.pi)) * (1/2)
        dx = dg_dr * 2*r_diff[:, :, 0]
        dy = dg_dr * 2*r_diff[:, :, 1]
        dz = dg_dr * 2*r_diff[:, :, 2]
        dx_sum = tf.reduce_sum(dx, axis=1)
        dy_sum = tf.reduce_sum(dy, axis=1)
        dz_sum = tf.reduce_sum(dz, axis=1)
        dg_dn = n[:, 0] * dx_sum + n[:, 1] * dy_sum + n[:, 2] * dz_sum
        return tf.reshape(dg_dn, (-1, 1)) 
    

    def source(self,X):
        r_vec_expanded = tf.expand_dims(X, axis=1)
        x_qs_expanded = tf.expand_dims(self.x_qs, axis=0)
        r_diff = r_vec_expanded - x_qs_expanded
        r2 = tf.reduce_sum(tf.square(r_diff), axis=2)
        delta = tf.exp((-1/(2*self.sigma**2))*r2)
        q_times_delta = self.qs*delta
        total_sum = tf.reduce_sum(q_times_delta, axis=1)
        normalizer = (1/((2*self.pi)**(3.0/2)*self.sigma**3))
        result = (-1/self.epsilon_1)*normalizer * total_sum
        result = tf.expand_dims(result, axis=1)
        return result
        
    @adim_function
    def G_Yukawa(self,X):
        r_vec_expanded = tf.expand_dims(X, axis=1)
        x_qs_expanded = tf.expand_dims(self.x_qs, axis=0)
        r_diff = r_vec_expanded - x_qs_expanded
        r = tf.sqrt(tf.reduce_sum(tf.square(r_diff), axis=2))
        q_over_r = self.qs*tf.exp(-self.kappa*r) / r
        total_sum = tf.reduce_sum(q_over_r, axis=1)
        result = (1 / (self.epsilon_2 * 4 * self.pi)) * total_sum
        result = tf.expand_dims(result, axis=1)
        return result


    def G_L(self,X,Xp):
        X_expanded = tf.expand_dims(X, axis=1)  # Shape: (n, 1, 3)
        Xp_expanded = tf.expand_dims(Xp, axis=0)  # Shape: (1, m, 3)
        r_diff = X_expanded - Xp_expanded  # Shape: (n, m, 3)
        r = tf.sqrt(tf.reduce_sum(tf.square(r_diff), axis=2))  # Shape: (n, m)
        over_r = 1 / r  # Shape: (n, m)
        green_function = (1 / (4 * self.pi)) * over_r  # Shape: (n, m)
        return green_function
    
    def G_Y(self,X,Xp):
        X_expanded = tf.expand_dims(X, axis=1)  # Shape: (n, 1, 3)
        Xp_expanded = tf.expand_dims(Xp, axis=0)  # Shape: (1, m, 3)
        r_diff = X_expanded - Xp_expanded  # Shape: (n, m, 3)
        r = tf.sqrt(tf.reduce_sum(tf.square(r_diff), axis=2))  # Shape: (n, m)
        e_over_r = tf.exp(-self.kappa*r) / r  # Shape: (n, m)
        green_function = (1 / (4 * self.pi)) * e_over_r  # Shape: (n, m)
        return green_function
    
    def dG_L(self,X,Xp,N):
        X_expanded = tf.expand_dims(X, axis=1)  # Shape: (n, 1, 3)
        Xp_expanded = tf.expand_dims(Xp, axis=0)  # Shape: (1, m, 3)
        n_expanded = tf.expand_dims(N, axis=0)  # Shape: (1, m, 3)
        r_diff = X_expanded - Xp_expanded  # Shape: (n, m, 3)
        r = tf.sqrt(tf.reduce_sum(tf.square(r_diff), axis=2))  # Shape: (n, m)
        dg_dr = (-1 / (4 * self.pi)) / (r**2)  # Shape: (n, m)
        grad_x = -1*dg_dr * r_diff[:, :, 0] / r  # Shape: (n, m)
        grad_y = -1*dg_dr * r_diff[:, :, 1] / r  # Shape: (n, m)
        grad_z = -1*dg_dr * r_diff[:, :, 2] / r  # Shape: (n, m)
        grad = tf.stack([grad_x, grad_y, grad_z], axis=2)  # Shape: (n, m, 3)
        dg_dn = tf.reduce_sum(grad * n_expanded, axis=2)  # Shape: (n, m)
        return dg_dn

    def dG_Y(self,X,Xp,N):
        X_expanded = tf.expand_dims(X, axis=1)  # Shape: (n, 1, 3)
        Xp_expanded = tf.expand_dims(Xp, axis=0)  # Shape: (1, m, 3)
        n_expanded = tf.expand_dims(N, axis=0)  # Shape: (1, m, 3)
        r_diff = X_expanded - Xp_expanded  # Shape: (n, m, 3)
        r = tf.sqrt(tf.reduce_sum(tf.square(r_diff), axis=2))  # Shape: (n, m)
        dg_dr = (-tf.exp(-self.kappa*r) / (4 * self.pi)) / (r)  *(-self.kappa/r - 1/r**2) # Shape: (n, m)
        grad_x = -1*dg_dr * r_diff[:, :, 0] / r  # Shape: (n, m)
        grad_y = -1*dg_dr * r_diff[:, :, 1] / r  # Shape: (n, m)
        grad_z = -1*dg_dr * r_diff[:, :, 2] / r  # Shape: (n, m)
        grad = tf.stack([grad_x, grad_y, grad_z], axis=2)  # Shape: (n, m, 3)
        dg_dn = tf.reduce_sum(grad * n_expanded, axis=2)  # Shape: (n, m)
        return dg_dn
        

    @adim_function
    def charges_Born_Ion(self,r, R, q):
        if R is None:
            R = self.mesh.R_mol
        epsilon_1 = self.epsilon_1
        epsilon_2 = self.epsilon_2
        kappa = self.kappa

        return (q / (4 * self.pi)) * (-1 / (epsilon_1 * R) + 1 / (epsilon_2 * (1 + kappa * R) * R))



    @adim_function
    def analytic_Born_Ion(self,r, R=None, index_q=0):
        if R is None:
            R = self.mesh.R_mol
        epsilon_1 = self.epsilon_1
        epsilon_2 = self.epsilon_2
        kappa = self.kappa
        q = self.qs[index_q]

        f_IN = lambda r: (q/(4*self.pi)) * ( - 1/(epsilon_1*R) + 1/(epsilon_2*(1+kappa*R)*R) )
        f_OUT = lambda r: (q/(4*self.pi)) * (np.exp(-kappa*(r-R))/(epsilon_2*(1+kappa*R)*r) - 1/(epsilon_1*r))

        y = np.piecewise(r, [r<=R, r>R], [f_IN, f_OUT])

        return y 



    @adim_function
    def analytic_Born_Ion_du(self,r):
        rI = self.mesh.R_mol
        epsilon_1 = self.epsilon_1
        epsilon_2 = self.epsilon_2
        kappa = self.kappa
        q = self.q_list[0].q

        f_IN = lambda r: (q/(4*self.pi)) * ( -1/(epsilon_1*r**2) )
        f_OUT = lambda r: (q/(4*self.pi)) * (np.exp(-kappa*(r-rI))/(epsilon_2*(1+kappa*rI))) * (-kappa/r - 1/r**2)

        y = np.piecewise(r, [r<=rI, r>rI], [f_IN, f_OUT])

        return y

    @adim_function
    def Spherical_Harmonics(self, x, flag, R=None, N=20):

        q = self.qs.numpy()
        xq = self.x_qs.numpy()
        E_1 = self.epsilon_1.numpy()
        E_2 = self.epsilon_2.numpy()
        kappa = self.kappa.numpy()
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
        
        return tf.constant(PHI, dtype=self.DTYPE) 

    @staticmethod
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

    
import matplotlib.pyplot as plt
from scipy.constants import pi

# Define a simple case with a punctual charge near the center
class SimpleTest(Solution_utils):
    def __init__(self):
        super().__init__()
        self.qs = tf.constant([1.0], dtype=tf.float32)  # Charge strength
        self.x_qs = tf.constant([[1.0, 1.0, 0.0]], dtype=tf.float32)  # Charge position
        self.epsilon_1 = tf.constant(80.0, dtype=tf.float32)  # Dielectric constant inside
        self.epsilon_2 = tf.constant(1.0, dtype=tf.float32)  # Dielectric constant outside
        self.kappa = tf.constant(0.0, dtype=tf.float32)  # Debye-Hückel parameter
        self.sigma = tf.constant(1.0, dtype=tf.float32)  # Gaussian width for charge distribution
        self.beta = tf.constant(1.0, dtype=tf.float32)  # Dimensionless factor
        self.DTYPE = tf.float32
        self.mesh = type('', (), {})()  # Dummy mesh object
        self.mesh.R_max_dist = 10.0  # Max radius

# Instantiate the test class
test = SimpleTest()

# Create a grid for the domain
domain_size = 10
num_points = 60
x = np.linspace(-domain_size, domain_size, num_points)
y = np.linspace(-domain_size, domain_size, num_points)
z = np.linspace(-domain_size, domain_size, num_points)
xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
points = np.stack((xv.flatten(), yv.flatten(), zv.flatten()), axis=-1)
points_tensor = tf.constant(points, dtype=tf.float32)

# Compute the 3D field using the spherical harmonics
phi_values = test.Spherical_Harmonics(points_tensor, flag='molecule', N=20)
phi_values_np = phi_values.numpy().reshape(num_points, num_points, num_points)

# Extract a 2D slice (e.g., z = 0 plane)
z_index = num_points // 2
phi_slice = phi_values_np[:, :, z_index]

# Plot the 2D slice
plt.figure(figsize=(8, 6))
plt.contourf(x, y, phi_slice, levels=50, cmap='viridis')
plt.colorbar(label='Potential (arbitrary units)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('2D Slice of the Potential Field (z = 0)')
plt.show()



















# try:
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D

#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot isosurface (requires proper packages like mayavi for detailed isosurfaces)
#     scat = ax.scatter(X, Y, Z, c=potentials.ravel(), cmap='viridis')
#     fig.colorbar(scat, ax=ax, label="Potential")

#     plt.title("3D Potential Field")
#     plt.show()
# except ImportError:
#     print("Matplotlib is not installed. Skipping visualization.")
