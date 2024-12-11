import numpy as np
from numpy import pi
from scipy import special
import pyvista as pv

def an_spherical_field(q, xq, E_1, E_2, E_0, R, N, points):
    xq = np.array(xq)
    q = np.array(q)
    points = np.array(points)
    Nq = len(q)
    Np = len(points)
    
    # Precompute spherical coords for charges
    rho_q = np.linalg.norm(xq, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        theta_q = np.arccos(np.where(rho_q > 1e-15, xq[:, 2] / rho_q, 1.0))
        phi_q = np.arctan2(xq[:, 1], xq[:, 0])
    phi_q[rho_q < 1e-15] = 0.0
    theta_q[rho_q < 1e-15] = 0.0
    
    # Precompute spherical harmonics for charges
    Ynm_charges = {}
    for n in range(N):
        for m in range(-n, n + 1):
            Ynm_charges[(n, m)] = special.sph_harm(m, n, phi_q, theta_q)
    
    PHI = np.zeros(Np, dtype=float)
    
    for iP in range(Np):
        x, y, z = points[iP]
        rr = np.sqrt(x * x + y * y + z * z)
        if rr < 1e-15:
            theta = 0.0
            phi = 0.0
        else:
            theta = np.arccos(z / rr)
            phi = np.arctan2(y, x)
        
        phi_field = 0.0 + 0.0j
        for nn in range(N):
            for mm in range(-nn, nn + 1):
                Ynm_field = special.sph_harm(mm, nn, phi, theta)
                cons1 = ((rr**nn) * (E_1 - E_2) * (nn + 1)) / (
                    E_1 * E_0 * (R**(2 * nn + 1)) * (E_1 * nn + E_2 * (nn + 1))
                )
                cons2 = 4 * pi / (2 * nn + 1)
                
                sum_over_charges = np.sum(q * (rho_q**nn) * np.conjugate(Ynm_charges[(nn, mm)]))
                phi_field += cons1 * cons2 * Ynm_field * sum_over_charges
        
        PHI[iP] = np.real(phi_field) / (4 * pi)
    
    return PHI

if __name__ == "__main__":
    # Sphere parameters
    R = 1.0             # radius
    E_0 = 8.854e-12      # vacuum permittivity
    E_1 = 80.0           # dielectric inside
    E_2 = 1.0            # dielectric outside
    N_expansion = 10     # expansion order
    
    # One charge inside the sphere
    q = [1.0e-19]
    xq = [[0.3 * R, 0.0, 0.0]]
    
    # Create a 3D grid of points
    ngrid = 50
    lin = np.linspace(-R, R, ngrid)
    X, Y, Z = np.meshgrid(lin, lin, lin)
    
    # Mask points inside the sphere
    inside_mask = (X**2 + Y**2 + Z**2) <= R**2
    # Extract only inside points
    X_in = X[inside_mask]
    Y_in = Y[inside_mask]
    Z_in = Z[inside_mask]
    points_in = np.vstack((X_in, Y_in, Z_in)).T
    
    # Compute potential inside the sphere
    PHI_in = an_spherical_field(q, xq, E_1, E_2, E_0, R, N_expansion, points_in)
    
    # Store results in a 3D array
    PHI_3D = np.full((ngrid, ngrid, ngrid), np.nan)
    PHI_3D[inside_mask] = PHI_in
    
    # Create a PyVista grid for visualization
    grid = pv.UniformGrid()
    grid.dimensions = PHI_3D.shape
    grid.origin = (-R, -R, -R)
    grid.spacing = (2 * R / (ngrid - 1), 2 * R / (ngrid - 1), 2 * R / (ngrid - 1))
    grid.point_data["Potential"] = PHI_3D.ravel(order="F")  # PyVista expects Fortran ordering

    # Visualize with PyVista
    pl = pv.Plotter()
    pl.add_volume(grid, scalars="Potential", cmap="viridis", opacity="sigmoid")
    pl.add_axes()
    pl.show()
