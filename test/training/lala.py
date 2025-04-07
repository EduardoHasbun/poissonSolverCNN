

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.special as sp















class InterfaceBoundaryLoss(nn.Module):
    def __init__(self, bound_weight, boundary, inner_mask, outer_mask, center, radius, points, e_in, e_out, dx, dy, dz):
        super().__init__()
        self.weight = bound_weight
        self.boundary = boundary
        self.inner_mask = inner_mask
        self.outer_mask = outer_mask
        self.points = points
        self.e_in = e_in
        self.e_out = e_out
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.center = center
        self.radius = radius

        # Get boundary indices
        boundary_indices = torch.nonzero(self.boundary, as_tuple=True)

        # Calculate the position of boundary nodes
        x_idx, y_idx, z_idx = boundary_indices[0], boundary_indices[1], boundary_indices[2]
        x_node, y_node, z_node = x_idx * self.dx, y_idx * self.dy, z_idx * self.dz

        # Calculate normal vectors for all boundary nodes
        normal_x = x_node - self.center[0]
        normal_y = y_node - self.center[1]
        normal_z = z_node - self.center[2]
        norm = torch.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
        normal_x /= norm
        normal_y /= norm
        normal_z /= norm

        self.x_idx, self.y_idx, self.z_idx = x_idx, y_idx, z_idx
        self.normal_x, self.normal_y, self.normal_z= normal_x, normal_y, normal_z











    def compute_gradients(self, output, data_norm = 1.):
        output = output / data_norm
        subdomain_in = output[:, 0, self.inner_mask]
        subdomain_out = output[:, 0, self.outer_mask]
        gradients_x_boundary_inner = torch.zeros_like(subdomain_in)
        gradients_x_boundary_outer = torch.zeros_like(subdomain_out)
        gradients_y_boundary_inner = torch.zeros_like(subdomain_in)
        gradients_y_boundary_outer = torch.zeros_like(subdomain_out)
        gradients_z_boundary_inner = torch.zeros_like(subdomain_in)
        gradients_z_boundary_outer = torch.zeros_like(subdomain_out)

        # Calculate the gradient for the x-direction
        left_inner = subdomain_in[:, 0, self.x_idx - 1, self.y_idx, self.z_idx]
        right_inner = subdomain_in[:, 0, self.x_idx + 1, self.y_idx, self.z_idx]
        left_outer = subdomain_out[:, 0, self.x_idx - 1, self.y_idx, self.z_idx]
        right_outer = subdomain_out[:, 0, self.x_idx + 1, self.y_idx, self.z_idx]

        gradients_x_boundary_inner[:, 0, self.x_idx, self.y_idx, self.z_idx] = torch.where(self.normal_x > 0, 
            (subdomain_in[:, 0, self.x_idx, self.y_idx, self.z_idx] - left_inner) / self.dx, 
            (right_inner - subdomain_in[:, 0, self.x_idx, self.y_idx, self.z_idx]) / self.dx)
        
        gradients_x_boundary_outer[:, 0, self.x_idx, self.y_idx, self.z_idx] = torch.where(self.normal_x > 0, 
            (-subdomain_out[:, 0, self.x_idx, self.y_idx, self.z_idx] + right_outer) / self.dx, 
            (subdomain_out[:, 0, self.x_idx, self.y_idx, self.z_idx] - left_outer) / self.dx)

        # Calculate the gradient for the y-direction
        above_inner = subdomain_in[:, 0, self.x_idx, self.y_idx + 1, self.z_idx]
        below_inner = subdomain_in[:, 0, self.x_idx, self.y_idx - 1, self.z_idx]
        above_outer = subdomain_out[:, 0, self.x_idx, self.y_idx + 1, self.z_idx]
        below_outer = subdomain_out[:, 0, self.x_idx, self.y_idx - 1, self.z_idx]

        gradients_y_boundary_inner[:, 0, self.x_idx, self.y_idx, self.z_idx] = torch.where(self.normal_y > 0, 
            (subdomain_in[:, 0, self.x_idx, self.y_idx, self.z_idx] - below_inner) / self.dy, 
            (above_inner - subdomain_in[:, 0, self.x_idx, self.y_idx, self.z_idx]) / self.dy)
        
        gradients_y_boundary_outer[:, 0, self.x_idx, self.y_idx, self.z_idx] = torch.where(self.normal_y > 0, 
            (-subdomain_out[:, 0, self.x_idx, self.y_idx, self.z_idx] + above_outer) / self.dy, 
            (subdomain_out[:, 0, self.x_idx, self.y_idx, self.z_idx] - below_outer) / self.dy)
        
        # Calculate the gradient for the z-direction
        front_inner = subdomain_in[:, 0, self.x_idx, self.y_idx, self.z_idx + 1]
        back_inner = subdomain_in[:, 0, self.x_idx, self.y_idx, self.z_idx - 1]
        front_outer = subdomain_out[:, 0, self.x_idx, self.y_idx, self.z_idx + 1]
        back_outer = subdomain_out[:, 0, self.x_idx, self.y_idx, self.z_idx - 1]

        gradients_z_boundary_inner[:, 0, self.x_idx, self.y_idx, self.z_idx] = torch.where(self.normal_z > 0, 
            (subdomain_in[:, 0, self.x_idx, self.y_idx, self.z_idx] - back_inner) / self.dy, 
            (front_inner - subdomain_in[:, 0, self.x_idx, self.y_idx, self.z_idx]) / self.dy)
        
        gradients_z_boundary_outer[:, 0, self.x_idx, self.y_idx, self.z_idx] = torch.where(self.normal_z > 0, 
            (-subdomain_out[:, 0, self.x_idx, self.y_idx, self.z_idx] + front_outer) / self.dz, 
            (subdomain_out[:, 0, self.x_idx, self.y_idx, self.z_idx] - back_outer) / self.dz)

        # Compute the normal derivatives
        normal_derivate_inner = gradients_x_boundary_inner[:, 0, self.boundary] * self.normal_x + \
            gradients_y_boundary_inner[:, 0, self.boundary] * self.normal_y + \
            gradients_z_boundary_inner[:, 0, self.boundary] * self.normal_z
        normal_derivate_outer = gradients_x_boundary_outer[:, 0, self.boundary] * self.normal_x +\
            gradients_y_boundary_outer[:, 0, self.boundary] * self.normal_y + \
            gradients_z_boundary_outer[:, 0, self.boundary] * self.normal_z

        return normal_derivate_inner, normal_derivate_outer





















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

    def forward(self, output, q, xq, data_norm = 1.):
        output = output / data_norm
        molecule = output[:, 0, self.inner_mask] + self.G(self.points, q, xq, self.e_in)[self.inner_mask][None, :]
        loss = F.mse_loss(molecule[:, 0, self.inner_mask], output[:, 0, self.outer_mask])
        normal_derivate_inner, normal_derivate_outer = self.compute_gradients(output, output, data_norm)
        return loss * self.weight