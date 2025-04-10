import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.special as sp


class LaplacianLossInterface(nn.Module):
    def __init__(self, cfg, lapl_weight, inner_mask, outer_mask, points):
        super().__init__()
        self.cfg = cfg
        self.weight = lapl_weight
        self.dx = (cfg['globals']['xmax'] - cfg['globals']['xmin']) / cfg['globals']['nnx']
        self.dy = (cfg['globals']['ymax'] - cfg['globals']['ymin']) / cfg['globals']['nny']
        self.dz = (cfg['globals']['zmax'] - cfg['globals']['zmin']) / cfg['globals']['nnz'] 
        self.epsilon_inside = cfg['globals']['epsilon_inside']
        self.epsilon_outside = cfg['globals']['epsilon_outside']
        self.inner_mask = inner_mask
        self.outer_mask = outer_mask
        # Physical constants
        self.e = 1.602176634e-19       
        self.kB = 1.380649e-23         
        self.eps_0 = 8.854187817e-12 
        self.points = points
    

    def k_w(self, points, q, xq, e_in, T=300):
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32, device=q.device)

        epsilon = e_in * self.eps_0

        # Mask out zero charges
        q = torch.as_tensor(q, device=points.device)
        xq = torch.as_tensor(xq, device=points.device)
        mask = ~torch.isclose(q, torch.tensor(0.0, dtype=q.dtype, device=q.device))
        q = q[mask]        
        xq = xq[mask]       

        if len(q) == 0:
            return torch.zeros(points.shape[0], device=points.device)  

        # Compute distances
        r = torch.linalg.norm(points[:, None, :] - xq[None, :, :], dim=2)  

        sigma = 1e-10  
        z = torch.round(q / self.e).int()
        z2_density = torch.sum(
            (z**2)[None, :] * torch.exp(-r**2 / (2 * sigma**2)) / ((2 * torch.pi * sigma**2)**1.5),
            dim=1
        )

        kappa_squared = (self.e**2 / (epsilon * self.kB * T)) * z2_density
        kappa = torch.sqrt(kappa_squared)
        return kappa



    def forward(self, output, q, xq, data_norm = 1.):
        laplacian = lapl_interface(output / data_norm, self.dx, self.dy, self.dz, self.inner_mask, self.epsilon_inside, self.epsilon_outside)
        k_w = self.k_w(self.points, q, xq, self.epsilon_inside).reshape(self.cfg['globals']['nnx'],
                  self.cfg['globals']['nny'],
                  self.cfg['globals']['nnz'])
        k_w = k_w.unsqueeze(0).unsqueeze(0)
        k_w = k_w.expand(output.shape[0], -1, -1, -1, -1)
        loss = F.mse_loss(laplacian[:, 0, self.inner_mask], torch.zeros_like(laplacian[:, 0, self.inner_mask])) 
        loss += F.mse_loss(laplacian[:, 0, self.outer_mask], k_w[:, 0, self.outer_mask] ** 2 * output[:, 0, self.outer_mask])
        return loss * self.weight


class DirichletBoundaryLoss(nn.Module):
    def __init__(self, bound_weight):
        super().__init__()
        self.weight = bound_weight

    def forward(self, output):
        bnd_loss = F.mse_loss(output[:, 0, -1, :, :], torch.zeros_like(output[:, 0, -1, :, :]))
        bnd_loss += F.mse_loss(output[:, 0, :, 0, :], torch.zeros_like(output[:, 0, :, 0, :]))
        bnd_loss += F.mse_loss(output[:, 0, :, -1, :], torch.zeros_like(output[:, 0, :, -1, :]))
        bnd_loss += F.mse_loss(output[:, 0, 0, :, :], torch.zeros_like(output[:, 0, 0, :, :]))
        bnd_loss += F.mse_loss(output[:, 0, :, :, 0], torch.zeros_like(output[:, 0, :, :, 0]))
        bnd_loss += F.mse_loss(output[:, 0, :, :, -1], torch.zeros_like(output[:, 0, :, :, -1]))
        return bnd_loss * self.weight
    

class InterfaceBoundaryLoss(nn.Module):
    def __init__(self, cfg, bound_weight, boundary, inner_mask, outer_mask, center, radius, points, e_in, e_out, dx, dy, dz):
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
        self.cfg = cfg

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
        subdomain_in = output
        subdomain_out = output
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
            (subdomain_in[:, 0, self.x_idx, self.y_idx, self.z_idx] - back_inner) / self.dz, 
            (front_inner - subdomain_in[:, 0, self.x_idx, self.y_idx, self.z_idx]) / self.dz)
        
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
    
    
    def G(self, q, xq):
        if not isinstance(self.points, torch.Tensor):
            points = torch.tensor(self.points, dtype=torch.float32, device=q.device)
        # Mask out zero charges
        q_mask = ~torch.isclose(q, torch.tensor(0.0, dtype=q.dtype, device=q.device))
        q = q[q_mask]           
        xq = xq[q_mask]          

        r_vec_expanded = points.unsqueeze(1)        
        x_qs_expanded = xq.unsqueeze(0)         
        r_diff = r_vec_expanded - x_qs_expanded 
        r = torch.norm(r_diff, dim=2)          
        r[r == 0] = torch.finfo(torch.float32).eps  

        q_over_r = q / r                       
        total_sum = torch.sum(q_over_r, dim=1) 
        result = (1 / (self.e_in * 4 * torch.pi)) * total_sum
        return result



    def grad_G(self, q, xq):
        if not isinstance(self.points, torch.Tensor):
            points = torch.tensor(self.points, dtype=torch.float32, device=q.device)
        # Mask out zero charges
        q_mask = ~torch.isclose(q, torch.tensor(0.0, dtype=q.dtype, device=q.device))
        q = q[q_mask]         
        xq = xq[q_mask]      

        r_vec_expanded = points.unsqueeze(1)       
        x_qs_expanded = xq.unsqueeze(0)      
        r_diff = r_vec_expanded - x_qs_expanded   

        r_squared = torch.sum(r_diff**2, dim=2)   
        r_cubed = r_squared.pow(1.5)
        r_cubed[r_cubed == 0] = torch.finfo(torch.float32).eps

        coef = -q / r_cubed                      
        grad = coef.unsqueeze(2) * r_diff      
        total_grad = torch.sum(grad, dim=1)     
        return total_grad / (self.e_in * 4 * torch.pi)



    def forward(self, output, q, xq, data_norm = 1.):
        output = output / data_norm
        g_c = self.G(q, xq).reshape(self.cfg['globals']['nnx'],
                    self.cfg['globals']['nny'],
                    self.cfg['globals']['nnz'])
        g_c = g_c.unsqueeze(0).unsqueeze(0)
        g_c = g_c.expand(output.shape[0], -1, -1, -1, -1)
        molecule = output + g_c
        loss = F.mse_loss(molecule[:, 0, self.boundary], output[:, 0, self.boundary])
        normal_derivate_inner, normal_derivate_outer = self.compute_gradients(output, data_norm)
        gc_grad = self.grad_G(q, xq).reshape(self.cfg['globals']['nnx'],
                    self.cfg['globals']['nny'], 
                    self.cfg['globals']['nnz'], 3)

        gc_grad_boundary = gc_grad[self.x_idx, self.y_idx, self.z_idx, :]  
        gc_normal_deriv = (gc_grad_boundary[:, 0] * self.normal_x
                        + gc_grad_boundary[:, 1] * self.normal_y
                        + gc_grad_boundary[:, 2] * self.normal_z)
        loss += F.mse_loss(self.e_in * (normal_derivate_inner + gc_normal_deriv.unsqueeze(0)),
                     self.e_out * normal_derivate_outer)

        return loss * self.weight


def lapl_interface(field, dx, dy, dz, interface_mask, epsilon_in, epsilon_out):
    batch_size, _, h, w, l = field.shape
    laplacian = torch.zeros_like(field).type(field.type())

    # Get epsilon at grid points
    epsilon = get_epsilon_tensor(field.shape, interface_mask, epsilon_in, epsilon_out)  
    epsilon = epsilon.unsqueeze(0).unsqueeze(0)  
    epsilon = epsilon.expand(batch_size, 1, h, w, l)  

    # Compute epsilon at cell faces using harmonic mean
    epsilon_x_ip = harmonic_mean(epsilon[:, :, :, :, :-1], epsilon[:, :, :, :, 1:])  
    epsilon_y_ip = harmonic_mean(epsilon[:, :, :, :-1, :], epsilon[:, :, :, 1:, :])  
    epsilon_z_ip = harmonic_mean(epsilon[:, :, :-1, :, :], epsilon[:, :, 1:, :, :]) 

    # Compute flux differences in x-direction
    flux_x_ip = epsilon_x_ip * (field[:, :, :, :, 1:] - field[:, :, :, :, :-1]) / dx 

    # Compute flux differences in y-direction
    flux_y_ip = epsilon_y_ip * (field[:, :, :, 1:, :] - field[:, :, :, :-1, :]) / dy  

    # Compute flux differences in z-direction
    flux_z_ip = epsilon_z_ip * (field[:, :, 1:, :, :] - field[:, :, :-1, :, :]) / dz 
    # Initialize divergence
    divergence = torch.zeros_like(field[:, 0, :, :, :]) 

    # Divergence calculation 
    divergence[:, 1:-1, 1:-1, 1:-1] = (
        (flux_x_ip[:, 0, 1:-1, 1:-1, 1:] - flux_x_ip[:, 0, 1:-1, 1:-1, :-1]) / dx +
        (flux_y_ip[:, 0, 1:-1, 1:, 1:-1] - flux_y_ip[:, 0, 1:-1, :-1, 1:-1]) / dy +
        (flux_z_ip[:, 0, 1:, 1:-1, 1:-1] - flux_z_ip[:, 0, :-1, 1:-1, 1:-1]) / dz
    )

    laplacian[:, 0, :, :, :] = divergence

    return laplacian


def get_epsilon_tensor(field_shape, interface_mask, epsilon_in, epsilon_out):
    epsilon = torch.zeros(field_shape[2:], device=interface_mask.device)
    epsilon[interface_mask] = epsilon_in
    epsilon[~interface_mask] = epsilon_out
    return epsilon


def harmonic_mean(a, b):
    return 2 * a * b / (a + b)
    

def ratio_potrhs(alpha, Lx, Ly, Lz):
    return alpha / (np.pi**2 / 8)**2 / (1 / Lx**2 + 1 / Ly**2 + 1 / Lz**2)
















