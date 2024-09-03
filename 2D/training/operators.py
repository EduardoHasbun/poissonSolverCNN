import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class LaplacianLoss(nn.Module):
    def __init__(self, cfg, lapl_weight, e_in = 1, e_out = 1, interface = 1):
        super().__init__()
        self.weight = lapl_weight
        xmin, xmax, ymin, ymax, nnx, nny = cfg['globals']['xmin'], cfg['globals']['xmax'],\
            cfg['globals']['ymin'], cfg['globals']['ymax'], cfg['globals']['nnx'], cfg['globals']['nny']
        self.Lx = xmax-xmin
        self.Ly = ymax-ymin
        self.dx = self.Lx/nnx
        self.dy = self.Ly/nny
        self.epsilon_inside = e_in
        self.epsilon_outside = e_out
        self.interface = interface
    def forward(self, output, data=None, data_norm=1.):
        laplacian = lapl(output / data_norm, self.dx, self.dy, self.interface, self.epsilon_inside, self.epsilon_outside)
        return self.Lx**2 * self.Ly**2 * F.mse_loss(laplacian[:, 0, 1:-1, 1:-1], - data[:, 0, 1:-1, 1:-1]) * self.weight
    
    
    
class DirichletBoundaryLoss(nn.Module):
    def __init__(self, bound_weight):
        super().__init__()
        self.weight = bound_weight
        self.base_weight = self.weight

    def forward(self, output):
        bnd_loss = F.mse_loss(output[:, 0, -1, :], torch.zeros_like(output[:, 0, -1, :]))
        bnd_loss += F.mse_loss(output[:, 0, :, 0], torch.zeros_like(output[:, 0, :, 0]))
        bnd_loss += F.mse_loss(output[:, 0, :, -1], torch.zeros_like(output[:, 0, :, -1]))
        bnd_loss += F.mse_loss(output[:, 0, 0, :], torch.zeros_like(output[:, 0, 0, :]))
        return bnd_loss * self.weight
    



class InterfaceBoundaryLoss(nn.Module):
    def __init__(self, bound_weight, boundary, center, radius, e_in, e_out, dx, dy):
        super().__init__()
        self.weight = bound_weight
        self.boundary = boundary
        self.e_in = e_in
        self.e_out = e_out
        self.dx = dx
        self.dy = dy
        self.center = center
        self.radius = radius


    def compute_gradients(self, subdomain_in, subdomain_out):
        gradients_x_boundary_inner = torch.zeros_like(subdomain_in)
        gradients_x_boundary_outer = torch.zeros_like(subdomain_out)
        gradients_y_boundary_inner = torch.zeros_like(subdomain_in)
        gradients_y_boundary_outer = torch.zeros_like(subdomain_out)

        # Get boundary indices
        boundary_indices = torch.nonzero(self.boundary, as_tuple=True)

        # Calculate the position of boundary nodes
        x_idx, y_idx = boundary_indices[0], boundary_indices[1]
        x_node, y_node = x_idx * self.dx, y_idx * self.dy

        # Calculate normal vectors for all boundary nodes
        normal_x = x_node - self.center[0]
        normal_y = y_node - self.center[1]
        norm = torch.sqrt(normal_x**2 + normal_y**2)
        normal_x /= norm
        normal_y /= norm

        # Calculate the gradient for the x-direction
        left_inner = subdomain_in[:, 0, x_idx - 1, y_idx]
        right_inner = subdomain_in[:, 0, x_idx + 1, y_idx]
        left_outer = subdomain_out[:, 0, x_idx - 1, y_idx]
        right_outer = subdomain_out[:, 0, x_idx + 1, y_idx]

        gradients_x_boundary_inner[:, 0, x_idx, y_idx] = torch.where(normal_x > 0, 
            (subdomain_in[:, 0, x_idx, y_idx] - left_inner) / self.dx, 
            (right_inner - subdomain_in[:, 0, x_idx, y_idx]) / self.dx
        )
        
        gradients_x_boundary_outer[:, 0, x_idx, y_idx] = torch.where(normal_x > 0, 
            (-subdomain_out[:, 0, x_idx, y_idx] + right_outer) / self.dx, 
            (subdomain_out[:, 0, x_idx, y_idx] - left_outer) / self.dx
        )

        # Calculate the gradient for the y-direction
        above_inner = subdomain_in[:, 0, x_idx, y_idx + 1]
        below_inner = subdomain_in[:, 0, x_idx, y_idx - 1]
        above_outer = subdomain_out[:, 0, x_idx, y_idx + 1]
        below_outer = subdomain_out[:, 0, x_idx, y_idx - 1]

        gradients_y_boundary_inner[:, 0, x_idx, y_idx] = torch.where(normal_y > 0, 
            (subdomain_in[:, 0, x_idx, y_idx] - below_inner) / self.dy, 
            (above_inner - subdomain_in[:, 0, x_idx, y_idx]) / self.dy
        )
        
        gradients_y_boundary_outer[:, 0, x_idx, y_idx] = torch.where(normal_y > 0, 
            (-subdomain_out[:, 0, x_idx, y_idx] + above_outer) / self.dy, 
            (subdomain_out[:, 0, x_idx, y_idx] - below_outer) / self.dy
        )

        # Compute the normal derivatives
        normal_derivate_inner = gradients_x_boundary_inner * normal_x + gradients_y_boundary_inner * normal_y
        normal_derivate_outer = gradients_x_boundary_outer * normal_x + gradients_y_boundary_outer * normal_y

        return normal_derivate_inner, normal_derivate_outer



    def forward(self, subdomain_in, subdomain_out):
        loss = F.mse_loss(subdomain_in[:, 0, self.boundary], subdomain_out[:, 0, self.boundary])
        normal_derivate_inner, normal_derivate_outer = self.compute_gradients(subdomain_in, subdomain_out)
        norm_d_in, norm_d_out = normal_derivate_inner[:, 0, self.boundary], normal_derivate_outer[:, 0, self.boundary]
        loss += F.mse_loss((norm_d_in), (norm_d_out))
        return loss * self.weight




class DirichletBoundaryLossFunction(nn.Module):
    def __init__(self, bound_weight, xmin, xmax, ymin, ymax, nnx, nny):
        super().__init__()
        self.weight = bound_weight
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        x = torch.linspace(self.xmin, self.xmax, nnx)
        y = torch.linspace(self.ymin, self.ymax, nny)
        X, Y = torch.meshgrid(x, y)
        
        def function2solve(x, y):
            return torch.pow(x,3) + torch.pow(y,3)
        
        domain = function2solve(X, Y) 
        self.domain = domain.unsqueeze(0)

    def forward(self, output, data_norm = 1.):
        batch, _, _, _ = output.size()
        domain = self.domain.repeat(batch, 1, 1, 1)
        output /= data_norm
        bnd_loss =  F.mse_loss(output[:, 0, -1, :], domain[:, 0, -1, :])
        bnd_loss += F.mse_loss(output[:, 0, :, 0], domain[:, 0, :, 0])
        bnd_loss += F.mse_loss(output[:, 0, :, -1], domain[:, 0, :, -1])
        bnd_loss += F.mse_loss(output[:, 0, 0, :], domain[:, 0, 0, :])
        return (bnd_loss * self.weight)

        

def lapl(field, dx, dy, interface, epsilon_in, epsilon_out, b=0):

    # Create laplacian tensor with shape (batch_size, 1, h, w)
    laplacian = torch.zeros_like(field).type(field.type())

    # Check sizes
    assert field.dim() == 4 and laplacian.dim() == 4, 'Dimension mismatch'

    assert field.is_contiguous() and laplacian.is_contiguous(), 'Input is not contiguous'

    laplacian[:, 0, 1:-1, 1:-1] = \
        (field[:, 0, 2:, 1:-1] + field[:, 0, :-2, 1:-1] - 2 * field[:, 0, 1:-1, 1:-1]) / dy**2 + \
        (field[:, 0, 1:-1, 2:] + field[:, 0, 1:-1, :-2] - 2 * field[:, 0, 1:-1, 1:-1]) / dx**2 

    
    # laplacian[:, 0, 0, 1:-1] = \
    #         (2 * field[:, 0, 0, 1:-1] - 5 * field[:, 0, 1, 1:-1] + 4 * field[:, 0, 2, 1:-1] - field[:, 0, 3, 1:-1]) / dy**2 + \
    #         (field[:, 0, 0, 2:] + field[:, 0, 0, :-2] - 2 * field[:, 0, 0, 1:-1]) / dx**2
        
    # laplacian[:, 0, -1, 1:-1] = \
    #     (2 * field[:, 0, -1, 1:-1] - 5 * field[:, 0, -2, 1:-1] + 4 * field[:, 0, -3, 1:-1] - field[:, 0, -4, 1:-1]) / dy**2 + \
    #     (field[:, 0, -1, 2:] + field[:, 0, -1, :-2] - 2 * field[:, 0, -1, 1:-1]) / dx**2
    # laplacian[:, 0, 1:-1, 0] = \
    #     (field[:, 0, 2:, 0] + field[:, 0, :-2, 0] - 2 * field[:, 0, 1:-1, 0]) / dy**2 + \
    #     (2 * field[:, 0, 1:-1, 0] - 5 * field[:, 0, 1:-1, 1] + 4 * field[:, 0, 1:-1, 2] - field[:, 0, 1:-1, 3]) / dx**2
    # laplacian[:, 0, 1:-1, -1] = \
    #     (field[:, 0, 2:, -1] + field[:, 0, :-2, -1] - 2 * field[:, 0, 1:-1, -1]) / dy**2 + \
    #     (2 * field[:, 0, 1:-1, -1] - 5 * field[:, 0, 1:-1, -2] + 4 * field[:, 0, 1:-1, -3] - field[:, 0, 1:-1, -4]) / dx**2

    
    # laplacian[:, 0, 0, 0] = \
    #         (2 * field[:, 0, 0, 0] - 5 * field[:, 0, 1, 0] + 4 * field[:, 0, 2, 0] - field[:, 0, 3, 0]) / dy**2 + \
    #         (2 * field[:, 0, 0, 0] - 5 * field[:, 0, 0, 1] + 4 * field[:, 0, 0, 2] - field[:, 0, 0, 3]) / dx**2
    # laplacian[:, 0, 0, -1] = \
    #         (2 * field[:, 0, 0, -1] - 5 * field[:, 0, 1, -1] + 4 * field[:, 0, 2, -1] - field[:, 0, 3, -1]) / dy**2 + \
    #         (2 * field[:, 0, 0, -1] - 5 * field[:, 0, 0, -2] + 4 * field[:, 0, 0, -3] - field[:, 0, 0, -4]) / dx**2

    # laplacian[:, 0, -1, 0] = \
    #     (2 * field[:, 0, -1, 0] - 5 * field[:, 0, -2, 0] + 4 * field[:, 0, -3, 0] - field[:, 0, -4, 0]) / dy**2 + \
    #     (2 * field[:, 0, -1, 0] - 5 * field[:, 0, -1, 1] + 4 * field[:, 0, -1, 2] - field[:, 0, -1, 3]) / dx**2
    # laplacian[:, 0, -1, -1] = \
    #     (2 * field[:, 0, -1, -1] - 5 * field[:, 0, -2, -1] + 4 * field[:, 0, -3, -1] - field[:, 0, -4, -1]) / dy**2 + \
    #     (2 * field[:, 0, 0, -1] - 5 * field[:, 0, 0, -2] + 4 * field[:, 0, 0, -3] - field[:, 0, 0, -4]) / dx**2

    # laplacian[:, 0, interface] *= epsilon_in
    # laplacian[:, 0, ~interface] *= epsilon_out

    return laplacian



def ratio_potrhs(alpha, Lx, Ly):
    return alpha / (np.pi**2 / 4)**2 / (1 / Lx**2 + 1 / Ly**2)




# class InterfaceBoundaryLoss(nn.Module):
#     def __init__(self, bound_weight, interface_mask, epsilon_1, epsilon_2, dx, dy, interface_center):
#         super().__init__()
#         self.weight = bound_weight
#         self.interface_mask = interface_mask
#         self.epsilon_1 = epsilon_1
#         self.epsilon_2 = epsilon_2
#         self.dx = dx
#         self.dy = dy
#         self.interface_center = interface_center

#     def forward(self, output_in, output_out, data_norm):
#         # Continuity of potential
#         bnd_loss_potential = F.mse_loss(output_in[:, 0, self.interface_mask], output_out[:, 0, self.interface_mask])
#         total_loss = self.weight * (bnd_loss_potential)
#         return total_loss