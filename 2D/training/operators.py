import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class LaplacianLoss(nn.Module):
    def __init__(self, cfg, lapl_weight):
        super().__init__()
        self.weight = lapl_weight
        self.dx = (cfg['globals']['xmax'] - cfg['globals']['xmin']) / cfg['globals']['nnx']
        self.dy = (cfg['globals']['ymax'] - cfg['globals']['ymin']) / cfg['globals']['nny']
        self.epsilon_inside = cfg['globals']['epsilon_inside']
        self.epsilon_outside = cfg['globals']['epsilon_outside']

    def forward(self, output, data=None, data_norm=1., mask = 1.):
        laplacian = lapl(output / data_norm, self.dx, self.dy, mask, self.epsilon_inside, self.epsilon_outside)
        loss = F.mse_loss(laplacian[:, 0, 1:-1, 1:-1], -data[:, 0, 1:-1, 1:-1]) * self.weight
        return loss

    
    
    
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

        self.x_idx, self.y_idx = x_idx, y_idx
        self.normal_x, self.normal_y = normal_x, normal_y


    def compute_gradients(self, subdomain_in, subdomain_out):
        gradients_x_boundary_inner = torch.zeros_like(subdomain_in)
        gradients_x_boundary_outer = torch.zeros_like(subdomain_out)
        gradients_y_boundary_inner = torch.zeros_like(subdomain_in)
        gradients_y_boundary_outer = torch.zeros_like(subdomain_out)

        # Calculate the gradient for the x-direction
        left_inner = subdomain_in[:, 0, self.x_idx - 1, self.y_idx]
        right_inner = subdomain_in[:, 0, self.x_idx + 1, self.y_idx]
        left_outer = subdomain_out[:, 0, self.x_idx - 1, self.y_idx]
        right_outer = subdomain_out[:, 0, self.x_idx + 1, self.y_idx]

        gradients_x_boundary_inner[:, 0, self.x_idx, self.y_idx] = torch.where(self.normal_x > 0, 
            (subdomain_in[:, 0, self.x_idx, self.y_idx] - left_inner) / self.dx, 
            (right_inner - subdomain_in[:, 0, self.x_idx, self.y_idx]) / self.dx)
        
        gradients_x_boundary_outer[:, 0, self.x_idx, self.y_idx] = torch.where(self.normal_x > 0, 
            (-subdomain_out[:, 0, self.x_idx, self.y_idx] + right_outer) / self.dx, 
            (subdomain_out[:, 0, self.x_idx, self.y_idx] - left_outer) / self.dx)

        # Calculate the gradient for the y-direction
        above_inner = subdomain_in[:, 0, self.x_idx, self.y_idx + 1]
        below_inner = subdomain_in[:, 0, self.x_idx, self.y_idx - 1]
        above_outer = subdomain_out[:, 0, self.x_idx, self.y_idx + 1]
        below_outer = subdomain_out[:, 0, self.x_idx, self.y_idx - 1]

        gradients_y_boundary_inner[:, 0, self.x_idx, self.y_idx] = torch.where(self.normal_y > 0, 
            (subdomain_in[:, 0, self.x_idx, self.y_idx] - below_inner) / self.dy, 
            (above_inner - subdomain_in[:, 0, self.x_idx, self.y_idx]) / self.dy)
        
        gradients_y_boundary_outer[:, 0, self.x_idx, self.y_idx] = torch.where(self.normal_y > 0, 
            (-subdomain_out[:, 0, self.x_idx, self.y_idx] + above_outer) / self.dy, 
            (subdomain_out[:, 0, self.x_idx, self.y_idx] - below_outer) / self.dy)

        # Compute the normal derivatives
        normal_derivate_inner = gradients_x_boundary_inner[:, 0, self.boundary] * self.normal_x + gradients_y_boundary_inner[:, 0, self.boundary] * self.normal_y
        normal_derivate_outer = gradients_x_boundary_outer[:, 0, self.boundary] * self.normal_x + gradients_y_boundary_outer[:, 0, self.boundary] * self.normal_y

        return normal_derivate_inner, normal_derivate_outer



    def forward(self, subdomain_in, subdomain_out, data_norm = 1.):
        subdomain_in_scaled = subdomain_in / data_norm
        subdomain_out_scaled = subdomain_out / data_norm
        loss = F.mse_loss(subdomain_in_scaled[:, 0, self.boundary], subdomain_out_scaled[:, 0, self.boundary])
        normal_derivate_inner, normal_derivate_outer = self.compute_gradients(subdomain_in_scaled, subdomain_out_scaled)
        loss += F.mse_loss((self.e_in * normal_derivate_inner), (self.e_out * normal_derivate_outer))
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

        

def lapl(field, dx, dy, interface_mask, epsilon_in, epsilon_out):
    batch_size, _, h, w = field.shape
    laplacian = torch.zeros_like(field).type(field.type())

    # Get epsilon at grid points
    epsilon = get_epsilon_tensor(field.shape, interface_mask, epsilon_in, epsilon_out)
    epsilon = epsilon.unsqueeze(0).unsqueeze(0)
    epsilon = epsilon.expand(batch_size, 1, h, w)

    # Compute epsilon at cell faces using harmonic mean
    epsilon_x_ip = harmonic_mean(epsilon[:, :, :, :-1], epsilon[:, :, :, 1:])
    epsilon_y_ip = harmonic_mean(epsilon[:, :, :-1, :], epsilon[:, :, 1:, :])

    # Compute flux differences
    flux_x_ip = epsilon_x_ip * (field[:, :, :, 1:] - field[:, :, :, :-1]) / dx
    flux_y_ip = epsilon_y_ip * (field[:, :, 1:, :] - field[:, :, :-1, :]) / dy

    # Initialize divergence
    divergence = torch.zeros_like(field[:, 0, :, :])

    # Corrected divergence calculation
    divergence[:, 1:-1, 1:-1] = (
        (flux_x_ip[:, 0, 1:-1, 1:] - flux_x_ip[:, 0, 1:-1, 0:-1]) / dx +
        (flux_y_ip[:, 0, 1:, 1:-1] - flux_y_ip[:, 0, 0:-1, 1:-1]) / dy
    )

    laplacian[:, 0, :, :] = divergence

    return laplacian





def ratio_potrhs(alpha, Lx, Ly):
    return alpha / (np.pi**2 / 4)**2 / (1 / Lx**2 + 1 / Ly**2)


def get_epsilon_tensor(field_shape, interface_mask, epsilon_in, epsilon_out):
    epsilon = torch.zeros(field_shape[2:], device=interface_mask.device)
    epsilon[interface_mask] = epsilon_in
    epsilon[~interface_mask] = epsilon_out
    return epsilon


def harmonic_mean(a, b):
    return 2 * a * b / (a + b)



