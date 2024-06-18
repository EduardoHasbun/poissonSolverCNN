import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
    def __init__(self, bound_weight, interface_mask, epsilon_1, epsilon_2, dx, dy, interface_center):
        super().__init__()
        self.weight = bound_weight
        self.interface_mask = interface_mask
        self.epsilon_1 = epsilon_1
        self.epsilon_2 = epsilon_2
        self.dx = dx
        self.dy = dy
        self.interface_center = interface_center

        # Compute the interface boundary coordinates once during initialization
        self.interface_boundary_coords = torch.nonzero(interface_mask, as_tuple=False)

    def forward(self, output_in, output_out):
        # Continuity of potential
        bnd_loss_potential = F.mse_loss(output_in[:, 0, self.interface_mask], output_out[:, 0, self.interface_mask])

        # Compute normal derivatives at the interface
        dphi1_dn = self.compute_normal_derivative(output_in[:, 0])
        dphi2_dn = self.compute_normal_derivative(output_out[:, 0])

        # Continuity of normal component of displacement field
        normal_derivative_mismatch = self.epsilon_1 * dphi1_dn - self.epsilon_2 * dphi2_dn
        bnd_loss_derivative = torch.mean(normal_derivative_mismatch[self.interface_mask] ** 2)

        # Total boundary loss
        total_loss = self.weight * (bnd_loss_potential + bnd_loss_derivative)
        return total_loss

    def compute_normal_derivative(self, phi):
        # Get the coordinates of the boundary points
        boundary_coords = self.interface_boundary_coords

        normal_x = (boundary_coords[:, 1].float() - self.interface_center[0])
        normal_y = (boundary_coords[:, 0].float() - self.interface_center[1])
        norm = torch.sqrt(normal_x**2 + normal_y**2)
        normal_x /= norm
        normal_y /= norm

        # Compute derivatives using central differences
        dphi_dx = (phi[:, 2:] - phi[:, :-2]) / (2 * self.dx)
        dphi_dy = (phi[2:, :] - phi[:-2, :]) / (2 * self.dy)

        # Pad derivatives to match original phi shape
        dphi_dx = F.pad(dphi_dx, (0, 0, 1, 1), mode='replicate')
        dphi_dy = F.pad(dphi_dy, (1, 1, 0, 0), mode='replicate')

        # Initialize normal derivatives with zeros
        dphi_dn = torch.zeros_like(phi)

        # Assign normal derivatives at boundary coordinates
        for i, coord in enumerate(boundary_coords):
            dphi_dn[coord[0], coord[1]] = normal_x[i] * dphi_dx[coord[0], coord[1]] + normal_y[i] * dphi_dy[coord[0], coord[1]]

        return dphi_dn

    


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
        (1 - b) * ((field[:, 0, 2:, 1:-1] + field[:, 0, :-2, 1:-1] - 2 * field[:, 0, 1:-1, 1:-1]) / dy**2 +
        (field[:, 0, 1:-1, 2:] + field[:, 0, 1:-1, :-2] - 2 * field[:, 0, 1:-1, 1:-1]) / dx**2) + \
        b * (field[:, 0, 2:, 2:] + field[:, 0, 2:, :-2] + field[:, 0, :-2, :-2] + field[:, 0, :-2, 2:] - 4 * field[:, 0, 1:-1, 1:-1]) \
        / (2 * dx**2)

    
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
