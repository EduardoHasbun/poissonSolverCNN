import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LaplacianLossInterface(nn.Module):
    def __init__(self, cfg, lapl_weight):
        super().__init__()
        self.weight = lapl_weight
        self.dx = (cfg['globals']['xmax'] - cfg['globals']['xmin']) / cfg['globals']['nnx']
        self.dy = (cfg['globals']['ymax'] - cfg['globals']['ymin']) / cfg['globals']['nny']
        self.dz = (cfg['globals']['zmax'] - cfg['globals']['zmin']) / cfg['globals']['nnz'] 
        self.epsilon_inside = cfg['globals']['epsilon_inside']
        self.epsilon_outside = cfg['globals']['epsilon_outside']
        

    def forward(self, output, data = None, data_norm = 1., mask = 1.):
        laplacian = lapl_interface(output / data_norm, self.dx, self.dy, mask, self.epsilon_inside, self.epsilon_outside)
        loss = F.mse_loss(laplacian[:, 0, mask], data[:, 0, mask]) * self.weight
        return loss
    

class LaplacianLoss(nn.Module):
    def __init__(self, cfg, lapl_weight, e_in = 1, e_out = 1, interface = 1):
        super().__init__()
        self.weight = lapl_weight
        self.dx = (cfg['globals']['xmax'] - cfg['globals']['xmin']) / cfg['globals']['nnx']
        self.dy = (cfg['globals']['ymax'] - cfg['globals']['ymin']) / cfg['globals']['nny']
        self.dz = (cfg['globals']['zmax'] - cfg['globals']['zmin']) / cfg['globals']['nnz']
    def forward(self, output, data=None, data_norm=1.):
        laplacian = lapl(output / data_norm, self.dx, self.dy)
        return self.Lx**2 * self.Ly**2 * F.mse_loss(laplacian[:, 0, 1:-1, 1:-1], - data[:, 0, 1:-1, 1:-1]) * self.weight
    

    
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
    def __init__(self, bound_weight, boundary, center, radius, e_in, e_out, dx, dy, dz):
        super().__init__()
        self.weight = bound_weight
        self.boundary = boundary
        self.e_in = e_in
        self.e_out = e_out
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.center = center
        self.radius = radius

        # Get boudnary indices
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
        self.normal_x, self.normal_y, self.normal_z = normal_x, normal_y, normal_z

    def compute_gradients(self,  subdomain_in, subdomain_out):
        gradients_x_boundary_inner = torch.zeros_like(subdomain_in)
        gradients_x_boundary_outer = torch.zeros_like(subdomain_out)
        gradients_y_boundary_inner = torch.zeros_like(subdomain_in)
        gradients_y_boundary_outer = torch.zeros_like(subdomain_out)
        gradients_z_boudnary_inner = torch.zeros_like(subdomain_in)
        gradients_z_boudnary_outer = torch.zeros_like(subdomain_out)

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

        gradients_z_boudnary_inner[:, 0, self.x_idx, self.y_idx, self.z_idx] = torch.where(self.normal_z > 0,
            (-subdomain_in[:, 0, self.x_idx, self.y_idx, self.z_idx] + front_inner) / self.dz,
            (subdomain_in[:, 0, self.x_idx, self.y_idx, self.z_idx] - back_inner) / self.dz)
        
        gradients_z_boudnary_outer[:, 0, self.x_idx, self.y_idx, self.z_idx] = torch.where(self.normal_z < 0,
            (-subdomain_out[:, 0, self.x_idx, self.y_idx, self.z_idx] + front_outer) / self.dz,
            (subdomain_out[:, 0, self.x_idx, self.y_idx, self.z_idx] - back_outer) / self.dz)

        # Compute the normal derivatives
        normal_derivate_inner = gradients_x_boundary_inner[:, 0, self.boundary] * self.normal_x + gradients_y_boundary_inner[:, 0, self.boundary] * self.normal_y + gradients_z_boudnary_inner[:, 0, self.boundary] * self.normal_z
        normal_derivate_outer = gradients_x_boundary_outer[:, 0, self.boundary] * self.normal_x + gradients_y_boundary_outer[:, 0, self.boundary] * self.normal_y + gradients_z_boudnary_outer[:, 0, self.boundary] * self.normal_z

        return normal_derivate_inner, normal_derivate_outer


    def forward(self, subdomain_in, subdomain_out, data_norm = 1.0):
        subdomain_in_scaled = subdomain_in / data_norm 
        subdomain_out_scaled = subdomain_out / data_norm
        loss = F.mse_loss(subdomain_in_scaled[:, 0, self.boundary], subdomain_out_scaled[:, 0, self.boundary])
        normal_derivate_inner, normal_derivate_outer = self.compute_gradients(subdomain_in_scaled, subdomain_out_scaled)
        loss += F.mse_loss((normal_derivate_inner), (normal_derivate_outer))
        return loss * self.weight
    

class DirichletBoundaryLossFunction(nn.Module):
    def __init__(self, bound_weight, xmin, xmax, ymin, ymax, zmin, zmax, nnx, nny, nnz):
        super().__init__()
        self.weight = bound_weight
        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = xmin, xmax, ymin, ymax, zmin, zmax
        x = torch.linspace(self.xmin, self.xmax, nnx)
        y = torch.linspace(self.ymin, self.ymax, nny)
        z = torch.linspace(self.zmin, self.zmax, nnz)
        X, Y, Z = torch.meshgrid(x, y, z)

        def function2solve(x, y, z):
            return torch.pow(x, 3) + torch.pow(y, 3) + torch.pow(z, 3) 
        
        domain = function2solve(X, Y, Z)
        self.domain = domain.unsqueeze(0)

    def forward(self, output, data_norm = 1.):
        batch, _, _, _, _ = output.size()
        domain = self.domain.repeat(batch, 1, 1, 1, 1)
        output /= data_norm
        bnd_loss = F.mse_loss(output[:, 0, -1, :, :], domain[:, 0, -1, :, :])
        bnd_loss += F.mse_loss(output[:, 0, :, 0, :], domain[:, 0, :, 0, :])
        bnd_loss += F.mse_loss(output[:, 0, :, -1, :], domain[:, 0, :, -1, :])
        bnd_loss += F.mse_loss(output[:, 0, 0, :, :], domain[:, 0, 0, :, :])
        bnd_loss += F.mse_loss(output[:, 0, :, :, 0], domain[:, 0, :, :, 0])
        bnd_loss += F.mse_loss(output[:, 0, :, :, -1], domain[:, 0, :, :, -1])
        return (bnd_loss * self.weight)
    

class InsideLoss(nn.Module):
    def __init__(self, cfg, inside_weight):
        super(InsideLoss, self).__init__()  # Call the parent class constructor
        self.nnx, self.nny, self.nnz = cfg['globals']['nnx'], cfg['globals']['nny'], cfg['globals']['nnz']
        self.weight = inside_weight

    def forward(self, output, target):
        return F.mse_loss(output[:, 0, 1:-1, 1:-1, 1:-1], target[:, 0, 1:-1, 1:-1, 1:-1]) * self.weight


def lapl(field, dx, dy, dz):

    # Create laplacian tensor with shape (batch_size, 1, d, h, w)
    laplacian = torch.zeros_like(field).type(field.type())

    # Check sizes
    assert field.dim() == 5 and laplacian.dim() == 5, 'Dimension mismatch'
    assert field.is_contiguous() and laplacian.is_contiguous(), 'Input is not contiguous'

    laplacian[:, 0, 1:-1, 1:-1, 1:-1] = \
            (field[:, 0, 2:, 1:-1, 1:-1] + field[:, 0, :-2, 1:-1, 1:-1] - 2 * field[:, 0, 1:-1, 1:-1, 1:-1]) / dz**2 + \
            (field[:, 0, 1:-1, 2:, 1:-1] + field[:, 0, 1:-1, :-2, 1:-1] - 2 * field[:, 0, 1:-1, 1:-1, 1:-1]) / dy**2 + \
            (field[:, 0, 1:-1, 1:-1, 2:] + field[:, 0, 1:-1, 1:-1, :-2] - 2 * field[:, 0, 1:-1, 1:-1, 1:-1]) / dx**2
    
    return laplacian


def ratio_potrhs(alpha, Lx, Ly, Lz):
    return alpha / (np.pi**2 / 8)**2 / (1 / Lx**2 + 1 / Ly**2 + 1 / Lz**2)
