import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LaplacianLoss(nn.Module):
    def __init__(self, cfg, lapl_weight):
        super().__init__()
        self.weight = lapl_weight
        xmin, xmax, ymin, ymax, zmax, zmin, nnx, nny, nnz = cfg['globals']['xmin'], cfg['globals']['xmax'],\
            cfg['globals']['ymin'], cfg['globals']['ymax'], cfg['globals']['zmin'], cfg['globals']['zmax'],\
            cfg['globals']['nnx'], cfg['globals']['nny'], cfg['globals']['nnz'] 
        self.Lx = xmax-xmin
        self.Ly = ymax-ymin
        self.Lz = zmax-zmin
        self.dx = self.Lx/nnx
        self.dy = self.Ly/nny
        self.dz = self.Lz/nnz

    def forward(self, output, data=None, data_norm=1.):
        laplacian = lapl(output / data_norm, self.dx, self.dy, self.dz)
        return self.Lx**2 * self.Ly**2 * self.Lz**2 * F.mse_loss(laplacian[:, 0, 1:-1, 1:-1, 1:-1], - data[:, 0, 1:-1, 1:-1, 1:-1]) * self.weight
    

    
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
    def __init__(self, bound_weight, boundary, e_in, e_out, dx, dy, dz):
        super().__init__()
        self.weight = bound_weight
        self.boundary = boundary
        self.e_in = e_in
        self.e_out = e_out
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def compute_gradients(self, output):
        # Calcular gradientes en la dirección x
        grad_x = (output[:, 0, 1:, :, :] - output[:, 0, :-1, :, :]) / self.dx
        # Calcular gradientes en la dirección y
        grad_y = (output[:, 0, :, 1:, :] - output[:, 0, :, :-1, :]) / self.dy
        # Calcular gradientes en la dirección z
        grad_z = (output[:, 0, :, :, 1:] - output[:, 0, :, :, :-1]) / self.dz
        return grad_x, grad_y, grad_z


    def forward(self, subdomain1, subdomain2, constant_value = 1.0):
        loss = F.mse_loss(subdomain1[:, 0, self.boudnary], subdomain2[:, 0, self.boundary])
        grad_x_sub1, grad_y_sub1, grad_z_sub1 = self.compute_gradients(subdomain1)
        grad_x_sub2, grad_y_sub2, grad_z_sub2 = self.compute_gradients(subdomain2)
        grad_x_sub1_interface, grad_y_sub1_interface, grad_z_sub1_interface = grad_x_sub1[self.boundary], grad_y_sub1[self.boundary], grad_z_sub1[self.boundary]
        grad_x_sub2_interface, grad_y_sub2_interface, grad_z_sub2_interface = grad_x_sub2[self.boundary], grad_y_sub2[self.boundary], grad_z_sub2[self.boundary]
        loss += torch.mean((self.e_in * grad_x_sub1_interface - constant_value) ** 2)
        loss += torch.mean((self.e_in * grad_y_sub1_interface - constant_value) ** 2)
        loss += torch.mean((self.e_in * grad_z_sub1_interface - constant_value) ** 2)
        loss += torch.mean((self.e_in * grad_x_sub2_interface - constant_value) ** 2)
        loss += torch.mean((self.e_in * grad_y_sub2_interface - constant_value) ** 2)
        loss += torch.mean((self.e_in * grad_z_sub2_interface - constant_value) ** 2)
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
    

    # # Left
    # laplacian[:, 0, 0, 1:-1, 1:-1] = \
    #     (2 * field[:, 0, 0, 1:-1, 1:-1] - 5 * field[:, 0, 1, 1:-1, 1:-1] + 4 * field[:, 0, 2, 1:-1, 1:-1] - field[:, 0, 3, 1:-1, 1:-1]) / dx**2 + \
    #     (field[:, 0, 0, 2:, 1:-1] - 2 * field[:, 0, 0, 1:-1, 1:-1] + field[:, 0, 0, :-2, 1:-1]) / dy**2 + \
    #     (field[:, 0, 0, 1:-1, 2:] - 2 * field[:, 0, 0, 1:-1, 1:-1] + field[:, 0, 0, 1:-1, :-2]) / dz**2

    # # Right
    # laplacian[:, 0, -1, 1:-1, 1:-1] = \
    #     (2 * field[:, 0, -1, 1:-1, 1:-1] - 5 * field[:, 0, -2, 1:-1, 1:-1] + 4 * field[:, 0, -3, 1:-1, 1:-1] - field[:, 0, -4, 1:-1, 1:-1]) / dx**2 + \
    #     (field[:, 0, -1, 2:, 1:-1] - 2 * field[:, 0, -1, 1:-1, 1:-1] + field[:, 0, -1, :-2, 1:-1]) / dy**2 + \
    #     (field[:, 0, -1, 1:-1, 2:] - 2 * field[:, 0, -1, 1:-1, 1:-1] + field[:, 0, -1, 1:-1, :-2]) / dz**2

    # # Bottom  
    # laplacian[:, 0, 1:-1, 0, 1:-1] = \
    #     (field[:, 0, 2:, 0, 1:-1] - 2 * field[:, 0, 1:-1, 0, 1:-1] + field[:, 0, :-2, 0 , 1:-1]) / dx**2 + \
    #     (2 * field[:, 0, 1:-1, 0, 1:-1] - 5 * field[:, 0, 1:-1, 1, 1:-1] + 4 * field[:, 0, 1:-1, 2, 1:-1] - field[:, 0, 1:-1, 3, 1:-1]) / dy**2 + \
    #     (field[:, 0, 1:-1, 0, 2:] - 2 * field[:, 0, 1:-1, 0, 1:-1] + field[:, 0, 1:-1, 0, :-2]) / dx**2
    
    # # Top
    # laplacian[:, 0, 1:-1, -1, 1:-1] = \
    #     (field[:, 0, 2:, -1, 1:-1] - 2 * field[:, 0, 1:-1, -1, 1:-1] + field[:, 0, :-2, -1, 1:-1]) / dx**2 + \
    #     (2 * field[:, 0, 1:-1, -1, 1:-1] - 5 * field[:, 0, 1:-1, -2, 1:-1] + 4 * field[:, 0, 1:-1, -3, 1:-1] - field[:, 0, 1:-1, -4, 1:-1]) / dy**2 + \
    #     (field[:, 0, 1:-1, -1, 2:] - 2 * field[:, 0, 1:-1, -1, 1:-1] + field[:, 0, 1:-1, -1, :-2])
    
    # # Back
    # laplacian[:, 0, 1:-1, 1:-1, 0] = \
    #     (field[:, 0, 2:, 1:-1, 0] - 2 * field[:, 0, 1:-1, 1:-1, 0] + field[:, 0, :-2, 1:-1, 0]) / dx**2 + \
    #     (field[:, 0, 1:-1, 2:, 0] - 2 * field[:, 0, 1:-1, 1:-1, 0] + field[:, 0, 1:-1, :-2, 0]) / dy**2 + \
    #     (2 * field[:, 0, 1:-1, 1:-1, 0] - 5 * field[:, 0, 1:-1, 1:-1, 1] + 4 * field[:, 0, 1:-1, 1:-1, 2] - field[:, 0, 1:-1, 1:-1, 3]) / dz**2
    
    # # Front
    # laplacian[:, 0, 1:-1, 1:-1, -1] = \
    #     (field[:, 0, 2:, 1:-1, -1] - 2 * field[:, 0, 1:-1, 1:-1, -1] + field[:, 0, :-2, 1:-1, -1]) / dx**2 + \
    #     (field[:, 0, 1:-1, 2:, -1] - 2 * field[:, 0, 1:-1, 1:-1, -1] + field[:, 0, 1:-1, :-2, -1]) / dy**2 + \
    #     (2 * field[:, 0, 1:-1, 1:-1, -1] - 5 * field[:, 0, 1:-1, 1:-1, -2] - 4 * field[:, 0, 1:-1, 1:-1, -3] - field[:, 0, 1:-1, 1:-1, -4])
    


    # # Corners
    # laplacian[:, 0, 0, 0, 0] = \
    #     (2 * field[:, 0, 0, 0, 0] - 5 * field[:, 0, 1, 0, 0] + 4 * field[:, 0, 2, 0, 0] - field[:, 0, 3, 0, 0]) / dx**2 + \
    #     (2 * field[:, 0, 0, 0, 0] - 5 * field[:, 0, 0, 1, 0] + 4 * field[:, 0, 0, 2, 0] - field[:, 0, 0, 3, 0]) / dy**2 + \
    #     (2 * field[:, 0, 0, 0, 0] - 5 * field[:, 0, 0, 0, 1] + 4 * field[:, 0, 0, 0, 2] - field[:, 0, 0, 0, 3]) / dz**2
    
    # laplacian[:, 0, -1, 0, 0] = \
    #     (2 * field[:, 0, -1, 0, 0] - 5 * field[:, 0, -2, 0, 0] + 4 * field[:, 0, -3, 0, 0] - field[:, 0, -4, 0, 0]) / dx**2 + \
    #     (2 * field[:, 0, -1, 0, 0] - 5 * field[:, 0, -1, 1, 0] + 4 * field[:, 0, -1, 2, 0] - field[:, 0, -1, 3, 0]) / dy**2 + \
    #     (2 * field[:, 0, -1, 0, 0] - 5 * field[:, 0, -1, 0, 1] + 4 * field[:, 0, -1, 0 ,2] - field[:, 0, -1, 0, 3]) / dz**2
    
    # laplacian[:, 0, 0, -1, 0] = \
    #     (2 * field[:, 0, 0, -1, 0] - 5 * field[:, 0, 1, -1, 0] + 4 * field[:, 0, 2, -1, 0] - field[:, 0, 3, -1, 0]) / dx**2 + \
    #     (2 * field[:, 0, 0, -1, 0] - 5 * field[:, 0, 0, -2, 0] + 4 * field[:, 0, 0, -3, 0] - field[:, 0, 0, -4, 0]) / dy**2 + \
    #     (2 * field[:, 0, 0, -1, 0] - 5 * field[:, 0, 0, -1, 1] + 4 * field[:, 0, 0, -1, 2] - field[:, 0, 0, -1, 3]) / dz**2
    
    # laplacian[:, 0, 0, 0, -1] = \
    #     (2 * field[:, 0, 0, 0, -1] - 5 * field[:, 0, 1, 0, -1] + 4 * field[:, 0, 2, 0, -1] - field[:, 0, 3, 0, -1]) / dx**2 + \
    #     (2 * field[:, 0, 0, 0, -1] - 5 * field[:, 0, 0, 1, -1] + 4 * field[:, 0, 0, 2, -1] - field[:, 0, 0, 3, -1]) / dy**2 + \
    #     (2 * field[:, 0, 0, 0, -1] - 5 * field[:, 0, 0, 0, -2] + 4 * field[:, 0, 0, 0, -3] - field[:, 0, 0, 0, -4]) / dz**2
    
    # laplacian[:, 0, -1, -1, 0] = \
    #     (2 * field[:, 0, -1, -1, 0] - 5 * field[:, 0, -2, -1, 0] + 4 * field[:, 0, -3, -1, 0] - field[:, 0, -4, -1, 0]) / dx**2 + \
    #     (2 * field[:, 0, -1, -1, 0] - 5 * field[:, 0, -1, -2, 0] + 4 * field[:, 0, -1, -3, 0] - field[:, 0, -1, -4, 0]) / dy**2 + \
    #     (2 * field[:, 0, -1, -1, 0] - 5 * field[:, 0, -1, -1, 1] + 4 * field[:, 0, -1, -1, 2] - field[:, 0, -1, -1, 3]) / dz**2 

    # laplacian[:, 0, -1, 0, -1] = \
    #     (2 * field[:, 0, -1, 0, -1] - 5 * field[:, 0, -2, 0, -1] + 4 * field[:, 0, -3, 0, -1] - field[:, 0, -4, 0, -1]) / dx**2 + \
    #     (2 * field[:, 0, -1, 0, -1] - 5 * field[:, 0, -1, 1, -1] + 4 * field[:, 0, -1, 2, -1] - field[:, 0, -1, 3, -1]) / dy**2 + \
    #     (2 * field[:, 0, -1, 0, -1] - 5 * field[:, 0, -1, 0, -2] + 4 * field[:, 0, -1, 0, -3] - field[:, 0, -1, 0, -4]) / dz**2
    
    # laplacian[:, 0, 0, -1, -1] = \
    #     (2 * field[:, 0, 0, -1, -1] - 5 * field[:, 0, 1, -1, -1] + 4 * field[:, 0, 2, -1, -1] - field[:, 0, 3, -1, -1]) / dx**2 + \
    #     (2 * field[:, 0, 0, -1, -1] - 5 * field[:, 0, 0, -2, -1] + 4 * field[:, 0, 0, -3, -1] - field[:, 0, 0, -4, -1]) / dy**2 + \
    #     (2 * field[:, 0, 0, -1, -1] - 5 * field[:, 0, 0, -1, -2] + 4 * field[:, 0, 0, -1, -3] - field[:, 0, 0, -1, -4]) / dz**2
    
    # laplacian[:, 0, -1, -1, -1] = \
    #     (2 * field[:, 0, -1, -1, -1] - 5 * field[:, 0, -2, -1, -1] + 4 * field[:, 0, -3, -1, -1] - field[:, 0, -4, -1, -1]) / dx**2 + \
    #     (2 * field[:, 0, -1, -1, -1] - 5 * field[:, 0, -1, -2, -1] + 4 * field[:, 0, -1, -3, -1] - field[:, 0, -1, -4, -1]) / dy**2 + \
    #     (2 * field[:, 0, -1, -1, -1] - 5 * field[:, 0, -1, -1, -2] + 4 * field[:, 0, -1, -1, -3] - field[:, 0, -1, -1, -4]) / dz**2

    return laplacian


def ratio_potrhs(alpha, Lx, Ly, Lz):
    return alpha / (np.pi**2 / 8)**2 / (1 / Lx**2 + 1 / Ly**2 + 1 / Lz**2)
