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
        self.base_weight = self.weight

    def forward(self, output):
        bnd_loss = F.mse_loss(output[:, 0, -1, :, :], torch.zeros_like(output[:, 0, -1, :, :]))
        bnd_loss += F.mse_loss(output[:, 0, :, 0, :], torch.zeros_like(output[:, 0, :, 0, :]))
        bnd_loss += F.mse_loss(output[:, 0, :, -1, :], torch.zeros_like(output[:, 0, :, -1, :]))
        bnd_loss += F.mse_loss(output[:, 0, 0, :, :], torch.zeros_like(output[:, 0, 0, :, :]))
        bnd_loss += F.mse_loss(output[:, 0, :, :, 0], torch.zeros_like(output[:, 0, :, :, 0]))
        bnd_loss += F.mse_loss(output[:, 0, :, :, -1], torch.zeros_like(output[:, 0, :, :, -1]))
        return bnd_loss * self.weight




def lapl(field, dx, dy, dz, b=0):

    # Create laplacian tensor with shape (batch_size, 1, d, h, w)
    laplacian = torch.zeros_like(field).type(field.type())

    # Check sizes
    assert field.dim() == 5 and laplacian.dim() == 5, 'Dimension mismatch'
    assert field.is_contiguous() and laplacian.is_contiguous(), 'Input is not contiguous'

    laplacian[:, 0, 1:-1, 1:-1, 1:-1] = \
        (1 - b) * (
            (field[:, 0, 2:, 1:-1, 1:-1] + field[:, 0, :-2, 1:-1, 1:-1] - 2 * field[:, 0, 1:-1, 1:-1, 1:-1]) / dz**2 +
            (field[:, 0, 1:-1, 2:, 1:-1] + field[:, 0, 1:-1, :-2, 1:-1] - 2 * field[:, 0, 1:-1, 1:-1, 1:-1]) / dy**2 +
            (field[:, 0, 1:-1, 1:-1, 2:] + field[:, 0, 1:-1, 1:-1, :-2] - 2 * field[:, 0, 1:-1, 1:-1, 1:-1]) / dx**2
        ) + \
        b * (
            (field[:, 0, 2:, 2:, 1:-1] + field[:, 0, 2:, :-2, 1:-1] + field[:, 0, :-2, :-2, 1:-1] + field[:, 0, :-2, 2:, 1:-1] - 4 * field[:, 0, 1:-1, 1:-1, 1:-1]) / (2 * dx**2) +
            (field[:, 0, 2:, 1:-1, 2:] + field[:, 0, 2:, 1:-1, :-2] + field[:, 0, :-2, 1:-1, :-2] + field[:, 0, :-2, 1:-1, 2:] - 4 * field[:, 0, 1:-1, 1:-1, 1:-1]) / (2 * dy**2) +
            (field[:, 0, 2:, 1:-1, 1:-1] + field[:, 0, 2:, 1:-1, 1:-1] + field[:, 0, :-2, 1:-1, 1:-1] + field[:, 0, :-2, 1:-1, 1:-1] - 4 * field[:, 0, 1:-1, 1:-1, 1:-1]) / (2 * dz**2)
        )
    
    #Borde
    laplacian[:, 0, 1:-1, 1:-1, 0] = \
        (field[:, 0, 1:-1, 2: , 0] - 2 * field[:, 0, 1:-1, 1:-1, 0] + field[:, 0, 1:-1, :-2, 0]) / dy**2 \
        + (2 * field[:, 0, 1:-1, 1:-1, 0] - 5 * field[:, 0, 1:-1, 1:-1, 1] + 4 * field[:, 0, 1:-1, 1:-1, 2] - field[:, 0, 1:-1, 1:-1, 3]) / dx**2 \
        + (field[:, 0, 2:, 1:-1, 0] - 2 * field[:, 0, 1:-1, 1:-1, 0] + field[:, 0, :-2, 1:-1, 0]) / dz**2

    laplacian[:, 0, 1:-1, 1:-1, -1] = \
        (field[:, 0, 1:-1, 2:, -1] - 2 * field[:, 0, 1:-1, 1:-1, -1] + field[:, 0, 1:-1, :-2, -1]) / dy**2 \
        + (2 * field[:, 0, 1:-1, 1:-1, -1] - 5 * field[:, 0, 1:-1, 1:-1, -2] + 4 * field[:, 0, 1:-1, 1:-1, -3] - field[:, 0, 1:-1, 1:-1, -4]) / dx**2 \
        + (field[:, 0, 2:, 1:-1, -1] - 2 * field[:, 0, 1:-1, 1:-1, -1] + field[:, 0, :-2, 1:-1, -1]) / dz**2
    
    laplacian[:, 0, 1:-1, 0, 1:-1] = \
        (field[:, 0, 1:-1, 0, 2:] - 2 * field[:, 0, 1:-1, 0, 1:-1] + field[:, 0, 1:-1, 0, :-2]) / dx**2 \
        + (field[:, 0, 2:, 0, 1:-1] - 2 * field[:, 0, 1:-1, 0, 1:-1] + field[:, 0, :-2, 0, 1:-1]) / dz**2 \
        + (2 * field[:, 0, 1:-1, 0, 1:-1] - 5 * field[:, 0, 1:-1, 1, 1:-1] + 4 * field[:, 0, 1:-1, 2, 1:-1] - field[:, 0, 1:-1, 3, 1:-1]) / dy**2
    
    laplacian[:, 0, 1:-1, -1, 1:-1] = \
        (field[:, 0, 1:-1, -1, 2:] - 2 * field[:, 0, 1:-1, -1, 1:-1] + field[:, 0, 1:-1, -1, :-2]) / dx**2 \
        + (field[:, 0, 2:, -1, 1:-1] - 2 * field[:, 0, 1:-1, -1, 1:-1] + field[:, 0, :-2, -1, 1:-1]) / dz**2 \
        + (2 * field[:, 0, 1:-1, -1, 1:-1] - 5 * field[:, 0, 1:-1, -2, 1:-1] + 4 * field[:, 0, 1:-1, -3, 1:-1] - field[:, 0, 1:-1, -4, 1:-1]) / dy**2
    
    laplacian[:, 0, 0, 1:-1, 1:-1] = \
        (2 * field[:, 0, 0, 1:-1, 1:-1] - 5 * field[:, 0, 1, 1:-1, 1:-1] + 4 * field[:, 0, 2, 1:-1, 1:-1] - field[:, 0, 3, 1:-1, 1:-1]) / dz**2 \
        + (field[:, 0, 0, 2:, 1:-1] - 2 * field[:, 0, 0, 1:-1, 1:-1] + field[:, 0, 0, :-2, 1:-1]) / dy**2 \
        + (field[:, 0, 0, 1:-1, 2:] - 2 * field[:, 0, 0, 1:-1, 1:-1] + field[:, 0, 0, 1:-1, :-2]) / dx**2
    
    laplacian[:, 0, -1, 1:-1, 1:-1] = \
        (2 * field[:, 0, -1, 1:-1, 1:-1] - 5 * field[:, 0, -2, 1:-1, 1:-1] + 4 * field[:, 0, -3, 1:-1, 1:-1] - field[:, 0, -4, 1:-1, 1:-1]) / dz**2 \
        + (field[:, 0, -1, 2:, 1:-1] - 2 * field[:, 0, -1, 1:-1, 1:-1] + field[:, 0, -1, :-2, 1:-1]) / dy**2 \
        + (field[:, 0, -1, 1:-1, 2:] - 2 * field[:, 0, -1, 1:-1, 1:-1] + field[:, 0, -1, 1:-1, :-2]) / dx**2
    



    laplacian[:, 0, 0, 0, 0] = \
        (2 * field[:, 0, 0, 0, 0] - 5 * field[:, 0, 1, 0, 0] + 4 * field[:, 0, 2, 0, 0] - field[:, 0, 3, 0, 0]) / dz**2 \
        + (2 * field[:, 0, 0, 0, 0] - 5 * field[:, 0, 0, 1, 0] + 4 * field[:, 0, 0, 2, 0] - field[:, 0, 0, 3, 0]) / dy**2 \
        + (2 * field[:, 0, 0, 0, 0] - 5 * field[:, 0, 0, 0, 1] + 4 * field[:, 0, 0, 0, 2] - field[:, 0, 0, 0, 3]) / dx**2
    
    laplacian[:, 0, -1, 0, 0] = \
        (2 * field[:, 0, -1, 0, 0] - 5 * field[:, 0, -2, 0, 0] + 4 * field[:, 0, -3, 0, 0] - field[:, 0, -4, 0, 0]) / dz**2 \
        + (2 * field[:, 0, -1, 0, 0] - 5 * field[:, 0, -1, 1, 0] + 4 * field[:, 0, -1, 2, 0] - field[:, 0, -1, 3, 0]) / dy**2 \
        + (2 * field[:, 0, -1, 0, 0] - 5 * field[:, 0, -1, 0, 1] + 4 * field[:, 0, -1, 0 ,2] - field[:, 0, -1, 0, 3]) / dx**2
    
    laplacian[:, 0, 0, -1, 0] = \
        (2 * field[:, 0, 0, -1, 0] - 5 * field[:, 0, 1, -1, 0] + 4 * field[:, 0, 2, -1, 0] - field[:, 0, 3, -1, 0]) / dz**2 \
        + (2 * field[:, 0, 0, -1, 0] - 5 * field[:, 0, 0, -2, 0] + 4 * field[:, 0, 0, -3, 0] - field[:, 0, 0, -4, 0]) / dy**2 \
        + (2 * field[:, 0, 0, -1, 0] - 5 * field[:, 0, 0, -1, 1] + 4 * field[:, 0, 0, -1, 2] - field[:, 0, 0, -1, 3]) / dx**2
    
    laplacian[:, 0, 0, 0, -1] = \
        (2 * field[:, 0, 0, 0, -1] - 5 * field[:, 0, 1, 0, -1] + 4 * field[:, 0, 2, 0, -1] - field[:, 0, 3, 0, -1]) / dz**2 \
        + (2 * field[:, 0, 0, 0, -1] - 5 * field[:, 0, 0, 1, -1] + 4 * field[:, 0, 0, 2, -1] - field[:, 0, 0, 3, -1]) / dy**2 \
        + (2 * field[:, 0, 0, 0, -1] - 5 * field[:, 0, 0, 0, -2] + 4 * field[:, 0, 0, 0, -3] - field[:, 0, 0, 0, -4]) / dx**2
    
    laplacian[:, 0, -1, -1, 0] = \
        (2 * field[:, 0, -1, -1, 0] - 5 * field[:, 0, -2, -1, 0] + 4 * field[:, 0, -3, -1, 0] - field[:, 0, -4, -1, 0]) / dz**2 \
        + (2 * field[:, 0, -1, -1, 0] - 5 * field[:, 0, -1, -2, 0] + 4 * field[:, 0, -1, -3, 0] - field[:, 0, -1, -4, 0]) / dy**2 \
        + (2 * field[:, 0, -1, -1, 0] - 5 * field[:, 0, -1, -1, 1] + 4 * field[:, 0, -1, -1, 2] - field[:, 0, -1, -1, 3]) / dx**2 

    laplacian[:, 0, -1, 0, -1] = \
        (2 * field[:, 0, -1, 0, -1] - 5 * field[:, 0, -2, 0, -1] + 4 * field[:, 0, -3, 0, -1] - field[:, 0, -4, 0, -1]) / dz**2 \
        + (2 * field[:, 0, -1, 0, -1] - 5 * field[:, 0, -1, 1, -1] + 4 * field[:, 0, -1, 2, -1] - field[:, 0, -1, 3, -1]) / dy**2 \
        + (2 * field[:, 0, -1, 0, -1] - 5 * field[:, 0, -1, 0, -2] + 4 * field[:, 0, -1, 0, -3] - field[:, 0, -1, 0, -4]) / dx**2
    
    laplacian[:, 0, 0, -1, -1] = \
        (2 * field[:, 0, 0, -1, -1] - 5 * field[:, 0, 1, -1, -1] + 4 * field[:, 0, 2, -1, -1] - field[:, 0, 3, -1, -1]) / dz**2 \
        + (2 * field[:, 0, 0, -1, -1] - 5 * field[:, 0, 0, -2, -1] + 4 * field[:, 0, 0, -3, -1] - field[:, 0, 0, -4, -1]) / dy**2 \
        + (2 * field[:, 0, 0, -1, -1] - 5 * field[:, 0, 0, -1, -2] + 4 * field[:, 0, 0, -1, -3] - field[:, 0, 0, -1, -4]) / dx**2
    
    laplacian[:, 0, -1, -1, -1] = \
        (2 * field[:, 0, -1, -1, -1] - 5 * field[:, 0, -2, -1, -1] + 4 * field[:, 0, -3, -1, -1] - field[:, 0, -4, -1, -1]) / dz**2 \
        + (2 * field[:, 0, -1, -1, -1] - 5 * field[:, 0, -1, -2, -1] + 4 * field[:, 0, -1, -3, -1] - field[:, 0, -1, -4, -1]) / dy**2 \
        + (2 * field[:, 0, -1, -1, -1] - 5 * field[:, 0, -1, -1, -2] + 4 * field[:, 0, -1, -1, -3] - field[:, 0, -1, -1, -4]) / dz**2



    return laplacian




def ratio_potrhs(alpha, Lx, Ly, Lz):
    return alpha / (np.pi**2 / 4)**2 / (1 / Lx**2 + 1 / Ly**2 + 1 / Lz**2)
