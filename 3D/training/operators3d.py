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
    

class NewDirichletBoundaryLoss(nn.Module):
    def __init__(self, bound_weight, xmin, xmax, ymin, ymax, zmin, zmax, nnx, nny, nnz, batch):
        super().__init__()
        self.weight = bound_weight
        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = xmin, xmax, ymin, ymax, zmin, zmax
        x = torch.linspace(self.xmin, self.xmax, nnx)
        y = torch.linspace(self.ymin, self.ymax, nny)
        z = torch.linspace(self.zmin, self.zmax, nnz)
        def function2solve(x, y, z):
            return torch.pow(x, 3) + torch.pow(y, 3) + torch.pow(z, 3)   
        domain = torch.zeros(nnx, nny, nnz)
        for i in range(nnx):
            for j in range(nny):
                for k in range(nnz):
                    domain[i, j, k] = function2solve(x[i], y[j], z[k])
        self.domain = domain.unsqueeze(0)

    def forward(self, output):
        batch, _, _, _, _ = output.size()
        self.domain = self.domain.repeat(batch, 1, 1, 1, 1)
        bnd_loss = F.mse_loss(output[:, 0, -1, :, :], self.domain[:, 0, -1, :, :])
        bnd_loss += F.mse_loss(output[:, 0, :, 0, :], self.domain[:, 0, :, 0, :])
        bnd_loss += F.mse_loss(output[:, 0, :, -1, :], self.domain[:, 0, :, -1, :])
        bnd_loss += F.mse_loss(output[:, 0, 0, :, :], self.domain[:, 0, 0, :, :])
        bnd_loss += F.mse_loss(output[:, 0, :, :, 0], self.domain[:, 0, :, :, 0])
        bnd_loss += F.mse_loss(output[:, 0, :, :, -1], self.domain[:, 0, :, :, -1])
        self.domain = np.squeeze(self.domain, axis=0)
        return bnd_loss * self.weight
    

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
    

    # Left
    laplacian[:, 0, 0, 1:-1, 1:-1] = \
        (2 * field[:, 0, 0, 1:-1, 1:-1] - 5 * field[:, 0, 1, 1:-1, 1:-1] + 4 * field[:, 0, 2, 1:-1, 1:-1] - field[:, 0, 3, 1:-1, 1:-1]) / dx**2 + \
        (field[:, 0, 0, 2:, 1:-1] - 2 * field[:, 0, 0, 1:-1, 1:-1] + field[:, 0, 0, :-2, 1:-1]) / dy**2 + \
        (field[:, 0, 0, 1:-1, 2:] - 2 * field[:, 0, 0, 1:-1, 1:-1] + field[:, 0, 0, 1:-1, :-2]) / dz**2

    # Right
    laplacian[:, 0, -1, 1:-1, 1:-1] = \
        (2 * field[:, 0, -1, 1:-1, 1:-1] - 5 * field[:, 0, -2, 1:-1, 1:-1] + 4 * field[:, 0, -3, 1:-1, 1:-1] - field[:, 0, -4, 1:-1, 1:-1]) / dx**2 + \
        (field[:, 0, -1, 2:, 1:-1] - 2 * field[:, 0, -1, 1:-1, 1:-1] + field[:, 0, -1, :-2, 1:-1]) / dy**2 + \
        (field[:, 0, -1, 1:-1, 2:] - 2 * field[:, 0, -1, 1:-1, 1:-1] + field[:, 0, -1, 1:-1, :-2]) / dz**2

    # Bottom  
    laplacian[:, 0, 1:-1, 0, 1:-1] = \
        (field[:, 0, 2:, 0, 1:-1] - 2 * field[:, 0, 1:-1, 0, 1:-1] + field[:, 0, :-2, 0 , 1:-1]) / dx**2 + \
        (2 * field[:, 0, 1:-1, 0, 1:-1] - 5 * field[:, 0, 1:-1, 1, 1:-1] + 4 * field[:, 0, 1:-1, 2, 1:-1] - field[:, 0, 1:-1, 3, 1:-1]) / dy**2 + \
        (field[:, 0, 1:-1, 0, 2:] - 2 * field[:, 0, 1:-1, 0, 1:-1] + field[:, 0, 1:-1, 0, :-2]) / dx**2
    
    # Top
    laplacian[:, 0, 1:-1, -1, 1:-1] = \
        (field[:, 0, 2:, -1, 1:-1] - 2 * field[:, 0, 1:-1, -1, 1:-1] + field[:, 0, :-2, -1, 1:-1]) / dx**2 + \
        (2 * field[:, 0, 1:-1, -1, 1:-1] - 5 * field[:, 0, 1:-1, -2, 1:-1] + 4 * field[:, 0, 1:-1, -3, 1:-1] - field[:, 0, 1:-1, -4, 1:-1]) / dy**2 + \
        (field[:, 0, 1:-1, -1, 2:] - 2 * field[:, 0, 1:-1, -1, 1:-1] + field[:, 0, 1:-1, -1, :-2])
    
    # Back
    laplacian[:, 0, 1:-1, 1:-1, 0] = \
        (field[:, 0, 2:, 1:-1, 0] - 2 * field[:, 0, 1:-1, 1:-1, 0] + field[:, 0, :-2, 1:-1, 0]) / dx**2 + \
        (field[:, 0, 1:-1, 2:, 0] - 2 * field[:, 0, 1:-1, 1:-1, 0] + field[:, 0, 1:-1, :-2, 0]) / dy**2 + \
        (2 * field[:, 0, 1:-1, 1:-1, 0] - 5 * field[:, 0, 1:-1, 1:-1, 1] + 4 * field[:, 0, 1:-1, 1:-1, 2] - field[:, 0, 1:-1, 1:-1, 3]) / dz**2
    
    # Front
    laplacian[:, 0, 1:-1, 1:-1, -1] = \
        (field[:, 0, 2:, 1:-1, -1] - 2 * field[:, 0, 1:-1, 1:-1, -1] + field[:, 0, :-2, 1:-1, -1]) / dx**2 + \
        (field[:, 0, 1:-1, 2:, -1] - 2 * field[:, 0, 1:-1, 1:-1, -1] + field[:, 0, 1:-1, :-2, -1]) / dy**2 + \
        (2 * field[:, 0, 1:-1, 1:-1, -1] - 5 * field[:, 0, 1:-1, 1:-1, -2] - 4 * field[:, 0, 1:-1, 1:-1, -3] - field[:, 0, 1:-1, 1:-1, -4])
    


    # Corners
    laplacian[:, 0, 0, 0, 0] = \
        (2 * field[:, 0, 0, 0, 0] - 5 * field[:, 0, 1, 0, 0] + 4 * field[:, 0, 2, 0, 0] - field[:, 0, 3, 0, 0]) / dx**2 + \
        (2 * field[:, 0, 0, 0, 0] - 5 * field[:, 0, 0, 1, 0] + 4 * field[:, 0, 0, 2, 0] - field[:, 0, 0, 3, 0]) / dy**2 + \
        (2 * field[:, 0, 0, 0, 0] - 5 * field[:, 0, 0, 0, 1] + 4 * field[:, 0, 0, 0, 2] - field[:, 0, 0, 0, 3]) / dz**2
    
    laplacian[:, 0, -1, 0, 0] = \
        (2 * field[:, 0, -1, 0, 0] - 5 * field[:, 0, -2, 0, 0] + 4 * field[:, 0, -3, 0, 0] - field[:, 0, -4, 0, 0]) / dx**2 + \
        (2 * field[:, 0, -1, 0, 0] - 5 * field[:, 0, -1, 1, 0] + 4 * field[:, 0, -1, 2, 0] - field[:, 0, -1, 3, 0]) / dy**2 + \
        (2 * field[:, 0, -1, 0, 0] - 5 * field[:, 0, -1, 0, 1] + 4 * field[:, 0, -1, 0 ,2] - field[:, 0, -1, 0, 3]) / dz**2
    
    laplacian[:, 0, 0, -1, 0] = \
        (2 * field[:, 0, 0, -1, 0] - 5 * field[:, 0, 1, -1, 0] + 4 * field[:, 0, 2, -1, 0] - field[:, 0, 3, -1, 0]) / dx**2 + \
        (2 * field[:, 0, 0, -1, 0] - 5 * field[:, 0, 0, -2, 0] + 4 * field[:, 0, 0, -3, 0] - field[:, 0, 0, -4, 0]) / dy**2 + \
        (2 * field[:, 0, 0, -1, 0] - 5 * field[:, 0, 0, -1, 1] + 4 * field[:, 0, 0, -1, 2] - field[:, 0, 0, -1, 3]) / dz**2
    
    laplacian[:, 0, 0, 0, -1] = \
        (2 * field[:, 0, 0, 0, -1] - 5 * field[:, 0, 1, 0, -1] + 4 * field[:, 0, 2, 0, -1] - field[:, 0, 3, 0, -1]) / dx**2 + \
        (2 * field[:, 0, 0, 0, -1] - 5 * field[:, 0, 0, 1, -1] + 4 * field[:, 0, 0, 2, -1] - field[:, 0, 0, 3, -1]) / dy**2 + \
        (2 * field[:, 0, 0, 0, -1] - 5 * field[:, 0, 0, 0, -2] + 4 * field[:, 0, 0, 0, -3] - field[:, 0, 0, 0, -4]) / dz**2
    
    laplacian[:, 0, -1, -1, 0] = \
        (2 * field[:, 0, -1, -1, 0] - 5 * field[:, 0, -2, -1, 0] + 4 * field[:, 0, -3, -1, 0] - field[:, 0, -4, -1, 0]) / dx**2 + \
        (2 * field[:, 0, -1, -1, 0] - 5 * field[:, 0, -1, -2, 0] + 4 * field[:, 0, -1, -3, 0] - field[:, 0, -1, -4, 0]) / dy**2 + \
        (2 * field[:, 0, -1, -1, 0] - 5 * field[:, 0, -1, -1, 1] + 4 * field[:, 0, -1, -1, 2] - field[:, 0, -1, -1, 3]) / dz**2 

    laplacian[:, 0, -1, 0, -1] = \
        (2 * field[:, 0, -1, 0, -1] - 5 * field[:, 0, -2, 0, -1] + 4 * field[:, 0, -3, 0, -1] - field[:, 0, -4, 0, -1]) / dx**2 + \
        (2 * field[:, 0, -1, 0, -1] - 5 * field[:, 0, -1, 1, -1] + 4 * field[:, 0, -1, 2, -1] - field[:, 0, -1, 3, -1]) / dy**2 + \
        (2 * field[:, 0, -1, 0, -1] - 5 * field[:, 0, -1, 0, -2] + 4 * field[:, 0, -1, 0, -3] - field[:, 0, -1, 0, -4]) / dz**2
    
    laplacian[:, 0, 0, -1, -1] = \
        (2 * field[:, 0, 0, -1, -1] - 5 * field[:, 0, 1, -1, -1] + 4 * field[:, 0, 2, -1, -1] - field[:, 0, 3, -1, -1]) / dx**2 + \
        (2 * field[:, 0, 0, -1, -1] - 5 * field[:, 0, 0, -2, -1] + 4 * field[:, 0, 0, -3, -1] - field[:, 0, 0, -4, -1]) / dy**2 + \
        (2 * field[:, 0, 0, -1, -1] - 5 * field[:, 0, 0, -1, -2] + 4 * field[:, 0, 0, -1, -3] - field[:, 0, 0, -1, -4]) / dz**2
    
    laplacian[:, 0, -1, -1, -1] = \
        (2 * field[:, 0, -1, -1, -1] - 5 * field[:, 0, -2, -1, -1] + 4 * field[:, 0, -3, -1, -1] - field[:, 0, -4, -1, -1]) / dx**2 + \
        (2 * field[:, 0, -1, -1, -1] - 5 * field[:, 0, -1, -2, -1] + 4 * field[:, 0, -1, -3, -1] - field[:, 0, -1, -4, -1]) / dy**2 + \
        (2 * field[:, 0, -1, -1, -1] - 5 * field[:, 0, -1, -1, -2] + 4 * field[:, 0, -1, -1, -3] - field[:, 0, -1, -1, -4]) / dz**2



    return laplacian




def ratio_potrhs(alpha, Lx, Ly, Lz):
    return alpha / (np.pi**2 / 8)**2 / (1 / Lx**2 + 1 / Ly**2 + 1 / Lz**2)
