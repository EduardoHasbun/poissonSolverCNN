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
    def __init__(self, bound_weight, boundary, interface, e_in, e_out, dx, dy):
        super().__init__()
        self.weight = bound_weight
        self.boundary = boundary
        self.interface = interface
        self.e_in = e_in
        self.e_out = e_out
        self.dx = dx
        self.dy = dy

    def compute_gradients(self, output, interface_mask):
        grad_x = torch.zeros_like(output)
        grad_y = torch.zeros_like(output)
        interface_mask[self.boundary] == True
        for i in range(1, output.shape[2]):
            for j in range(1, output.shape[3]):
                if interface_mask[i, j] == interface_mask[i - 1, j]:
                    grad_x[:, 0, i, j] = (output[:, 0, i, j] - output[:, 0, i - 1, j]) / self.dx
                elif interface_mask[i, j] == interface_mask[i + 1, j]:
                    grad_x[:, 0, i, j] = (output[:, 0, i, j] - output[:, 0, i + 1, j]) / self.dx

                if interface_mask[i, j] == interface_mask[i, j - 1]:
                    grad_y[:, 0, i, j] = (output[:, 0, i, j] - output[:, 0, i, j - 1]) / self.dy
                elif interface_mask[i, j] == interface_mask[i, j + 1]:
                    grad_y[:, 0, i, j] = (output[:, 0, i, j] - output[:, 0, i, j + 1]) / self.dy
                
        return grad_x, grad_y


    def forward(self, subdomain1, subdomain2, constant_value = 1.0):
        loss = F.mse_loss(subdomain1[:, 0, self.boundary], subdomain2[:, 0, self.boundary])
        grad_x_sub1, grad_y_sub1 = self.compute_gradients(subdomain1, self.interface)
        grad_x_sub2, grad_y_sub2 = self.compute_gradients(subdomain2, ~self.interface)
        grad_x_sub1_interface, grad_y_sub1_interface = grad_x_sub1[:, self.boundary], grad_y_sub1[:, self.boundary]
        grad_x_sub2_interface, grad_y_sub2_interface = grad_x_sub2[:, self.boundary], grad_y_sub2[:, self.boundary]
        loss += torch.mean((self.e_in * grad_x_sub1_interface - constant_value) ** 2)
        loss += torch.mean((self.e_in * grad_y_sub1_interface - constant_value) ** 2)
        loss += torch.mean((self.e_in * grad_x_sub2_interface - constant_value) ** 2)
        loss += torch.mean((self.e_in * grad_y_sub2_interface - constant_value) ** 2)
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

    laplacian[:, 0, interface] *= epsilon_in
    laplacian[:, 0, ~interface] *= epsilon_out

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
