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
    def __init__(self, bound_weight, boundary, interface, center, radius, e_in, e_out, dx, dy):
        super().__init__()
        self.weight = bound_weight
        self.boundary = boundary
        self.interface = interface
        self.e_in = e_in
        self.e_out = e_out
        self.dx = dx
        self.dy = dy
        self.center = center
        self.radius = radius

    def is_inside(self, x_idx, y_idx):
    # Calculate the real position of the node in physical space
        x_node = x_idx * self.dx
        y_node = y_idx * self.dy
        
        # Compute the distance of the node from the center of the circle
        distance_to_center = torch.sqrt((x_node - self.center[0]) ** 2 + (y_node - self.center[1]) ** 2)
        
        # Check if the node is inside the circle (i.e., distance is less than the radius)
        return distance_to_center < self.radius


    def compute_gradients(self, output, interface_mask, inside = True):
        # Prepare gradient tensors
        grad_x = torch.zeros_like(output)
        grad_y = torch.zeros_like(output)

        # Compute mask for inner nodes
        mask_x = (interface_mask[1:, :] == interface_mask[:-1, :])
        mask_y = (interface_mask[:, 1:] == interface_mask[:, :-1])

        grad_x[:, :, 1:, :] = ((output[:, :, 1:, :] - output[:, :, :-1, :]) / self.dx) * mask_x
        grad_y[:, :, :, 1:] = ((output[:, :, :, 1:] - output[:, :, :, :-1]) / self.dy) * mask_y

        # Handle the boundary nodes
        boundary_indices = torch.nonzero(self.boundary, as_tuple=True)
        for idx in zip(*boundary_indices):
            x_idx, y_idx = idx[0], idx[1]
            x_node, y_node = x_idx * self.dx, y_idx * self.dy

            # Compute normal vector
            normal_x = (x_node - self.center[0])
            normal_y = (y_node - self.center[1])
            norm = torch.sqrt(normal_x**2 + normal_y**2)
            normal_x /= norm
            normal_y /= norm

            # Determine which neighbor to use for gradient
            if normal_x > 0:  # Use node to the right
                grad_x[:, 0, x_idx, y_idx] = (output[:, 0, x_idx, y_idx] - output[:, 0, x_idx + 1, y_idx]) / self.dx
            else:  # Use node to the left
                grad_x[:, 0, x_idx, y_idx] = (output[:, 0, x_idx, y_idx] - output[:, 0, x_idx - 1, y_idx]) / self.dx

            if normal_y > 0:  # Use node above
                grad_y[:, 0, x_idx, y_idx] = (output[:, 0, x_idx, y_idx] - output[:, 0, x_idx, y_idx + 1]) / self.dy
            else:  # Use node below
                grad_y[:, 0, x_idx, y_idx] = (output[:, 0, x_idx, y_idx] - output[:, 0, x_idx, y_idx - 1]) / self.dy


    def forward(self, subdomain1, subdomain2, constant_value = 1.0):
        loss = F.mse_loss(subdomain1[:, 0, self.boundary], subdomain2[:, 0, self.boundary])
        grad_x_sub1, grad_y_sub1 = self.compute_gradients(subdomain1, self.interface)
        grad_x_sub2, grad_y_sub2 = self.compute_gradients(subdomain2, ~self.interface)
        grad_x_sub1_interface, grad_y_sub1_interface = grad_x_sub1[:, 0, self.boundary], grad_y_sub1[:, 0, self.boundary]
        grad_x_sub2_interface, grad_y_sub2_interface = grad_x_sub2[:, 0, self.boundary], grad_y_sub2[:, 0, self.boundary]
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
