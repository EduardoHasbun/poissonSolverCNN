import torch
from unet import UNet
import yaml
from torch.utils.data import DataLoader
import numpy as np
from operators import ratio_potrhs, LaplacianLoss, DirichletBoundaryLoss
import torch.optim as optim
import os
import argparse

#Import external parameteres
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('-c', '--cfg', type=str, default=None, help='Config filename')
args = parser.parse_args()
with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)
scales_data = cfg.get('arch', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
kernel_size = cfg['arch']['kernel_sizes']
batch_size = cfg['data_loader']['batch_size']
num_epochs = cfg['trainer']['epochs']
lapl_weight = cfg['loss']['args']['lapl_weight']
bound_weight = cfg['loss']['args']['bound_weight']
lr = cfg['loss']['args']['optimizer_lr']
scales_data = cfg.get('arch', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
kernel_sizes = cfg['arch']['kernel_sizes']
xmin, xmax, ymin, ymax, nnx, nny = cfg['globals']['xmin'], cfg['globals']['xmax'],\
            cfg['globals']['ymin'], cfg['globals']['ymax'], cfg['globals']['nnx'], cfg['globals']['nny']
interface_center = (cfg['globals']['interface_center']['x'], cfg['globals']['interface_center']['y'])
interface_radius = cfg['globals']['interface_radius']
epsilon_inside = cfg['globals']['epsilon_inside']
epsilon_outside = cfg['globals']['epsilon_outside']
Lx = xmax-xmin
Ly = ymax-ymin
save_dir = os.getcwd()
data_dir = os.path.join(save_dir, '..', 'dataset', 'generated', 'domain.npy')


# Parameters for data
x, y= np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny)
X, Y = np.meshgrid(x,y)
interface_mask = (X - interface_center[0])**2 + (Y - interface_center[1])**2 <= interface_radius**2
interface_boundary = np.zeros_like(interface_mask, dtype=bool)
for i in range(1, interface_mask.shape[0]):
    for j in range(1, interface_mask.shape[1]):
        if interface_mask[i, j] != interface_mask[i - 1, j]:
            interface_boundary[i, j] = True
        elif interface_mask[i, j] != interface_mask[i, j - 1]:
            interface_boundary[i, j] = True


# Load Data
data = np.load(data_dir)
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)


# Parameters to Nomalize
alpha = 0.1
# ratio_max = ratio_potrhs(alpha, Lx, Ly)
ratio_max = 1


#Create model and losses
model = UNet(scales, kernel_sizes=kernel_size, input_res=nnx)
model= model.double()
laplacian_loss = LaplacianLoss(cfg, lapl_weight, epsilon_inside, epsilon_outside, interface_mask)
dirichlet_loss = DirichletBoundaryLoss(bound_weight)
# interface_loss = InterfaceBoundaryLoss(bound_weight, interface_boundary)
optimizer = optim.Adam(model.parameters(), lr = lr)


#Train loop
for epoch in range (num_epochs):
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        data = batch[:, np.newaxis, :, :]
        optimizer.zero_grad()
        data = torch.DoubleTensor(data) 
        data_norm = torch.ones((data.size(0), data.size(1), 1, 1)) / ratio_max
        output = model(data)
        loss = laplacian_loss(output, data = data, data_norm = data_norm)
        loss += dirichlet_loss(output)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 20 ==0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {total_loss / len(dataloader)}")
    torch.save(model.state_dict(), os.path.join(save_dir, 'interface_model_4.pth'))
