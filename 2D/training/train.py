import torch
from unet import UNet
import yaml
from torch.utils.data import DataLoader
import numpy as np
from operators import ratio_potrhs, LaplacianLoss, DirichletBoundaryLoss, DirichletBoundaryLossFunction
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
Lx = xmax-xmin
Ly = ymax-ymin
save_dir = os.getcwd()
data_dir = os.path.join(save_dir, '..', 'dataset', 'generated', 'random_data_5.npy')


#Parameters to Nomalize
alpha = 0.1
ratio_max = ratio_potrhs(alpha, Lx, Ly)


#Create Data
dataset = np.load(data_dir) 
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


#Create model and losses
model = UNet(scales, kernel_sizes=kernel_size, input_res = nnx)
model = model.float()
laplacian_loss = LaplacianLoss(cfg, lapl_weight=lapl_weight)
# dirichlet_loss = DirichletBoundaryLoss(bound_weight)
dirichlet_loss_function = DirichletBoundaryLossFunction(bound_weight, xmin, xmax, ymin, ymax, nnx, nny)
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
        # loss = laplacian_loss(output, data = data, data_norm = data_norm)
        loss = dirichlet_loss_function(output, data_norm)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 20 ==0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {total_loss / len(dataloader)}")
    torch.save(model.state_dict(), os.path.join(save_dir, 'test_7.pth'))

