import torch
from unet3d import UNet3D
from msnet3d import MSNet3D
import yaml
from torch.utils.data import DataLoader
import numpy as np
from operators3d import ratio_potrhs, LaplacianLoss, DirichletBoundaryLoss, InsideLoss
import torch.optim as optim
import os
import argparse
from torch.utils.data import TensorDataset

#Import external parameteres
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('-c', '--cfg', type=str, default=None, help='Config filename')
args = parser.parse_args()
with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)
scales_data = cfg.get('arch', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
kernel_size = cfg['arch']['kernel_sizes']
model_type = cfg['arch']['type']
batch_size = cfg['data_loader']['batch_size']
num_epochs = cfg['trainer']['epochs']
lapl_weight = cfg['loss']['args']['lapl_weight']
inside_weight = cfg['loss']['args']['inside_weight']
bound_weight = cfg['loss']['args']['bound_weight']
lr = cfg['loss']['args']['optimizer_lr']
scales_data = cfg.get('arch', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
kernel_sizes = cfg['arch']['kernel_sizes']
xmin, xmax, ymin, ymax, zmax, zmin, nnx, nny, nnz = cfg['globals']['xmin'], cfg['globals']['xmax'],\
            cfg['globals']['ymin'], cfg['globals']['ymax'], cfg['globals']['zmin'], cfg['globals']['zmax'],\
            cfg['globals']['nnx'], cfg['globals']['nny'], cfg['globals']['nnz']         
Lx = xmax-xmin
Ly = ymax-ymin
Lz = zmax-zmin
save_dir = os.getcwd()
data_dir = os.path.join(save_dir, '..', 'dataset', 'generated', 'fields.npy')
target_dir = os.path.join(save_dir, '..', 'dataset', 'generated', 'potentials.npy')


#Create Data
dataset = np.load(data_dir)
target  = np.load(target_dir)
dataset = torch.tensor(dataset)
target = torch.tensor(target)
data_set = TensorDataset(dataset, target)
dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

#Parameters to Nomalize
alpha = 0.1
ratio_max = ratio_potrhs(alpha, Lx, Ly, Lz)



#Create model and losses
if model_type == 'UNet':
    model = UNet3D(scales, kernel_sizes=kernel_size)
    print('Using UNet model')
elif model_type == 'MSNet':
    model = MSNet3D(scales=scales, kernel_sizes=kernel_size, input_res=nnx)
    print('Using MSNet model')
else:
    print('No model found')

model = model.float() 
laplacian_loss = LaplacianLoss(cfg, lapl_weight=lapl_weight)
dirichlet_loss = DirichletBoundaryLoss(bound_weight)
inside_loss = InsideLoss(cfg, inside_weight=inside_weight)
optimizer = optim.Adam(model.parameters(), lr = lr)

#Train loop
for epoch in range (num_epochs):
    total_loss = 0
    for batch_idx, (batch, target) in enumerate(dataloader):
        data = batch[:, np.newaxis, :, :].float()
        target = target[:, np.newaxis, :, :].float()
        optimizer.zero_grad()
        # data = data.to(model.parameters().__next__().dtype)
        optimizer.zero_grad()
        # data_norm = torch.ones((data.size(0), data.size(1), 1, 1))# / ratio_max
        output = model(data)
        # loss = laplacian_loss(output, data = data, data_norm = data_norm)
        loss = inside_loss(output, target)
        loss += dirichlet_loss(output)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 20 ==0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {total_loss / len(dataloader)}")
    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))