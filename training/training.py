import torch
from unet import UNet
import yaml
from torch.utils.data import DataLoader
import numpy as np
from operators import ratio_potrhs, LaplacianLoss, DirichletBoundaryLoss
import torch.optim as optim
import os

#Import external parameteres
with open('C:\Codigos/poissonSolverCNN/training/train.yml', 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
scales_data = cfg.get('arch', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
kernel_size = cfg['arch']['kernel_sizes']
data_dir = cfg['data_loader']['data_dir']
save_dir = cfg['trainer']['save_dir']
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



#Create Data
dataset = np.load(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#Parameters to Nomalize
alpha = 0.1
ratio_max = ratio_potrhs(alpha, Lx, Ly)



#Create model and losses
model = UNet(scales, kernel=kernel_size)
model = model.double()
laplacian_loss = LaplacianLoss(cfg, lapl_weight=lapl_weight)
dirichlet_loss = DirichletBoundaryLoss(bound_weight)
optimizer = optim.Adam(model.parameters(), lr = lr)

#Train loop
for epoch in range (num_epochs):
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        data = batch[:, np.newaxis, :, :]
        optimizer.zero_grad()
        data = torch.DoubleTensor(data) 
        optimizer.zero_grad()
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


# Save the trained model
torch.save(model.state_dict(), os.path.join(save_dir, 'unet_model.pth'))



