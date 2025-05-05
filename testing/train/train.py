
# import torch
# from unet import UNet
# import yaml
# from torch.utils.data import DataLoader
# import numpy as np
# from operators import ratio_potrhs, LaplacianLoss, DirichletBoundaryLoss
# import torch.optim as optim

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# # Import external parameters
# with open('C:/Codigos/poissonSolverCNN_Gpu/testing/train/tain.yml', 'r') as file:
#     cfg = yaml.load(file, Loader=yaml.FullLoader)

# scales_data = cfg.get('arch', {}).get('scales', {})
# scales = [value for key, value in sorted(scales_data.items())]
# kernel_size = cfg['arch']['kernel_sizes']
# data_dir = cfg['data_loader']['data_dir']
# batch_size = cfg['data_loader']['batch_size']
# num_epochs = cfg['trainer']['epochs']
# lapl_weight = cfg['loss']['args']['lapl_weight']
# bound_weight = cfg['loss']['args']['bound_weight']
# lr = cfg['loss']['args']['optimizer_lr']
# xmin, xmax, ymin, ymax = cfg['globals']['xmin'], cfg['globals']['xmax'], cfg['globals']['ymin'], cfg['globals']['ymax']
# nnx, nny = cfg['globals']['nnx'], cfg['globals']['nny']
# Lx = xmax - xmin
# Ly = ymax - ymin

# # Create and normalize data
# alpha = 0.1
# ratio_max = ratio_potrhs(alpha, Lx, Ly)
# dataset_np = np.load(data_dir).astype(np.float64) * ratio_max
# dataset_tensor = torch.from_numpy(dataset_np).unsqueeze(1)  # Add channel dimension
# dataloader = DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True)

# # Create model and move to GPU
# model = model = UNet(scales, kernel_sizes=kernel_size, input_res=nnx).double().to(device)

# # Losses and optimizer
# laplacian_loss = LaplacianLoss(cfg, lapl_weight=lapl_weight).to(device)
# dirichlet_loss = DirichletBoundaryLoss(bound_weight).to(device)
# optimizer = optim.Adam(model.parameters(), lr=lr)

# # Training loop
# for epoch in range(num_epochs):
#     total_loss = 0
#     for batch_idx, batch in enumerate(dataloader):
#         data = batch.to(device).double()  # move to GPU
#         optimizer.zero_grad()
#         data_norm = torch.ones((data.size(0), data.size(1), 1, 1), device=device, dtype=torch.float64) / ratio_max

#         output = model(data)
#         loss = laplacian_loss(output, data=data, data_norm=data_norm)
#         loss += dirichlet_loss(output)

#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#         if batch_idx % 20 == 0:
#             print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

#     print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {total_loss / len(dataloader)}")

#     # Save model
#     torch.save(model.state_dict(), 'unet_model.pth')






# # import torch
# # from unet import UNet
# # import yaml
# # from torch.utils.data import DataLoader
# # import numpy as np
# # from operators import ratio_potrhs, LaplacianLoss, DirichletBoundaryLoss
# # import torch.optim as optim
# # import os
# # import argparse

# # # Setup device
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print(f"Using device: {device}")

# # # Import external parameters
# # parser = argparse.ArgumentParser(description='Training')
# # parser.add_argument('-c', '--cfg', type=str, default=None, help='Config filename')
# # args = parser.parse_args()

# # with open(args.cfg, 'r') as yaml_stream:
# #     cfg = yaml.safe_load(yaml_stream)

# # scales_data = cfg.get('arch', {}).get('scales', {})
# # scales = [value for key, value in sorted(scales_data.items())]
# # batch_size = cfg['data_loader']['batch_size']
# # num_epochs = cfg['trainer']['epochs']
# # lapl_weight = cfg['loss']['args']['lapl_weight']
# # bound_weight = cfg['loss']['args']['bound_weight']
# # lr = cfg['loss']['args']['optimizer_lr']
# # arch_model = cfg['arch']['model']
# # arch_type = cfg['arch']['type']
# # arch_dir = os.path.join('../../', cfg['arch']['arch_dir'])

# # with open(arch_dir) as yaml_stream1:
# #     arch = yaml.safe_load(yaml_stream1)

# # scales_data = arch.get(arch_type, {}).get('args', {}).get('scales', {})
# # scales = [value for key, value in sorted(scales_data.items())]
# # kernel_size = arch[arch_type]['args']['kernel_sizes']
# # xmin, xmax, ymin, ymax, nnx, nny = cfg['globals']['xmin'], cfg['globals']['xmax'],\
# #                                    cfg['globals']['ymin'], cfg['globals']['ymax'],\
# #                                    cfg['globals']['nnx'], cfg['globals']['nny']
# # data_dir = cfg['general']['data_dir']
# # Lx = xmax - xmin
# # Ly = ymax - ymin
# # save_dir = os.getcwd()
# # data_dir = os.path.join(save_dir, '..', data_dir)
# # save_dir = os.path.join(save_dir, 'trained_models')
# # os.makedirs(save_dir, exist_ok=True)
# # case_name = cfg['general']['name_case']

# # # Normalize
# # alpha = 0.1
# # ratio_max = ratio_potrhs(alpha, Lx, Ly)

# # # Data
# # dataset = np.load(data_dir).astype(np.float64) * ratio_max
# # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # # Model
# # model = UNet(scales, kernel_sizes=kernel_size, input_res=nnx)

# # # model = model.float().to(device)
# # model = model.double().to(device)

# # # Losses and optimizer
# # laplacian_loss = LaplacianLoss(cfg, lapl_weight).to(device)
# # Dirichlet_loss = DirichletBoundaryLoss(bound_weight).to(device)
# # optimizer = optim.Adam(model.parameters(), lr=lr)

# # laplacian_losses = []
# # dirichlet_losses = []
# # total_losses = []

# # # Training loop
# # for epoch in range(num_epochs):
# #     total_loss = 0
# #     for batch_idx, batch in enumerate(dataloader):
# #         data = batch[:, np.newaxis, :, :].to(device).double()
# #         data_norm = torch.ones((data.size(0), data.size(1), 1, 1), device=device) / ratio_max
        
# #         optimizer.zero_grad()
# #         output = model(data)

# #         laplacian_loss_value = laplacian_loss(output, data=data, data_norm=data_norm)
# #         dirichlet_loss_value = Dirichlet_loss(output)
# #         loss = laplacian_loss_value + dirichlet_loss_value

# #         loss.backward()
# #         optimizer.step()

# #         total_loss += loss.item()
# #         laplacian_losses.append(laplacian_loss_value.item())
# #         dirichlet_losses.append(dirichlet_loss_value.item())
# #         total_losses.append(loss.item())

# #         if batch_idx % 20 == 0:
# #             print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

# #     print(f"Epoch [{epoch + 1}/{num_epochs}] - Total Loss: {total_loss / len(dataloader)}")
    
# #     torch.save(model.state_dict(), os.path.join(save_dir, case_name + '.pth'))

# # # Save loss data
# # loss_file_path = os.path.join(save_dir, f"{case_name}_losses.txt")
# # with open(loss_file_path, "w") as f:
# #     f.write("Laplacian Losses:\n")
# #     f.write(", ".join(map(str, laplacian_losses)) + "\n\n")
# #     f.write("Dirichlet Losses:\n")
# #     f.write(", ".join(map(str, dirichlet_losses)) + "\n")

# # print(f"Losses saved to {loss_file_path}")







import torch
from unet import UNet
import yaml
from torch.utils.data import DataLoader
import numpy as np
from operators import ratio_potrhs, LaplacianLoss, DirichletBoundaryLoss
import torch.optim as optim
import argparse

# Import external parameters
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('-c', '--cfg', type=str, default=None, help='Config filename')
args = parser.parse_args()

with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)
scales_data = cfg.get('arch', {}).get('scales', {})
scales = [value for key, value in sorted(scales_data.items())]
kernel_size = cfg['arch']['kernel_sizes']
data_dir = cfg['data_loader']['data_dir']
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
model = UNet(scales, kernel_sizes=kernel_size, input_res=nnx)
model = model.double()
laplacian_loss = LaplacianLoss(cfg, lapl_weight=lapl_weight)
dirichlet_loss = DirichletBoundaryLoss(bound_weight)
optimizer = optim.Adam(model.parameters(), lr = lr)

#Train loop
for epoch in range (num_epochs):
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        data = batch[:, np.newaxis, :, :].double()
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
torch.save(model.state_dict(), 'unet_model.pth')