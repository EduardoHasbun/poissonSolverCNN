#############################################################################################################
#                                                                                                           #
#                           RUN:    python solver.py -c solver.yml                                          #
#                                                                                                           #
#############################################################################################################




import torch
import torch.nn as nn
import numpy as np
import os
import yaml
import argparse
import matplotlib.pyplot as plt


#Define generals Parameters
os.environ['OPENBLAS_NUM_THREADS'] = '1'

args = argparse.ArgumentParser(description='Training')
args.add_argument('-c', '--cfg', type=str, default=None,
                help='Config filename')
args.add_argument('--case', type=str, default=None, help='Case name')
args = args.parse_args()
with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)

xmin, xmax, nnx = cfg['mesh']['xmin'], cfg['mesh']['xmax'], cfg['mesh']['nnx']
ymin, ymax, nny  = cfg['mesh']['ymin'], cfg['mesh']['ymax'], cfg['mesh']['nny']
x_1d = np.linspace(xmin, xmax, nnx)
y_1d = np.linspace(ymin, ymax, nny)
X, Y = np.meshgrid(x_1d, y_1d)


#Define U-Net architecture
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Define the encoder part
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Define the decoder part
        self.decoder = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(24, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Adjust activation function based on your task
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#Define Gaussians's Functions
def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x)**2
                              - ((y - y0) / sigma_y)**2)
def gaussians(x, y, params):
    profile = np.zeros_like(x)
    ngauss = int(len(params) / 5)
    params = np.array(params).reshape(ngauss, 5)
    for index in range(ngauss):
        profile += gaussian(x, y, *params[index, :])
    return profile

input_data = gaussians(X, Y, cfg['init']['args']).astype(np.float32)
input_tensor = torch.from_numpy(input_data).float()  # Convert to float32
input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
desired_input_size = (101, 101)
input_tensor = torch.nn.functional.interpolate(input_tensor, size=desired_input_size)

# Load the pretrained model
model = UNet(1, 1) 
model.load_state_dict(torch.load('C:/Codigos/poissonSolverCNN/training/model_best.pth'))
model.eval() 


with torch.no_grad():
    output = model(input_tensor)

output_array = output.numpy()[0, 0, :, :]
plt.imshow(output_array, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='viridis')
plt.colorbar(label='results')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
