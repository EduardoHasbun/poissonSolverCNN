import torch
import torch.nn as nn
import numpy as np

class _ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(_ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class MSNet3D(nn.Module):
    def __init__(self, scales, kernels, input_res):
        super(MSNet3D, self).__init__()

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for i in range(len(scales)):
            down_layers = nn.ModuleList()
            up_layers = nn.ModuleList()

            for j in range(len(scales[i])):
                if j == 0:
                    down_layers.append(_ConvBlock3D(scales[i][j], scales[i][j+1], kernels[i]))
                else:
                    down_layers.append(_ConvBlock3D(scales[i][j], scales[i][j+1], kernels[i], pool=True))

            upsample_size = (int(input_res / 2 ** (len(scales) - i)), int(input_res / 2 ** (len(scales) - i)),
                             int(input_res / 2 ** (len(scales) - i)))

            for j in range(len(scales[i]) - 1, 0, -1):
                if j == len(scales[i]) - 1:
                    up_layers.append(_ConvBlock3D(scales[i][j+1], scales[i][j], kernels[i], upsample_size=upsample_size))
                else:
                    up_layers.append(_ConvBlock3D(scales[i][j+1] * 2, scales[i][j], kernels[i]))

            self.down_blocks.append(nn.Sequential(*down_layers))
            self.up_blocks.append(nn.Sequential(*up_layers))

    def forward(self, x):
        down_outputs = []
        for down_block in self.down_blocks:
            x = down_block(x)
            down_outputs.append(x)

        out = down_outputs[-1]
        for i, up_block in enumerate(self.up_blocks):
            out = torch.cat([out, down_outputs[-2-i]], dim=1)
            out = up_block(out)

        return out
