import torch
import torch.nn as nn
import numpy as np

class _ConvBlock3D(nn.Module):
    def __init__(self, scales, kernel_size, pool=False, upsample_size=None, last_one=False):
        super(_ConvBlock3D, self).__init__()
        layers = list()

        if pool:
            layers.append(nn.MaxPool3d(2))

        for i in range(len(scales) - 1):
            layers.append(nn.Conv3d(scales[i], scales[i + 1], kernel_size=kernel_size,
                                    stride=1, padding=1))
            layers[-1].bias.data = layers[-1].bias.data.to(torch.double)

            if not last_one:
                layers.append(nn.ReLU())
            else:
                if i != len(scales) - 2:
                    layers.append(nn.ReLU())

        if upsample_size is not None:
            layers.append(nn.Upsample(size=upsample_size, mode='nearest'))

        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class UNet3D(nn.Module):
    def __init__(self, scales, kernel):
        super(UNet3D, self).__init__()

        self.ConvsDown = nn.ModuleList([
            _ConvBlock3D(scales[0][0], kernel),
            _ConvBlock3D(scales[1][0], kernel, pool=True),
            _ConvBlock3D(scales[2][0], kernel, pool=True),
            _ConvBlock3D(scales[3][0], kernel, pool=True),
        ])

        self.ConvBottom = _ConvBlock3D(scales[4][0], kernel, pool=True, upsample_size=6)

        self.ConvsUp = nn.ModuleList([
            _ConvBlock3D(scales[1][1], kernel, upsample_size=25),
            _ConvBlock3D(scales[2][1], kernel, upsample_size=50),
            _ConvBlock3D(scales[3][1], kernel, upsample_size=101),
            _ConvBlock3D(scales[0][1], kernel, last_one=True),
        ])

    def forward(self, x):
        # List of the temporary x that are used for linking with the up branch
        inputs_down = list()

        # Apply the down loop
        for ConvDown in self.ConvsDown:
            x = ConvDown(x)
            inputs_down.append(x)

        # Bottom part of the U
        x = self.ConvBottom(x)

        # Apply the up loop
        for ConvUp in self.ConvsUp:
            input_tmp = inputs_down.pop()
            x = ConvUp(torch.cat((x, input_tmp), dim=1))


        return x
