import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomPadLayer3D(nn.Module):
    def __init__(self):
        super(CustomPadLayer3D, self).__init__()
        self.padx = int((self.kernel_sizes[2] - 1) / 2)
        self.pady = int((self.kernel_sizes[1] - 1) / 2)
        self.padz = int((self.kernel_sizes[0] - 1) / 2)

    def forward(self, x):
        x = F.pad(x, (self.padx, self.padx, self.pady, self.pady, self.padz, self.padz), "constant", 0)
        return x


class _ConvBlock3D(nn.Module):
    def __init__(self, fmaps, block_type, kernel_size, padding_mode='zeros', upsample_mode='nearest', out_size=None):
        super(_ConvBlock3D, self).__init__()
        layers = []
        # Apply pooling on down and bottom blocks
        if block_type == 'down' or block_type == 'bottom':
            layers.append(nn.MaxPool3d(2))

        # Append all the specified layers
        for i in range(len(fmaps) - 1):
            layers.append(CustomPadLayer3D(kernel_size))
            layers.append(nn.Conv3d(fmaps[i], fmaps[i + 1], kernel_size=kernel_size, padding=0))
            layers.append(nn.ReLU())

        # Apply either Upsample or deconvolution
        if block_type == 'up' or block_type == 'bottom':
            layers.append(nn.Upsample(out_size, mode=upsample_mode))

        # Build the sequence of layers
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class UNet3D(nn.Module):
    def __init__(self, scales, kernel_sizes):
        super(UNet3D, self).__init__()
        self.scales = scales
        self.max_scale = len(scales) - 1
        self.kernel_sizes = kernel_sizes

        # Entry layer
        self.ConvsDown = nn.ModuleList()
        self.ConvsDown.append(_ConvBlock3D(self.scales[0][0], 'in', kernel_sizes[0]))

        # Intermediate down layers (with MaxPool at the beginning)
        for i in range(1, self.max_scale):
            self.ConvsDown.append(_ConvBlock3D(self.scales[i][0], 'down', kernel_sizes[i]))

        # Bottom layer (MaxPool at the beginning and Upsample/Deconv at the end)
        self.ConvBottom = _ConvBlock3D(self.scales[self.max_scale][0], 'bottom', kernel_sizes[self.max_scale])

        # Intemediate layers up (UpSample/Deconv at the end)
        self.ConvsUp = nn.ModuleList()
        for i in range(self.max_scale - 1, -1, -1):
            self.ConvsUp.append(_ConvBlock3D(self.scales[i][1], 'up', kernel_sizes[i]))

        # Out layer
        self.ConvsUp.append(_ConvBlock3D(self.scales[0][1], 'out', kernel_sizes[0]))

    def forward(self, x):
        # List of the temporary x that are used for linking with the up branch
        inputs_down = []

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
