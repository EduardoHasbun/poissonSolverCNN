import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class _ConvBlock3D(nn.Module):
    """
    General convolution block for MSNet. Depending on the location of the block
    in the architecture, the block can begin with a MaxPool3d (for down)
    or end with an Upsample layer (for up)
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool=False, upsample_size=None):
        super(_ConvBlock3D, self).__init__()
        layers = []

        if pool:
            layers.append(nn.MaxPool3d(2))

        layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=1))
        layers.append(nn.ReLU())

        if upsample_size is not None:
            layers.append(nn.Upsample(size=upsample_size, mode='nearest'))

        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class MSNet3D(nn.Module):

    def __init__(self, scales, kernel_sizes, input_res):
        super(MSNet3D, self).__init__()

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # Create down_blocks and up_blocks
        for i in range(len(scales)):
            self.down_blocks.append(_ConvBlock3D(scales[i][0], scales[i][1], kernel_sizes[i], pool=True))

            if i != len(scales) - 1:
                self.up_blocks.append(_ConvBlock3D(scales[i][1], scales[i][0], kernel_sizes[i], upsample_size=(int(input_res / 2 ** (len(scales) - i - 1)), int(input_res / 2 ** (len(scales) - i - 1)), int(input_res / 2 ** (len(scales) - i - 1)))))

        # Out layer
        self.up_blocks.append(_ConvBlock3D(scales[-1][1], scales[-1][0], kernel_sizes[-1]))

    def forward(self, x):
        down_outputs = []
        # Apply the down loop
        for down_block in self.down_blocks:
            x = down_block(x)
            down_outputs.append(x)

        out = down_outputs[-1]
        # Apply the up loop
        for i, up_block in enumerate(self.up_blocks[:-1]):
            out = torch.cat([out, down_outputs[-2-i]], dim=1)
            out = up_block(out)

        # Last up block (no concatenation)
        out = self.up_blocks[-1](out)

        return out
