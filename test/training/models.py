import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- Custom Padding for 3D --------
class CustomPadLayer3D(nn.Module):
    def __init__(self, kernel_size):
        super(CustomPadLayer3D, self).__init__()
        self.padx = int((kernel_size - 1) / 2)
        self.pady = int((kernel_size - 1) / 2)
        self.padz = int((kernel_size - 1) / 2)

    def forward(self, x):
        return F.pad(x, (self.padx, self.padx, self.pady, self.pady, self.padz, self.padz), "constant", 0)

# -------- 3D Conv Block --------
class ConvBlock3D(nn.Module):
    def __init__(self, fmaps, block_type, kernel_size, padding_mode='zeros', upsample_mode='nearest', out_size=None):
        super(ConvBlock3D, self).__init__()
        layers = []

        # Downsampling
        if block_type in ['down', 'bottom']:
            layers.append(nn.MaxPool3d(2))

        # Convolutions
        for i in range(len(fmaps) - 1):
            if padding_mode == 'custom':
                layers.append(CustomPadLayer3D(kernel_size))
                layers.append(nn.Conv3d(fmaps[i], fmaps[i + 1], kernel_size=kernel_size, padding=0))
            else:
                pad = int((kernel_size - 1) / 2)
                layers.append(nn.Conv3d(fmaps[i], fmaps[i + 1], kernel_size=kernel_size,
                                        padding=(pad, pad, pad), padding_mode=padding_mode))

            # No activation at final output layer
            if i != len(fmaps) - 2 or block_type != 'out':
                layers.append(nn.ReLU())

        # Upsampling
        if block_type in ['up', 'bottom']:
            layers.append(nn.Upsample(out_size, mode=upsample_mode))

        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

# -------- Full 3D UNet --------
class UNet3D(nn.Module):
    def __init__(self, scales, kernel_sizes, input_res, padding_mode='zeros', upsample_mode='nearest'):
        super(UNet3D, self).__init__()
        self.scales = scales
        self.max_scale = len(scales) - 1

        if isinstance(kernel_sizes, int):   
            self.kernel_sizes = [kernel_sizes] * len(scales)
        else:   
            self.kernel_sizes = kernel_sizes
        
        in_fmaps = self.scales[0][0]  # e.g., 2 channels: [source, mask]

        down_blocks = [self.scales[i][0] for i in range(1, self.max_scale)]
        bottom_fmaps = self.scales[self.max_scale]
        up_blocks = [self.scales[i][1] for i in range(self.max_scale - 1, 0, -1)]
        out_fmaps = self.scales[0][1]

        # Prepare resolution list
        if isinstance(input_res, list) and len(input_res) == 3:
            self.input_res = tuple(input_res)
            list_res = [(int(input_res[0] / 2**i),
                         int(input_res[1] / 2**i),
                         int(input_res[2] / 2**i)) for i in range(self.max_scale)]
        else:
            raise ValueError("input_res must be a list of 3 integers for 3D input")

        # Downward path
        self.ConvsDown = nn.ModuleList()
        self.ConvsDown.append(ConvBlock3D(in_fmaps, 'in', self.kernel_sizes[0], padding_mode=padding_mode))
        for idown, down_fmaps in enumerate(down_blocks):
            self.ConvsDown.append(ConvBlock3D(down_fmaps, 'down', self.kernel_sizes[idown + 1],
                                              padding_mode=padding_mode))

        # Bottom layer
        self.ConvBottom = ConvBlock3D(bottom_fmaps, 'bottom', self.kernel_sizes[-1],
                                      padding_mode=padding_mode, upsample_mode=upsample_mode,
                                      out_size=list_res.pop())

        # Upward path
        self.ConvsUp = nn.ModuleList()
        for iup, up_fmaps in enumerate(up_blocks):
            self.ConvsUp.append(ConvBlock3D(up_fmaps, 'up', self.kernel_sizes[-2 - iup],
                                            padding_mode=padding_mode, upsample_mode=upsample_mode,
                                            out_size=list_res.pop()))

        # Output layer
        self.ConvsUp.append(ConvBlock3D(out_fmaps, 'out', self.kernel_sizes[0], padding_mode=padding_mode))

    def forward(self, x):
        inputs_down = []

        for ConvDown in self.ConvsDown:
            x = ConvDown(x)
            inputs_down.append(x)

        x = self.ConvBottom(x)

        for ConvUp in self.ConvsUp:
            input_tmp = inputs_down.pop()
            x = ConvUp(torch.cat((x, input_tmp), dim=1))

        return x

# -------- Combined UNet for 3D domains --------
class UNet3DCombined(nn.Module):
    def __init__(self, scales, kernel_sizes, input_res, padding_mode='zeros', upsample_mode='nearest'):
        super(UNet3DCombined, self).__init__()
        self.model = UNet3D(scales, kernel_sizes, input_res,
                            padding_mode=padding_mode,
                            upsample_mode=upsample_mode)

    def forward(self, x):
        return self.model(x)  # x shape: (B, 2, D, H, W)
