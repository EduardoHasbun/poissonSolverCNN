import torch
import torch.nn as nn
import torch.nn.functional as F

# Create the model
class CustomPadLayer3D(nn.Module):
    def __init__(self, kernel_sizes):
        super(CustomPadLayer3D, self).__init__()
        self.padx = int((kernel_sizes[2] - 1) / 2)
        self.pady = int((kernel_sizes[1] - 1) / 2)
        self.padz = int((kernel_sizes[0] - 1) / 2)

    def forward(self, x):
        x = F.pad(x, (self.padx, self.padx, self.pady, self.pady, self.padz, self.padz), "constant", 0)
        return x

class ConvBlock3D(nn.Module):
    def __init__(self, fmaps, block_type, kernel_size,kernel_sizes, padding_mode='zeros', upsample_mode='nearest', out_size=None):
        super(ConvBlock3D, self).__init__()
        layers = list()

        # Apply pooling on down and bottom blocks
        if block_type == 'down' or block_type == 'bottom':
            layers.append(nn.MaxPool3d(2))

        # Append all the specified layers
        for i in range(len(fmaps) - 1):
            if padding_mode == 'custom':
                layers.append(CustomPadLayer3D(kernel_size))
                layers.append(nn.Conv3d(fmaps[i], fmaps[i + 1], 
                    kernel_size=kernel_size, padding=0, 
                    padding_mode='zeros'))
            else:
                layers.append(nn.Conv3d(fmaps[i], fmaps[i + 1], 
                    kernel_size=kernel_size, 
                    padding=(int((kernel_size[0] - 1) / 2), int((kernel_size[0] - 1) / 2)), 
                    padding_mode=padding_mode))
            # No ReLu at the very last layer
            if i != len(fmaps) - 2 or block_type != 'out':
                layers.append(nn.ReLU())

        # Apply either Upsample or deconvolution
        if block_type == 'up' or block_type == 'bottom':
            layers.append(nn.Upsample(out_size, mode=upsample_mode))

        # Build the sequence of layers
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class UNet3D(nn.Module):

    def __init__(self, scales, kernel_sizes, input_res, 
                    padding_mode='zeros', upsample_mode='nearest'):
        super(UNet3D, self).__init__(scales, kernel_sizes)
        self.scales = scales
        self.max_scale = len(scales) - 1
        self.kernel_sizes = kernel_sizes
        
        # create down_blocks, bottom_fmaps and up_blocks
        in_fmaps = self.scales[0][0]

        down_blocks = list()
        for local_depth in range(1, self.max_scale):
            down_blocks.append(self.scales[local_depth][0])
        
        bottom_fmaps = self.scales[self.max_scale]

        up_blocks = list()
        for local_depth in range(self.max_scale - 1, 0, -1):
            up_blocks.append(self.scales[local_depth][1])
        
        out_fmaps = self.scales[0][1]
        
        # For upsample the list of resolution is needed when 
        # the number of points is not a power of 2
        if isinstance(input_res, list):
            self.input_res = tuple(input_res)
            list_res = [(int(input_res[0] / 2**i), int(input_res[1] / 2**i)) for i in range(self.max_scale)]
        else:
            self.input_res = tuple([input_res, input_res])
            list_res = [int(input_res / 2**i) for i in range(self.max_scale)]

        # Entry layer
        self.ConvsDown = nn.ModuleList()
        self.ConvsDown.append(ConvBlock3D(in_fmaps, 'in', self.kernel_sizes[0], padding_mode=padding_mode))

        # Intermediate down layers (with MaxPool at the beginning)
        for idown, down_fmaps in enumerate(down_blocks):
            self.ConvsDown.append(ConvBlock3D(down_fmaps, 'down', self.kernel_sizes[idown + 1],
                padding_mode=padding_mode))

        # Bottom layer (MaxPool at the beginning and Upsample/Deconv at the end)
        self.ConvBottom = ConvBlock3D(bottom_fmaps, 'bottom', self.kernel_sizes[-1],
                padding_mode=padding_mode, upsample_mode=upsample_mode, out_size=list_res.pop())

        # Intemediate layers up (UpSample/Deconv at the end)
        self.ConvsUp = nn.ModuleList()
        for iup, up_fmaps in enumerate(up_blocks):
            self.ConvsUp.append(ConvBlock3D(up_fmaps, 'up', self.kernel_sizes[-2 - iup], 
                padding_mode=padding_mode, upsample_mode=upsample_mode, out_size=list_res.pop()))
        
        # Out layer
        self.ConvsUp.append(ConvBlock3D(out_fmaps, 'out', self.kernel_sizes[0],
                padding_mode=padding_mode))

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
