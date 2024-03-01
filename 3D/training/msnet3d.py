import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock3D(nn.Module):
    def __init__(self, fmaps, out_size, block_type, kernel_size, 
            padding_mode='zeros', upsample_mode='bilinear'):
        super(_ConvBlock3D, self).__init__()
        layers = list()
        # Append all the specified layers
        for i in range(len(fmaps) - 1):
            layers.append(nn.Conv3d(fmaps[i], fmaps[i + 1], 
                kernel_size=kernel_size, padding=int((kernel_size - 1) / 2),
                padding_mode=padding_mode))
            # No ReLu at the very last layer
            if i != len(fmaps) - 2 or block_type != 'out':
                layers.append(nn.ReLU())

        # Apply either Upsample or deconvolution
        if block_type == 'middle':
            layers.append(nn.Upsample(out_size, mode=upsample_mode))

        # Build the sequence of layers
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class MSNet3D(nn.Module):
    def __init__(self, scales, kernel, input_res, padding_mode='zeros', upsample_mode='bilinear'):
        super(MSNet3D, self).__init__()
        # For upsample the list of resolution is needed when 
        # the number of points is not a power of 2
        self.input_res = tuple([input_res, input_res, input_res])
        self.n_scales = len(scales)
        self.kernel = kernel
        self.scales = scales
        print(scales)
        self.max_scales = int(self.n_scales)
        self.list_res = [int(input_res / 2**i) for i in range(self.n_scales)]

        # create down_blocks, bottom_fmaps and up_blocks
        middle_blocks = list()
        for local_depth in range(self.n_scales):
            middle_blocks.append(self.scales[f'scale_{self.max_scales - local_depth:d}'])
        out_fmaps = self.scales['scale_0']

        # Intemediate layers up (UpSample/Deconv at the end)
        self.ConvsUp = nn.ModuleList()
        for imiddle, middle_fmaps in enumerate(middle_blocks):
            self.ConvsUp.append(_ConvBlock3D(middle_fmaps, 
                out_size=self.list_res[-2 -imiddle], 
                block_type='middle', kernel_size=self.kernel[-1 - imiddle],
                padding_mode=padding_mode, upsample_mode=upsample_mode))
        
        # Out layer
        self.ConvsUp.append(_ConvBlock3D(out_fmaps, 
            out_size=self.list_res[0],
            block_type='out', kernel_size=self.kernel[0], padding_mode=padding_mode))

    def forward(self, x):
        initial_map = x
        # Apply the up loop
        for iconv, ConvUp in enumerate(self.ConvsUp):
            # First layer of convolution doesn't need concatenation
            if iconv == 0:
                x = ConvUp(x)
            else:
                tmp_map = F.interpolate(initial_map, x[0, 0].shape, mode='bilinear', align_corners=False)
                x = ConvUp(torch.cat((x, tmp_map), dim=1))
                
        return x

