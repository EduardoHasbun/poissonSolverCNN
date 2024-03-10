import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock3D(nn.Module):
    def __init__(self, fmaps, out_size, block_type, kernel_size, 
            padding_mode='zeros', upsample_mode='trilinear'):
        super(_ConvBlock3D, self).__init__()
        layers = list()
        # Append all the specified layers
        for i in range(len(fmaps) - 1):
            layers.append(nn.Conv3d(fmaps[i], fmaps[i + 1], 
                kernel_size=kernel_size, padding=int((kernel_size[0] - 1) / 2),
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
    def __init__(self, scales, kernel_sizes, input_res, padding_mode='zeros',
                 upsample_mode='trilinear'):
        super(MSNet3D, self).__init__()
        # For upsample the list of resolution is needed when 
        # the number of points is not a power of 2
        self.scales = scales
        self.n_scales = len(scales)
        self.max_scale = self.n_scales - 1
        self.input_res = tuple([input_res, input_res, input_res])
        self.list_res = [int(input_res / 2**i) for i in range(self.n_scales)]
        if isinstance(kernel_sizes, int):
            self.kernel_sizes = [tuple([kernel_sizes, kernel_sizes])] * len(scales)
        elif isinstance(kernel_sizes, list):
            if isinstance(kernel_sizes[0], list):
                self.kernel_sizes = [tuple(kernel_sizes[0])] * len(scales)
            else:
                # Convert the list of integers to a list of tuples
                self.kernel_sizes = [tuple([ks, ks]) for ks in self.kernel_sizes]

        # create down_blocks, bottom_fmaps and up_blocks
        middle_blocks = list()
        for local_depth in range(self.n_scales):
            middle_blocks.append(self.scales[self.max_scale - local_depth])
        out_fmaps = self.scales[0]

        # Intemediate layers up (UpSample/Deconv at the end)
        self.ConvsUp = nn.ModuleList()
        for imiddle, middle_fmaps in enumerate(middle_blocks):
            print("Length of list_res:", len(self.list_res))
            print("Value of imiddle:", imiddle)
            self.ConvsUp.append(_ConvBlock3D(middle_fmaps, 
                out_size=self.list_res[-2 -imiddle], 
                block_type='middle', kernel_size=self.kernel_sizes[-1 - imiddle],
                padding_mode=padding_mode, upsample_mode=upsample_mode))
        
        # Out layer
        self.ConvsUp.append(_ConvBlock3D(out_fmaps, 
            out_size=self.list_res[0],
            block_type='out', kernel_size=self.kernel_sizes[0], padding_mode=padding_mode))

    def forward(self, x):
        initial_map = x
        # Apply the up loop
        for iconv, ConvUp in enumerate(self.ConvsUp):
            # First layer of convolution doesn't need concatenation
            if iconv == 0:
                x = ConvUp(x)
            else:
                tmp_map = F.interpolate(initial_map, x[0, 0, 0].shape, mode='trilinear', align_corners=False)
                x = ConvUp(torch.cat((x, tmp_map), dim=1))
                
        return x


