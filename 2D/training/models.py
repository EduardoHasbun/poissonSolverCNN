import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================================================
#                           UNet
# ===================================================================

class CustomPadLayer(nn.Module):
    """Custom padding layer with replicate + constant padding."""
    def __init__(self, kernel_size):
        super().__init__()
        self.padx = (kernel_size[1] - 1) // 2
        self.pady = (kernel_size[0] - 1) // 2

    def forward(self, x):
        x = F.pad(x, (0, 0, self.pady, 0), mode="replicate")
        x = F.pad(x, (self.padx, self.padx, 0, self.pady), mode="constant", value=0)
        return x


class ConvBlock(nn.Module):
    """Generic convolutional block for UNet with optional pooling and upsampling."""
    def __init__(self, fmaps, block_type, kernel_size,
                 padding_mode="zeros", upsample_mode="nearest", out_size=None):
        super().__init__()
        layers = []

        # Downsampling with pooling
        if block_type in {"down", "bottom"}:
            layers.append(nn.MaxPool2d(2))

        # Convolution + activation layers
        for i in range(len(fmaps) - 1):
            if padding_mode == "custom":
                layers.append(CustomPadLayer(kernel_size))
                layers.append(nn.Conv2d(
                    fmaps[i], fmaps[i + 1],
                    kernel_size=kernel_size,
                    padding=0,
                    padding_mode="zeros"
                ))
            else:
                layers.append(nn.Conv2d(
                    fmaps[i], fmaps[i + 1],
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                    padding_mode=padding_mode
                ))

            # Add ReLU except for last layer of "out" block
            if i != len(fmaps) - 2 or block_type != "out":
                layers.append(nn.ReLU())

        # Upsampling in up or bottom blocks
        if block_type in {"up", "bottom"}:
            layers.append(nn.Upsample(out_size, mode=upsample_mode))

        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class UNet(nn.Module):
    """Classic UNet architecture with customizable scales and kernel sizes."""
    def __init__(self, scales, kernel_sizes, input_res,
                 padding_mode="zeros", upsample_mode="nearest"):
        super().__init__()
        self.scales = scales
        self.max_scale = len(scales) - 1

        # Normalize kernel sizes into a list
        self.kernel_sizes = (
            [kernel_sizes] * len(scales) if isinstance(kernel_sizes, int) else kernel_sizes
        )

        # Build block configurations
        in_fmaps = scales[0][0]
        down_blocks = [scales[d][0] for d in range(1, self.max_scale)]
        bottom_fmaps = scales[self.max_scale]
        up_blocks = [scales[d][1] for d in range(self.max_scale - 1, 0, -1)]
        out_fmaps = scales[0][1]

        # Compute resolutions (handle non power-of-2 cases)
        if isinstance(input_res, list):
            self.input_res = tuple(input_res)
            list_res = [(input_res[0] // 2**i, input_res[1] // 2**i) for i in range(self.max_scale)]
        else:
            self.input_res = (input_res, input_res)
            list_res = [input_res // 2**i for i in range(self.max_scale)]

        # Entry block
        self.ConvsDown = nn.ModuleList([
            ConvBlock(in_fmaps, "in", self.kernel_sizes[0], padding_mode=padding_mode)
        ])

        # Downsampling blocks
        for idx, down_fmaps in enumerate(down_blocks):
            self.ConvsDown.append(ConvBlock(
                down_fmaps, "down", self.kernel_sizes[idx + 1], padding_mode=padding_mode
            ))

        # Bottom block
        self.ConvBottom = ConvBlock(
            bottom_fmaps, "bottom", self.kernel_sizes[-1],
            padding_mode=padding_mode, upsample_mode=upsample_mode,
            out_size=list_res.pop()
        )

        # Upsampling blocks
        self.ConvsUp = nn.ModuleList()
        for idx, up_fmaps in enumerate(up_blocks):
            self.ConvsUp.append(ConvBlock(
                up_fmaps, "up", self.kernel_sizes[-2 - idx],
                padding_mode=padding_mode, upsample_mode=upsample_mode,
                out_size=list_res.pop()
            ))

        # Output block
        self.ConvsUp.append(ConvBlock(
            out_fmaps, "out", self.kernel_sizes[0], padding_mode=padding_mode
        ))

    def forward(self, x):
        # Down path
        inputs_down = []
        for conv in self.ConvsDown:
            x = conv(x)
            inputs_down.append(x)

        # Bottom
        x = self.ConvBottom(x)

        # Up path
        for conv in self.ConvsUp:
            skip = inputs_down.pop()
            x = conv(torch.cat((x, skip), dim=1))

        return x


# ===================================================================
#                           MSNet
# ===================================================================

class _ConvBlockMSNnet(nn.Module):
    """Convolutional block for MSNet with optional upsampling."""
    def __init__(self, fmaps, out_size, block_type, kernel_size,
                 padding_mode="zeros", upsample_mode="bilinear"):
        super().__init__()
        layers = []

        # Convolution layers
        for i in range(len(fmaps) - 1):
            layers.append(nn.Conv2d(
                fmaps[i], fmaps[i + 1],
                kernel_size=kernel_size,
                padding=(kernel_size[0] - 1) // 2,
                padding_mode=padding_mode, stride=1
            ))

            # Add ReLU except for last layer of "out" block
            if i != len(fmaps) - 2 or block_type != "out":
                layers.append(nn.ReLU())

        # Optional upsampling
        if block_type == "middle":
            layers.append(nn.Upsample(out_size, mode=upsample_mode))

        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class MSNet(nn.Module):
    """Multi-scale network (MSNet) with hierarchical upsampling and concatenation."""
    def __init__(self, scales, kernel_sizes, input_res,
                 padding_mode="zeros", upsample_mode="bilinear"):
        super().__init__()
        self.scales = scales
        self.n_scales = len(scales)
        self.max_scale = self.n_scales - 1
        self.input_res = (input_res, input_res, input_res)
        self.list_res = [input_res // 2**i for i in range(self.n_scales)]

        # Normalize kernel sizes
        if isinstance(kernel_sizes, int):
            self.kernel_sizes = [(kernel_sizes, kernel_sizes)] * len(scales)
        elif isinstance(kernel_sizes, list):
            if isinstance(kernel_sizes[0], list):
                self.kernel_sizes = [tuple(ks) for ks in kernel_sizes]
            else:
                self.kernel_sizes = [(ks, ks, ks) for ks in kernel_sizes]

        # Middle and output blocks
        middle_blocks = [scales[self.max_scale - d] for d in range(self.n_scales)]
        out_fmaps = scales[0]

        self.ConvsUp = nn.ModuleList()
        for idx, fmaps in enumerate(middle_blocks):
            self.ConvsUp.append(_ConvBlockMSNnet(
                fmaps, out_size=self.list_res[-1 - idx],
                block_type="middle", kernel_size=self.kernel_sizes[-1 - idx],
                padding_mode=padding_mode, upsample_mode=upsample_mode
            ))

        self.ConvsUp.append(_ConvBlockMSNnet(
            out_fmaps, out_size=self.list_res[0],
            block_type="out", kernel_size=self.kernel_sizes[0],
            padding_mode=padding_mode
        ))

    def forward(self, x):
        initial_map = x
        for idx, conv in enumerate(self.ConvsUp):
            if idx == 0:
                x = conv(x)
            else:
                tmp = F.interpolate(initial_map, x[0, 0].shape, mode="bilinear", align_corners=False)
                x = conv(torch.cat((x, tmp), dim=1))
        return x


# ===================================================================
#                       UNet Interface
# ===================================================================

class UNet_Submodel(nn.Module):
    """UNet submodel used for interface networks."""
    def __init__(self, scales, kernel_sizes, input_res,
                 padding_mode="zeros", upsample_mode="nearest"):
        super().__init__()
        self.scales = scales
        self.max_scale = len(scales) - 1
        self.kernel_sizes = (
            [kernel_sizes] * len(scales) if isinstance(kernel_sizes, int) else kernel_sizes
        )

        in_fmaps = scales[0][0]
        down_blocks = [scales[d][0] for d in range(1, self.max_scale)]
        bottom_fmaps = scales[self.max_scale]
        up_blocks = [scales[d][1] for d in range(self.max_scale - 1, 0, -1)]
        out_fmaps = scales[0][1]

        if isinstance(input_res, list):
            self.input_res = tuple(input_res)
            list_res = [(input_res[0] // 2**i, input_res[1] // 2**i) for i in range(self.max_scale)]
        else:
            self.input_res = (input_res, input_res)
            list_res = [input_res // 2**i for i in range(self.max_scale)]

        self.ConvsDown = nn.ModuleList([
            ConvBlock(in_fmaps, "in", self.kernel_sizes[0], padding_mode=padding_mode)
        ])

        for idx, down_fmaps in enumerate(down_blocks):
            self.ConvsDown.append(ConvBlock(
                down_fmaps, "down", self.kernel_sizes[idx + 1], padding_mode=padding_mode
            ))

        self.ConvBottom = ConvBlock(
            bottom_fmaps, "bottom", self.kernel_sizes[-1],
            padding_mode=padding_mode, upsample_mode=upsample_mode,
            out_size=list_res.pop()
        )

        self.ConvsUp = nn.ModuleList()
        for idx, up_fmaps in enumerate(up_blocks):
            self.ConvsUp.append(ConvBlock(
                up_fmaps, "up", self.kernel_sizes[-2 - idx],
                padding_mode=padding_mode, upsample_mode=upsample_mode,
                out_size=list_res.pop()
            ))

        self.ConvsUp.append(ConvBlock(
            out_fmaps, "out", self.kernel_sizes[0], padding_mode=padding_mode
        ))

    def forward(self, x):
        inputs_down = []
        for conv in self.ConvsDown:
            x = conv(x)
            inputs_down.append(x)

        x = self.ConvBottom(x)

        for conv in self.ConvsUp:
            skip = inputs_down.pop()
            x = conv(torch.cat((x, skip), dim=1))

        return x


class UNetInterface(nn.Module):
    """Interface model combining two UNet submodels on masked domains."""
    def __init__(self, scales, kernel_sizes, input_res, inner_mask, outer_mask,
                 padding_mode="zeros", upsample_mode="nearest"):
        super().__init__()
        self.inner_mask = inner_mask
        self.outer_mask = outer_mask

        self.submodel1 = UNet_Submodel(scales, kernel_sizes, input_res, padding_mode, upsample_mode)
        self.submodel2 = UNet_Submodel(scales, kernel_sizes, input_res, padding_mode, upsample_mode)

    def forward(self, x):
        x1 = x * self.inner_mask
        x2 = x * self.outer_mask

        out1 = self.submodel1(x1)
        out2 = self.submodel2(x2)
        return out1, out2
