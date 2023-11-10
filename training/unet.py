import torch
import torch.nn as nn

class _ConvBlock(nn.Module):
    def __init__(self, scales, kernel_size, pool= False, upsample_size=None, last_one = False):
        super(_ConvBlock, self).__init__()
        layers = list()

        #Aplly pooling if needed
        if pool == True:
            layers.append(nn.MaxPool2d(2))
        
        for i in range(len(scales)-1):
            layers.append(nn.Conv2d(scales[i], scales[i+1], kernel_size=kernel_size,\
                                     stride=1, padding=1))
            
            if last_one == False:
                layers.append(nn.ReLU())

            else:
                if i != len(scales)-2:
                    layers.append(nn.ReLU())
                    
        if upsample_size is not None:
            layers.append(nn.Upsample(size=upsample_size, mode='nearest'))

        self.encode = nn.Sequential(*layers)

        

    def forward(self, x):
        return self.encode(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.ConvsDown = nn.ModuleList([
            _ConvBlock((1, 30, 20), 3),
            _ConvBlock((20, 20 ,20), 3, True),
            _ConvBlock((20, 16, 16, 20), 3, True),
            _ConvBlock((20, 20, 20), 3, True),
        ])

        self.ConvBottom = _ConvBlock((20, 60, 20), 3, True, upsample_size=12)

        self.ConvsUp = nn.ModuleList([
            _ConvBlock((40, 20, 20), 3, upsample_size=25),
            _ConvBlock((40, 16, 16, 20), 3, upsample_size=50),
            _ConvBlock((40, 20, 20), 3, upsample_size=101),
            _ConvBlock((40, 30, 1), 3, last_one = True),
        ])

    def forward(self, x):
        skips = []
        for conv in self.ConvsDown:
            x = conv(x)
            skips.append(x.clone())
        x = self.ConvBottom(x)
        skips = skips[::-1]

        for conv, skip in zip(self.ConvsUp, skips):
            x = torch.cat([x, skip], dim=1)
            x = conv(x)

        return x

# Create an instance of the UNet model
model = UNet()

# Print the model architecture
model_str = str(model)

# Specify the path to the .txt file
file_path = 'model_architecture.log'

# Write the model architecture to the .txt file
with open(file_path, 'w') as file:
    file.write(model_str)
