import torch
import lightning as L
import torch.nn as nn
from torch.functional import F

class SingleConv(L.LightningModule):

    '''
        One convolutional layers
        Batch norm and relus in between
    '''

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SingleUp(L.LightningModule):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = SingleConv(in_channels, out_channels)

    def forward(self, x1, x2):

        '''
        Both x1, x2 are BCHW
        x1: current layer to upsample
        x2: corresponding layer to pad and concatenate
        '''

        x1 = self.up(x1)

        deltaY = x2.shape[2] - x1.shape[2]
        deltaX = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [deltaX//2, deltaX - deltaX//2, deltaY//2, deltaY - deltaY//2])
        x = torch.cat([x1, x2], dim=1)

        return self.conv(x)


class SingleDown(L.LightningModule):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            SingleConv(in_channels, out_channels)
        )

    def forward(self, x):

        return self.down_conv(x)


'''
Below are the parts of a standard U-net - i.e. using double convolution layers etc.
'''

class DoubleConv(L.LightningModule):

    '''
        Two convolutional layers
        Batch norms and relus in between
    '''

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleDown(L.LightningModule):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )


    def forward(self, x):

        return self.down_conv(x)



class DoubleUp(L.LightningModule):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels//2)

    def forward(self, x1, x2):

        '''
        Both x1, x2 are BCHW
        x1: current layer to upsample
        x2: corresponding layer to pad and concatenate
        '''

        x1 = self.up(x1)

        deltaY = x2.shape[2] - x1.shape[2]
        deltaX = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [deltaX//2, deltaX - deltaX//2, deltaY//2, deltaY - deltaY//2])

        x = torch.cat([x1, x2], dim=1)

        return self.conv(x)



class OutConv(L.LightningModule):
    '''
    Final conv layer, outputting image
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):

        '''
        Both x1, x2 are BCHW
        x1: current layer to upsample
        x2: corresponding layer to pad and concatenate
        '''

        return self.conv(x)
