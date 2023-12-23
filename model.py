import torch
import lightning as L
from utils import *
from PIL import Image
from torchvision.transforms import ToPILImage
from torch import nn
from torch.functional import F

class DiffusionModel(L.LightningModule):
    def __init__(self, beta, unet_type='double', n_channels=1):
        super().__init__()

        if unet_type == 'double':

            self.denoiser = DoubleUNet(n_channels)

        # find the parameters of the noising process
        self.beta = beta
        self.alpha = 1 - beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def test_forward(self, data_loader, test_dir, n=10):

        imgs, _ = next(iter(data_loader))
        batch_size = imgs.shape[0]

        batch_t, batch_tp1, t = get_batch(imgs, self.alpha_bar)

        for i in range(n):

            cur_t = t[i]
            to_pil = ToPILImage()
            cur_img_t = to_pil(batch_t[i])
            cur_img_tp1 = to_pil(batch_tp1[i])
            cur_img_t.save(f'{test_dir}/img_{i}_timestep_{cur_t}.jpg')
            cur_img_tp1.save(f'{test_dir}/img_{i}_timestep_{cur_t+1}.jpg')


class DoubleUNet(L.LightningModule):
    '''
    Standard U-Net architecture

    Code adapted from https://github.com/milesial/Pytorch-UNet/
    '''
    def __init__(self, n_channels):
        super().__init__()

        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleDown(64, 128)
        self.down2 = DoubleDown(128, 256)
        self.down3 = DoubleDown(256, 512)
        self.down4 = DoubleDown(512, 1024 // 2)
        self.up1 = DoubleUp(1024, 512 // 2)
        self.up2 = DoubleUp(512, 256 // 2)
        self.up3 = DoubleUp(256, 128 // 2)
        self.up4 = DoubleUp(128, 64)
        self.up5 = DoubleUp(64, n_channels)

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x)

        return x


class DoubleConv(nn.Module):

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

        self.down_conv(x)


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

        x1 = F.pad(x1, [deltaX//2, deltaX//2, deltaY//2, deltaY//2])

        x = torch.cat([x1, x2], dim=1)

        return self.conv(x)
