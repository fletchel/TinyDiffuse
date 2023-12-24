import torch
import lightning as L
from utils import *
from PIL import Image
from torchvision.transforms import ToPILImage
from torch import nn, optim
from torch.functional import F
from model_parts import *

class DiffusionModel(L.LightningModule):
    def __init__(self, beta, unet_type='double', n_channels=1, lr=1e-3):
        super().__init__()

        if unet_type == 'double':

            self.denoiser = DoubleUNet28(n_channels)

        if unet_type == 'single':

            self.denoiser = SingleUNet28(n_channels)

        self.lr = lr

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

    def forward(self, x):

        return self.denoiser(x)

    def training_step(self, batch):

        x, y = batch
        noise = torch.randn(x)

        pred_noise =

    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class SingleUNet28(L.LightningModule):
    '''
    U-Net architecture, with one fewer up/down sampling (should work for 28x28 images)

    Code adapted from https://github.com/milesial/Pytorch-UNet/
    '''
    def __init__(self, n_channels):
        super().__init__()

        self.n_channels = n_channels

        self.inc = SingleConv(n_channels, 64)
        self.down1 = SingleDown(64, 128)
        self.down2 = SingleDown(128, 256)
        self.down3 = SingleDown(256, 512 // 2)

        self.up1 = SingleUp(512, 256 // 2)
        self.up2 = SingleUp(256, 128 // 2)
        self.up3 = SingleUp(128, 64)
        self.outconv = OutConv(64, n_channels)

    def forward(self, x):

        x1 = self.inc(x)  # 28x28
        x2 = self.down1(x1)  # output = 14x14
        x3 = self.down2(x2)  # 7x7
        x4 = self.down3(x3)  # 3x3

        x = self.up1(x4, x3)  # 7x7
        x = self.up2(x, x2)  # 14x14
        x = self.up3(x, x1)  # 28x28
        x = self.outconv(x)  # 28x28

        return x

class DoubleUNet28(L.LightningModule):
    '''
    U-Net architecture, with one fewer up/down sampling (should work for 28x28 images)

    Code adapted from https://github.com/milesial/Pytorch-UNet/
    '''
    def __init__(self, n_channels):
        super().__init__()

        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleDown(64, 128)
        self.down2 = DoubleDown(128, 256)
        self.down3 = DoubleDown(256, 512 // 2)

        self.up1 = DoubleUp(512, 256 // 2)
        self.up2 = DoubleUp(256, 128 // 2)
        self.up3 = DoubleUp(128, 64)
        self.outconv = OutConv(64, n_channels)

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outconv(x)

        return x



class DoubleUNet32(L.LightningModule):
    '''
    Standard U-Net architecture (should work for 32x32 images)

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
