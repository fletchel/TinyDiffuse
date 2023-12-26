from typing import Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from utils import *
from PIL import Image
from torchvision.transforms import ToPILImage
from torch import nn, optim
from torch.functional import F
from model_parts import *
import sys

import pytorch_fid_wrapper as pfw


class DiffusionModel(pl.LightningModule):
    def __init__(self, beta, unet_type='double', n_channels=1, lr=1e-3, T=1000):
        super().__init__()

        if unet_type == 'double':

            self.denoiser = DoubleUNet28(n_channels)

        if unet_type == 'single':

            self.denoiser = SingleUNet28(n_channels)

        self.lr = lr
        self.T = T

        # find the parameters of the noising process
        self.beta = beta.to(self.device)
        self.alpha = 1 - beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # set up validation step output list

        self.validation_step_outputs = []

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

    def forward(self, x, t):

        return self.denoiser(x, t)

    def training_step(self, batch):

        x, y = batch

        batch_t = torch.randint(0, self.T, (x.shape[0],)).to(self.device)
        cur_alpha_bar = self.alpha_bar[batch_t].reshape(-1, 1, 1, 1)

        noise = torch.randn(x.shape).to(self.device)
        pred_noise = self.denoiser(x*(cur_alpha_bar**0.5) + noise*(1 - cur_alpha_bar)**0.5, batch_t)

        loss = F.mse_loss(noise, pred_noise)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch):

        '''
        Use FID as validation metric
        '''

        if not self.trainer.sanity_checking:
            if self.current_epoch == 0 or self.current_epoch % 5 != 0:
                return None

        if self.trainer.sanity_checking:
            pfw.set_config(dims=192)

        else:
            pfw.set_config(dims=2048)

        real_imgs = batch[0]
        fake_imgs = self.generate_samples(real_imgs.shape)

        self.validation_step_outputs.append((real_imgs, fake_imgs))
        return real_imgs, fake_imgs

    def on_validation_epoch_end(self):

        real_imgs, fake_imgs = zip(*self.validation_step_outputs)

        # FID expects 3 channel input, so repeat single channel three times for MNIST
        real_imgs = torch.cat(real_imgs, dim=0).flatten(0, 1).repeat_interleave(3, dim=1)
        fake_imgs = torch.cat(fake_imgs, dim=0).flatten(0, 1).repeat_interleave(3, dim=1)

        val_fid = pfw.fid(fake_imgs, real_imgs)
        print(f'Epoch {self.current_epoch} val FID: {round(val_fid, 3)}')
        self.log('val_loss', val_fid)

        self.validation_step_outputs = []

    def generate_samples(self, sample_shape=(1, 1, 28, 28)):
        '''
        Generate samples from the trained model
        sample_shape: B, C, H, W
        '''

        batch_size = sample_shape[0]

        x = torch.randn(sample_shape).to(self.device)

        for t in range(self.T, 1, -1):

            cur_alpha_bar = self.alpha_bar[t].reshape(1, 1, 1, 1)
            cur_alpha = self.alpha[t].reshape(1, 1, 1, 1)
            sigma = (self.beta[t]**0.5).reshape(1, 1, 1, 1)

            z = torch.randn(x.shape) if t > 1 else torch.zeros(x.shape)
            pred_noise = self.denoiser(x, torch.tensor(t).expand(batch_size))

            x = (1/cur_alpha)**0.5*(x - pred_noise*(1-cur_alpha)/((1-cur_alpha_bar)**0.5)) + sigma*z

        x = x.clamp(0, 1)
        return x



    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class SingleUNet28(pl.LightningModule):
    '''
    U-Net architecture, with one fewer up/down sampling (should work for 28x28 images)

    Code adapted from https://github.com/milesial/Pytorch-UNet/
    '''
    def __init__(self, n_channels):
        super().__init__()

        self.n_channels = n_channels

        self.p_inc = get_positional_encodings(64)
        self.p1 = get_positional_encodings(128)
        self.p2 = get_positional_encodings(256)

        self.inc = SingleConv(n_channels, 64)
        self.down1 = SingleDown(64, 128)
        self.down2 = SingleDown(128, 256)
        self.down3 = SingleDown(256, 512 // 2)

        self.up1 = SingleUp(512, 256 // 2)
        self.up2 = SingleUp(256, 128 // 2)
        self.up3 = SingleUp(128, 64)

        self.outconv = OutConv(64, n_channels)

    def forward(self, x, t):

        batch_size = x.shape[0]
        p1 = self.p_inc[t, :].reshape(batch_size, -1, 1, 1)
        p2 = self.p1[t, :].reshape(batch_size, -1, 1, 1)
        p3 = self.p2[t, :].reshape(batch_size, -1, 1, 1)

        x1 = self.inc(x) + p1  # output = 28x28
        x2 = self.down1(x1) + p2  # 14x14
        x3 = self.down2(x2) + p3  # 7x7
        x4 = self.down3(x3) + p3  # 3x3

        x = self.up1(x4, x3) + p2  # 7x7

        x = self.up2(x, x2) + p1  # 14x14
        x = self.up3(x, x1) + p1  # 28x28
        x = self.outconv(x)  # 28x28
        if torch.isnan(x).any():
            print(f"x is nan at {self.global_step} and timestep {t}")
            sys.exit()


        return x


class DoubleUNet28(pl.LightningModule):
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



class DoubleUNet32(pl.LightningModule):
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
