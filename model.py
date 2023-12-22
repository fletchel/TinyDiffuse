import torch
import lightning as L
from utils import *
from PIL import Image
from torchvision.transforms import ToPILImage
class DiffusionModel(L.LightningModule):

    def __init__(self, denoiser, beta):
        super().__init__()
        self.denoiser = denoiser

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

