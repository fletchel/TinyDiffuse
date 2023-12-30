import pytorch_lightning as pl
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

class TestLogging(pl.Callback):

    def __init__(self):
        super().__init__()
    def on_train_epoch_end(self, trainer, pl_module):

        cur_epoch = trainer.current_epoch
        generated_img = pl_module.generate_samples().squeeze(0)
        to_pil = ToPILImage()
        cur_img = to_pil(generated_img)

        cur_img.save(f'test_samples/epoch_{cur_epoch}.jpg')


class GenerateSamplesMNIST(pl.Callback):

    def __init__(self, sample_every_n, grid_size=4):
        self.sample_every_n = sample_every_n
        self.grid_size = grid_size
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):

        cur_epoch = trainer.current_epoch

        if cur_epoch % self.sample_every_n == 0:

            generated_imgs = pl_module.generate_samples((self.grid_size**2, 1, 28, 28))
            grid = make_grid(generated_imgs, nrow=self.grid_size)
            to_pil = ToPILImage()
            pil_grid = to_pil(grid)

            pil_grid.save(f'test_samples/sample_grid_epoch_{cur_epoch}.jpg')
            pl_module.logger.experiment.add_image('sample_images', grid, cur_epoch)



