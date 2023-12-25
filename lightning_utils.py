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

    def __init__(self, sample_every_n):
        self.sample_every_n = sample_every_n
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):

        cur_epoch = trainer.current_epoch
        grid_size = pl_module.grid_size

        if cur_epoch % self.sample_every_n == 0:

            generated_imgs = pl_module.generate_samples((grid_size, 1, 28, 28))
            grid = make_grid(generated_imgs, nrow=grid_size)
            to_pil = ToPILImage()
            grid = to_pil(grid)

            grid.save(f'test_samples/sample_grid_epoch_{cur_epoch}.jpg')
            pl_module.logger.experiment.add_image('sample_images', grid, cur_epoch)



