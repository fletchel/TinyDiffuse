import pytorch_lightning as pl
from torchvision.transforms import ToPILImage
class TestLogging(pl.Callback):

    def __init__(self):
        super().__init__()
    def on_train_epoch_end(self, trainer, pl_module):

        cur_epoch = trainer.current_epoch
        generated_img = pl_module.generate_samples().squeeze(0)
        to_pil = ToPILImage()
        cur_img = to_pil(generated_img)

        cur_img.save(f'test_samples/epoch_{cur_epoch}.jpg')



