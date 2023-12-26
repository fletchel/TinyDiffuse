import torch
import torchvision
import torchvision.transforms as transforms
import argparse
from utils import *
import torch.utils
from model import *
from lightning_utils import *
from pytorch_lightning.callbacks import ModelCheckpoint

def parse_args():

    parser = argparse.ArgumentParser(description="Train a diffusion model")

    parser.add_argument('--dataset',
                        help='training dataset',
                        default='MNIST',
                        type=str)
    parser.add_argument('--transforms',
                        help='transforms to apply to dataset (only available is normalize)',
                        nargs="+",
                        default=['normalize_greyscale'])
    parser.add_argument('--batch_size',
                        help='batch size',
                        default=128,
                        type=int)
    parser.add_argument('--val_prop',
                        help='proportion of training data to use for validation',
                        default=0.1,
                        type=float)
    parser.add_argument('--data_dir',
                        help='directory for data',
                        default='./data')
    parser.add_argument('--test_dir',
                        help='directory to save test images',
                        default='./test')
    parser.add_argument('--num_steps',
                        help='number of steps in the forward diffusion process',
                        default=1000,
                        type=int)
    parser.add_argument('--min_beta',
                        help='minimum beta',
                        default=1e-4,
                        type=float)
    parser.add_argument('--max_beta',
                        help='maximum beta',
                        default=0.02,
                        type=float)
    parser.add_argument('--unet_type',
                        help='which type of unet to use - single or double',
                        default='single',
                        type=str)
    parser.add_argument('--epochs',
                        help='number of epochs to train for',
                        default=20,
                        type=int)
    parser.add_argument('--generate_sample_every_n_epochs',
                        help='generate a grid of samples every n epochs, None to not',
                        default=None,
                        type=int)
    parser.add_argument('--checkpoint_every_n_epochs',
                        help='save a checkpoint ever n epochs, None to not',
                        default=None,
                        type=int)

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if args.dataset == "MNIST":

        transforms_ = transforms.Compose(get_transforms(args.transforms))

        full_train_set = torchvision.datasets.MNIST(root=args.data_dir,
                                                    train=True,
                                                    download=True,
                                                    transform=transforms_)
        test_set = torchvision.datasets.MNIST(root=args.data_dir,
                                                    train=False,
                                                    download=True,
                                                    transform=transforms_)

        val_size = int(args.val_prop * len(full_train_set))
        train_size = len(full_train_set) - val_size

        train_set, val_set = torch.utils.data.random_split(full_train_set, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # test the forward process
    beta = torch.cat((torch.Tensor([0]), torch.linspace(args.min_beta, args.max_beta, args.num_steps)))
    model = DiffusionModel(beta=beta, unet_type=args.unet_type, T=args.num_steps)

    # set parameters of the FID score
    pfw.set_config(batch_size=args.batch_size, device=model.device)

    callbacks = []

    if args.generate_sample_every_n_epochs:
        callbacks.append(GenerateSamplesMNIST(args.generate_sample_every_n_epochs))

    if args.checkpoint_every_n_epochs:
        callbacks.append(ModelCheckpoint(
            dirpath='./checkpoints',
            every_n_epochs=args.checkpoint_every_n_epochs,
            save_top_k=1))

    trainer = pl.Trainer(max_epochs=args.epochs, callbacks=callbacks, gpus=1)
    trainer.fit(model, train_loader, None)