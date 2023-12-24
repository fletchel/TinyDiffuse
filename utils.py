import torch
import torchvision.transforms as transforms
def get_batch(imgs, alpha_bar):

    '''
    Function to get forward diffused images given the original images
    at randomly chosen timesteps t, t+1 (different timestep for each image)

    imgs: tensor of images at t=0 (B, C, H, W)
    alpha_bar: parameters of the forward process
    T: maximum number of timesteps in forward process
    '''

    batch_size = imgs.shape[0]
    T = alpha_bar.shape[0]
    flat_imgs = imgs.reshape(batch_size, -1)

    forward_t = torch.randint(0, T-1, (batch_size,))
    forward_tp1 = forward_t + 1

    cur_alpha_bar_t = alpha_bar[forward_t].unsqueeze(1)
    cur_alpha_bar_tp1 = alpha_bar[forward_tp1].unsqueeze(1)

    mu_t = (cur_alpha_bar_t)**0.5 * flat_imgs
    std_t = (1 - cur_alpha_bar_t)**0.5

    mu_tp1 = (cur_alpha_bar_tp1) ** 0.5 * flat_imgs
    std_tp1 = (1 - cur_alpha_bar_tp1)**0.5

    flat_imgs_t = torch.normal(mu_t, std_t)
    flat_imgs_tp1 = torch.normal(mu_tp1, std_tp1)

    imgs_t = flat_imgs_t.reshape(imgs.shape)
    imgs_tp1 = flat_imgs_tp1.reshape(imgs.shape)

    return imgs_t, imgs_tp1, forward_t

def get_transforms(transform_args):

    '''
    returns a list of transforms given the transform arguments
    always starts with transforms.ToTensor()
    '''

    transform_list = [transforms.ToTensor()]

    if "normalize_greyscale" in transform_args:

        transform_list.append(transforms.Normalize((0.5,), (0.5,)))

    return transform_list


def get_positional_encodings(d, T=1000):

    '''
    returns [T, d] positional encodings for each timestep for feature vector of dim d
    '''

    pos_encodings = torch.zeros((T,d), requires_grad=False)

    i = torch.arange(d//2)
    p = torch.arange(T).unsqueeze(1)

    pos_encodings[:, 0::2] = torch.sin(p / (10000 ** (2 * i / d)))
    pos_encodings[:, 1::2] = torch.cos(p / (10000 ** (2 * i / d)))

    return pos_encodings
