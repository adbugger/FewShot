import math
import random
import numbers

import torch
nn = torch.nn
F = nn.functional

from torchvision.transforms import (
    RandomResizedCrop, ToTensor, Compose, Normalize, RandomHorizontalFlip,
    ColorJitter, RandomApply, RandomGrayscale, ToPILImage
)

def CropResize(options):
    return Compose([
        RandomResizedCrop(size=options.image_size),
        RandomHorizontalFlip(0.5),
        ToTensor(),
        Normalize(options.image_mean, options.image_std),
    ])

def ColorDistort(options):
    s = options.jitter_strength
    # No need to normalize after color jitter?
    # Or normalize before color jitter?
    return Compose([
        ToTensor(),
        Normalize(options.image_mean, options.image_std),
        ToPILImage(),
        RandomApply([
            ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        ], p=0.8),
        RandomGrayscale(p=0.2),
        ToTensor(),
    ])

# https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
class GaussianSmoothing():
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        self.pad_size = math.floor((kernel_size-1) / 2)
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.weight = kernel
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def __call__(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(
            F.pad(
                input.unsqueeze(0),
                (self.pad_size, self.pad_size, self.pad_size, self.pad_size),
                mode='reflect'
            ),
            weight=self.weight, groups=self.groups).squeeze(0)

def GaussBlur(options):
    kernel_size = math.floor(options.image_size / 10)
    # ensure odd kernel size so that we can
    # reflect pad with (kernel_size - 1)/2 to get same size output
    if kernel_size % 2 == 0:
        kernel_size += 1
    return Compose([
        ToTensor(),
        Normalize(options.image_mean, options.image_std),
        RandomApply([
            GaussianSmoothing(
                channels=options.image_channels,
                kernel_size=kernel_size,
                sigma=random.uniform(0.1, 2.0),
                dim=2
            )
        ], p=0.5),
    ])
