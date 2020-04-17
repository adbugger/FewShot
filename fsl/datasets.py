from os.path import abspath, join as path_join
from torchvision import datasets as tv_datasets, transforms as tv_transforms
from torch.utils.data import Dataset

from utils import getattr_or_default
import transforms

def get_transforms(options):
    return [
        getattr(transforms, options.first_augment)(options),
        getattr(transforms, options.second_augment)(options),
    ]

class MultiTransformDataset(Dataset):
    def __init__(self, dataset, transform_list, options):
        self.dataset = dataset
        self.transform_list = transform_list

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return tuple(tr(image) for tr in self.transform_list) + (label,)

    # TODO: cleanly expose the underlying dataset
    @property
    def classes(self):
        return self.dataset.classes

class cifar100fs():
    def __init__(self, options):
        self.dataset_root = abspath(getattr_or_default(options, 'dataset_root', '/home/aditya.bharti/cifar100'))
        self.train_root = path_join(self.dataset_root, 'train')
        self.test_root = path_join(self.dataset_root, 'test')
        self.valid_root = path_join(self.dataset_root, 'val')

        self.mean = (0.5071, 0.4867, 0.4408)
        self.std = (0.2675, 0.2565, 0.2761)

        self.image_channels = 3
        self.image_size = 32

        options.image_size = self.image_size
        options.image_channels = self.image_channels

        options.image_mean = self.mean
        options.image_std = self.std

        multi_transforms = get_transforms(options)
        other_transform = tv_transforms.Compose([
                            tv_transforms.ToTensor(),
                            tv_transforms.Normalize(self.mean, self.std),
                        ])

        self.train_set = MultiTransformDataset(
            tv_datasets.ImageFolder(root=self.train_root, transform=None),
            transform_list=multi_transforms, options=options
        )
        self.plain_train_set = tv_datasets.ImageFolder(root=self.train_root, transform=other_transform)
        self.test_set = tv_datasets.ImageFolder(root=self.test_root, transform=other_transform)
        self.valid_set = tv_datasets.ImageFolder(root=self.valid_root, transform=other_transform)


class fc100():
    def __init__(self, options):
        self.dataset_root = abspath(getattr_or_default(options, 'dataset_root', '/home/aditya.bharti/FC100'))
        self.train_root = path_join(self.dataset_root, 'train')
        self.test_root = path_join(self.dataset_root, 'test')
        self.valid_root = path_join(self.dataset_root, 'val')

        self.mean = (0.5071, 0.4867, 0.4408)
        self.std = (0.2675, 0.2565, 0.2761)

        self.image_channels = 3
        self.image_size = 84

        options.image_size = self.image_size
        options.image_channels = self.image_channels

        options.image_mean = self.mean
        options.image_std = self.std

        multi_transforms = get_transforms(options)
        other_transform = tv_transforms.Compose([
                            tv_transforms.ToTensor(),
                            tv_transforms.Normalize(self.mean, self.std),
                        ])

        self.train_set = MultiTransformDataset(
            tv_datasets.ImageFolder(root=self.train_root, transform=None),
            transform_list=multi_transforms, options=options
        )
        self.plain_train_set = tv_datasets.ImageFolder(root=self.train_root, transform=other_transform)
        self.test_set = tv_datasets.ImageFolder(root=self.test_root, transform=other_transform)
        self.valid_set = tv_datasets.ImageFolder(root=self.valid_root, transform=other_transform)


class miniImagenet():
    def __init__(self, options):
        self.dataset_root = abspath(getattr_or_default(options, 'dataset_root', '/home/aditya.bharti/mini-imagenet'))
        self.train_root = path_join(self.dataset_root, 'train')
        self.test_root = path_join(self.dataset_root, 'test')
        self.valid_root = path_join(self.dataset_root, 'val')

        self.mean = (0.4802, 0.4481, 0.3975)
        self.std = (0.2302, 0.2265, 0.2262)

        self.image_channels = 3
        self.image_size = 84

        options.image_size = self.image_size
        options.image_channels = self.image_channels

        options.image_mean = self.mean
        options.image_std = self.std

        multi_transforms = get_transforms(options)
        other_transform = tv_transforms.Compose([
                            tv_transforms.ToTensor(),
                            tv_transforms.Normalize(self.mean, self.std),
                        ])

        self.train_set = MultiTransformDataset(
            tv_datasets.ImageFolder(root=self.train_root, transform=None),
            transform_list=multi_transforms, options=options
        )
        self.plain_train_set = tv_datasets.ImageFolder(root=self.train_root, transform=other_transform)
        self.test_set = tv_datasets.ImageFolder(root=self.test_root, transform=other_transform)
        self.valid_set = tv_datasets.ImageFolder(root=self.valid_root, transform=other_transform)
