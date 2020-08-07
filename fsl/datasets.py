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


class Imagenet1k():
    def __init__(self, options):
        self.dataset_root = abspath(getattr_or_default(options, 'dataset_root', '/ssd_scratch/cvit/aditya.bharti/Imagenet-orig'))

        # Repeat the other steps as for FewShotDataset. Maybe find a way to absorb this common code into something?
        self.image_size = 256
        self.image_channels = 3

        self.mean = (0.5071, 0.4867, 0.4408)
        self.std = (0.2675, 0.2565, 0.2761)

        options.image_size = self.image_size
        options.image_channels = self.image_channels
        
        options.image_mean = self.mean
        options.image_std = self.std

        multi_transforms = get_transforms(options)
        other_transform = tv_transforms.Compose([
            tv_transforms.Resize(size=(self.image_size, self.image_size)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(self.mean, self.std),
        ])

        self.train_set = MultiTransformDataset(
            tv_datasets.ImageNet(root=self.dataset_root, split="train", transform=tv_transforms.Resize(size=(self.image_size, self.image_size))),
            transform_list=multi_transforms, options=options
        )
        self.trainval_set = self.train_set

        self.plain_train_set = tv_datasets.ImageNet(root=self.dataset_root, split="train", transform=other_transform)
        self.plain_trainval_set = self.plain_train_set

        self.test_set = tv_datasets.ImageNet(root=self.dataset_root, split="val", transform=other_transform)
        self.valid_set = tv_datasets.ImageNet(root=self.dataset_root, split="val", transform=other_transform)

class FewShotDataset():
    def __init__(self, options):
        # needs self. (dataset_root, image_size, image_channels, mean, std)
        self.train_root = path_join(self.dataset_root, 'train')
        self.test_root = path_join(self.dataset_root, 'test')
        self.valid_root = path_join(self.dataset_root, 'val')
        self.trainval_root = path_join(self.dataset_root, 'trainval')

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
        self.trainval_set = MultiTransformDataset(
            tv_datasets.ImageFolder(root=self.trainval_root, transform=None),
            transform_list=multi_transforms, options=options
        )
        self.plain_train_set = tv_datasets.ImageFolder(root=self.train_root, transform=other_transform)
        self.plain_trainval_set = tv_datasets.ImageFolder(root=self.trainval_root, transform=other_transform)
        self.test_set = tv_datasets.ImageFolder(root=self.test_root, transform=other_transform)
        self.valid_set = tv_datasets.ImageFolder(root=self.valid_root, transform=other_transform)


class cifar100fs(FewShotDataset):
    def __init__(self, options):
        self.dataset_root = abspath(getattr_or_default(options, 'dataset_root', '/home/aditya.bharti/cifar100'))

        self.mean = (0.5071, 0.4867, 0.4408)
        self.std = (0.2675, 0.2565, 0.2761)

        self.image_channels = 3
        self.image_size = 32
        super(cifar100fs, self).__init__(options)


class fc100(FewShotDataset):
    def __init__(self, options):
        self.dataset_root = abspath(getattr_or_default(options, 'dataset_root', '/home/aditya.bharti/FC100'))

        self.mean = (0.5071, 0.4867, 0.4408)
        self.std = (0.2675, 0.2565, 0.2761)

        self.image_channels = 3
        self.image_size = 84
        super(fc100, self).__init__(options)


class miniImagenet(FewShotDataset):
    def __init__(self, options):
        self.dataset_root = abspath(getattr_or_default(options, 'dataset_root', '/home/aditya.bharti/mini-imagenet'))

        self.mean = (0.4802, 0.4481, 0.3975)
        self.std = (0.2302, 0.2265, 0.2262)

        self.image_channels = 3
        self.image_size = 84
        super(miniImagenet, self).__init__(options)      


if __name__ == "__main__":
    from arguments import parse_args
    opts = parse_args()

    for dataset in [fc100, miniImagenet, cifar100fs, Imagenet1k]:
        d = dataset(opts)
        print(str(d), "loaded")
        if(hasattr(d, "trainval_set")):
            print(str(d), "has trainval")
        else:
            print(str(d), "couldn't find trainval")