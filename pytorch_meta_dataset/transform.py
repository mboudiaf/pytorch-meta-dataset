import torchvision.transforms as transforms
from .utils import Split
from .config import DataConfig

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def get_transforms(data_config: DataConfig,
                   split: Split):
    if split == Split["TRAIN"]:
        return train_transform(data_config)
    else:
        return test_transform(data_config)


def test_transform(data_config: DataConfig):
    resize_size = int(data_config.image_size*256./224.)
    transf_dict = {'resize': transforms.Resize(resize_size),
                   'center_crop': transforms.CenterCrop(data_config.image_size),
                   'normalize': normalize}
    augmentations = data_config.test_transforms
    return transforms.Compose([transf_dict[key] for key in augmentations])


def train_transform(data_config: DataConfig):

    transf_dict = {'resize': transforms.Resize(data_config.image_size),
                   'center_crop': transforms.CenterCrop(data_config.image_size),
                   'random_resized_crop': transforms.RandomResizedCrop(data_config.image_size),
                   'jitter': transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                   'random_flip': transforms.RandomHorizontalFlip(),
                   'normalize': normalize}
    augmentations = data_config.train_transforms
    return transforms.Compose([transf_dict[key] for key in augmentations])
