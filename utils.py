import os
import random
import cv2
import torch
from tiatoolbox.tools.stainnorm import MacenkoNormalizer
from torch import nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms

target_img = cv2.imread('resources/target_image.png')  # target image
target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
macenko_norm = MacenkoNormalizer()
macenko_norm.fit(target_img)


class CustomDataset(torch.utils.data.Dataset):
    """Dataset for Tumour Segmentation"""

    def __init__(self, images, masks, transform=None, mask_transform=None, seed_fn=None):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.mask_transform = mask_transform
        self.seed_fn = seed_fn

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_file_path = self.images[idx]
        image_name = os.path.basename(image_file_path)

        image = read_image(image_file_path)
        image = (image / 255.0).to(torch.float32)

        if self.masks is not None:
            mask_file_path = self.masks[idx]
            mask_name = os.path.basename(mask_file_path)

            mask = read_image(mask_file_path, mode=ImageReadMode.GRAY)
            mask = (mask > 0.5).to(torch.float32)

            assert (len(self.images) == len(self.masks))
            assert (image_name == mask_name)

            if self.transform:
                seed = random.randint(0, 2 ** 32)
                self._set_seed(seed)
                image = self.transform(image)

                if self.mask_transform:
                    self._set_seed(seed)
                    mask = self.mask_transform(mask)

            sample = {'image': image, 'mask': mask, 'image_file_path': image_file_path, 'image_name': image_name,
                      'mask_file_path': mask_file_path, 'mask_name': mask_name}

            return sample

        else:
            if self.transform:
                seed = random.randint(0, 2 ** 32)
                self._set_seed(seed)
                image = self.transform(image)

            sample = {'image': image, 'image_file_path': image_file_path, 'image_name': image_name}

            return sample

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        if self.seed_fn:
            self.seed_fn(seed)


def macenko_normalize(image):
    return macenko_norm.transform(image)


class Normalizer:
    def __init__(self, name):
        self.name = name.lower()
        if self.name == 'macenko':
            self.norm = macenko_normalize

    def get_name(self):
        return f'{self.name}'

    def get_norm(self):
        return self.norm


def make_tfs(augs):
    return transforms.Compose(augs)


def create_model():
    model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, 1, 1)
    model.aux_classifier[4] = nn.Conv2d(10, 1, 1)

    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True

    # Freeze auxillary classifier
    for param in model.aux_classifier.parameters():
        param.requires_grad = False

    return model