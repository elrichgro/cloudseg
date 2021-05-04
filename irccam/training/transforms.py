from torchvision import transforms
from irccam.utils.constants import PROJECT_PATH
import random
import os
import cv2
import numpy as np
from torchvision.transforms import functional as F


class Identity:
    def __call__(self, sample):
        return sample


class RandomMask:
    def __init__(self, training=True):
        mode = "training" if training else "validation"
        masks_path = f"irccam/datasets/resources/{mode}_masks"
        num_masks = 30 if training else 3
        self.mask_files = [os.path.join(PROJECT_PATH, masks_path, f"{i}.bmp") for i in range(1, num_masks + 1)]

    def __call__(self, input):
        img, label = input
        mask_file = random.choice(self.mask_files)
        mask = cv2.imread(mask_file, -1)
        assert mask is not None, f"Could not find mask {mask_file}"
        img[mask == 255] = 0
        label[mask == 255] = -1
        return img, label


class PairToTensor:
    def __call__(self, input):
        img, label = input
        return transforms.ToTensor()(img), transforms.ToTensor()(label)


class PairRotate:
    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None, fill_label=-1):
        if degrees < 0:
            raise ValueError("If degrees is a single number, it must be positive.")
        self.degrees = (-degrees, degrees)

        self.resample = resample  # default is nearest, good for labels
        self.expand = expand
        self.center = center
        self.fill = fill
        self.fill_label = fill_label

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, input):
        img, label = input
        angle = self.get_params(self.degrees)

        return (
            F.rotate(img, angle, self.resample, self.expand, self.center, self.fill),
            F.rotate(label, angle, self.resample, self.expand, self.center, self.fill_label),
        )


def get_transforms(args):
    trans = transforms.Compose(
        [
            RandomMask() if args.random_mask else Identity(),
            PairToTensor(),
            PairRotate(360) if args.random_rotations else Identity(),
        ]
    )
    return trans


def get_validation_transforms(args):
    trans = transforms.Compose(
        [
            RandomMask(training=False) if args.val_random_mask else Identity(),
            PairToTensor(),
        ]
    )
    return trans
