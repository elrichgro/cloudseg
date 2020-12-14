from torchvision import transforms


class Identity:
    def __call__(self, sample):
        return sample


def get_transforms(hparams):
    trans = transforms.Compose(
        [transforms.RandomRotation(360, fill=0) if hparams.random_rotations else Identity, transforms.ToTensor()]
    )
    return trans
