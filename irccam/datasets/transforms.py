from torchvision import transforms


class Identity:
    def __call__(self, sample):
        return sample


def get_transforms(hparams):
    trans = transforms.Compose(
        [transforms.ToTensor(), transforms.RandomRotation(360) if hparams.random_rotations else Identity()]
    )
    return trans
