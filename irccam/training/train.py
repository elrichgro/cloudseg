import torch
from pytorch_lightning import Trainer
from argparse import Namespace
import os

from irccam.training.cloud_segmentation import CloudSegmentation
from irccam.utils.definitions import *


def train(args):
    model = CloudSegmentation(args)

    # TODO: callbacks, logging
    trainer = Trainer(
        # logger=logger,
        # checkpoint_callback = checkpoint_callback,
        gpus="-1"
        if torch.cuda.is_available()
        else None,
    )

    trainer.fit(model)

    trainer.test(model)


if __name__ == "__main__":
    args = {
        "batch_size": 4,
        "batch_size_val": 8,
        "num_epochs": 16,
        "model_name": "unet",
        "dataset_root": os.path.join(DATASET_PATH, "dataset_v1"),
        "learning_rate": 0.01,
    }
    train(Namespace(**args))
