import torch
from pytorch_lightning import Trainer
import argparse
import os
import json

from irccam.training.cloud_segmentation import CloudSegmentation
from irccam.utils.definitions import *


def train(config):
    model = CloudSegmentation(config)

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


def get_config(config_file):
    with open(config_file) as f:
        config = json.load(f)
    if not os.path.isdir(config["dataset_root"]):
        config["dataset_root"] = os.path.join(DATASET_PATH, config["dataset_root"])
    return argparse.Namespace(**config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        required=True,
        type=str,
        help="config file path (default: None)",
    )
    args = parser.parse_args()
    config = get_config(args.config)
    train(config)
