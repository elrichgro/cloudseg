import torch
from pytorch_lightning import Trainer
import argparse
import json
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import datetime

from irccam.training.cloud_segmentation import CloudSegmentation
from irccam.utils.definitions import *


def train(config):
    model = CloudSegmentation(config)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logger = TestTubeLogger(
        save_dir=os.path.join(config.log_dir, "{}-{}".format(timestamp, config.experiment_name)),
        name="tube_logs",
        version=0,
    )

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor="val_iou",
        mode="max",
        prefix="",
        filename="best-{epoch:02d}-{val_iou:.2f}",
    )

    trainer = Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        gpus="-1" if torch.cuda.is_available() else None,
        max_nb_epochs=config.num_epochs,
    )

    trainer.fit(model)

    trainer.test(model)


def get_config(config_file):
    with open(config_file) as f:
        config = json.load(f)
    if not os.path.isdir(config["dataset_root"]):
        config["dataset_root"] = os.path.join(DATASET_PATH, config["dataset_root"])
    config["log_dir"] = config.get("log_dir", os.path.join(PROJECT_PATH, "training_logs"))
    return argparse.Namespace(**config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default=None, required=True, type=str, help="config file path (default: None)",
    )
    args = parser.parse_args()
    config = get_config(args.config)
    train(config)
