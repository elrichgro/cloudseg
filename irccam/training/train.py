import torch
from pytorch_lightning import Trainer
import argparse
import json
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import datetime
from torchvision import transforms
from torch.utils.data import DataLoader

from irccam.training.cloud_segmentation import CloudSegmentation
from irccam.utils.constants import *
from irccam.training.transforms import get_transforms
from irccam.training.cloud_dataset import HDF5Dataset


def train(config):
    hparams = argparse.Namespace(**config)

    trainer = Trainer(
        logger=configure_logger(hparams),
        checkpoint_callback=configure_checkpoints(hparams),
        gpus=hparams.gpus if torch.cuda.is_available() else None,
        max_epochs=hparams.num_epochs,
        distributed_backend="ddp" if hparams.cluster == True else None,
    )

    ## Data
    base_trans = transforms.Compose([transforms.ToTensor(),])
    train_trans = get_transforms(hparams)
    dataset_train = HDF5Dataset(hparams.dataset_root, "train", train_trans, hparams.use_clear_sky)
    dataset_val = HDF5Dataset(hparams.dataset_root, "val", base_trans, hparams.use_clear_sky)
    dataset_test = HDF5Dataset(hparams.dataset_root, "test", base_trans, hparams.use_clear_sky)
    train_loader = DataLoader(
        dataset_train, hparams.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4
    )
    val_loader = DataLoader(
        dataset_val, hparams.batch_size_val, shuffle=False, pin_memory=True, drop_last=False, num_workers=4
    )
    test_loader = DataLoader(
        dataset_test, hparams.batch_size_val, shuffle=False, pin_memory=True, drop_last=False, num_workers=4
    )

    ## Model & Training
    model = CloudSegmentation(**config)
    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, test_dataloaders=test_loader)


def configure_logger(hparams):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return TestTubeLogger(
        save_dir=os.path.join(hparams.log_dir, "{}-{}".format(timestamp, hparams.experiment_name)),
        name="tube_logs",
        version=0,
    )


def configure_checkpoints(hparams):
    return ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor="val_iou",
        mode="max",
        prefix="",
        filename="best-{epoch:02d}-{val_iou:.2f}",
    )


def get_config(config_file):
    with open(config_file) as f:
        config = json.load(f)
    if not os.path.isdir(config["dataset_root"]):
        config["dataset_root"] = os.path.join(DATASET_PATH, config["dataset_root"])
    config["log_dir"] = config.get("log_dir", os.path.join(PROJECT_PATH, "training_logs"))
    config["use_clear_sky"] = config.get("use_clear_sky", False)
    config["random_rotations"] = config.get("random_rotations", False)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default=None, required=True, type=str, help="config file path (default: None)",
    )
    args = parser.parse_args()
    config = get_config(args.config)
    train(config)
