import torch
from pytorch_lightning import Trainer
import json
from pytorch_lightning.loggers import TestTubeLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import datetime
from torch.utils.data import DataLoader

from cloudseg.training.cloud_segmentation import CloudSegmentation
from cloudseg.utils.constants import *
from cloudseg.training.transforms import get_transforms, PairToTensor, get_validation_transforms
from cloudseg.training.cloud_dataset import HDF5Dataset
from cloudseg.utils.args import parse_args


def train(args):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{timestamp}-{args.experiment_name}")
    trainer = Trainer(
        logger=configure_logger(args, log_dir),
        checkpoint_callback=configure_checkpoints(args, log_dir),
        gpus=args.gpus if torch.cuda.is_available() else None,
        max_epochs=args.num_epochs,
        distributed_backend="ddp" if args.cluster == True else None,
        fast_dev_run=args.fast_dev_run,
    )

    ## Data
    val_trans = get_validation_transforms(args)
    train_trans = get_transforms(args)
    dataset_train = HDF5Dataset(args.dataset_root, "train", train_trans, args.use_clear_sky)
    dataset_val = HDF5Dataset(args.dataset_root, "val", val_trans, args.use_clear_sky)
    dataset_test = HDF5Dataset(args.dataset_root, "test", val_trans, args.use_clear_sky)
    train_loader = DataLoader(
        dataset_train, args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        dataset_val, args.batch_size_val, shuffle=False, pin_memory=True, drop_last=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        dataset_test,
        args.batch_size_val,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    ## Model & Training
    model = CloudSegmentation(**vars(args))
    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, test_dataloaders=test_loader)


def configure_logger(args, log_dir):
    loggers = [
        TestTubeLogger(
            save_dir=log_dir,
            name="tube_logs",
            version=0,
        )
    ]
    if args.use_wandb:
        loggers.append(WandbLogger(project=args.wandb_project, entity=args.wandb_entity))
    return loggers


def configure_checkpoints(args, log_dir):
    return ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor="val_iou",
        mode="max",
        prefix="",
        filename="best-{epoch:02d}-{val_iou:.2f}",
        dirpath=log_dir,
    )


if __name__ == "__main__":
    args = parse_args()
    train(args)
