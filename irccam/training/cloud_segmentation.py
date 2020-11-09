import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import os

from irccam.datasets.cloud_dataset import CloudDataset
from irccam.models.helpers import get_model
from irccam.models.unet import UNet


class CloudSegmentation(pl.LightningModule):
    def __init__(self, args):
        super(CloudSegmentation, self).__init__()
        self.args = args
        self.dataset_train = CloudDataset(args.dataset_root, "train")
        self.dataset_val = CloudDataset(args.dataset_root, "val")
        self.dataset_test = CloudDataset(args.dataset_root, "test")

        self.model = get_model(args.model_name, args)

        # TODO: add ignore_index arg for masked out pixels
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        # TODO: metrics

    def training_step(self, batch, batch_idx):
        batch_input = batch["irc"]
        batch_labels = batch["label"].squeeze(1)

        pred_labels = self.model(batch_input)

        loss = self.cross_entropy_loss(pred_labels, batch_labels)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_input = batch["irc"]
        batch_labels = batch["label"].squeeze(1)

        pred_labels = self.model(batch_input)

        loss = self.cross_entropy_loss(pred_labels, batch_labels)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        batch_input = batch["irc"]
        batch_labels = batch["label"].squeeze(1)

        pred_labels = self.model(batch_input)

        loss = self.cross_entropy_loss(pred_labels, batch_labels)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            self.args.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            self.args.batch_size_val,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            self.args.batch_size_val,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
