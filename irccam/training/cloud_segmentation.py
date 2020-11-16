import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from pytorch_lightning.metrics.functional.classification import iou

from irccam.datasets.helpers import get_dataset_class
from irccam.models.helpers import get_model


class CloudSegmentation(pl.LightningModule):
    def __init__(self, args):
        super(CloudSegmentation, self).__init__()
        self.args = args
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        dataset_class = get_dataset_class(args.dataset_class)
        print("Loading training data")
        self.dataset_train = dataset_class(args.dataset_root, "train", trans)
        print("Loading val data")
        self.dataset_val = dataset_class(args.dataset_root, "val", trans)
        print("Loading test data")
        self.dataset_test = dataset_class(args.dataset_root, "test", trans)

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

        preds = self.model(batch_input)

        loss = self.cross_entropy_loss(preds, batch_labels)
        self.log("val_loss", loss)

        return {"preds": preds, "labels": batch_labels}

    def validation_step_end(self, outputs):
        val_iou = iou(torch.argmax(outputs["preds"], 1), outputs["labels"])
        self.log("val_iou", val_iou)

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
