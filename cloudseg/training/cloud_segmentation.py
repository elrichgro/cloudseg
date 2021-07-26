import pytorch_lightning as pl
import torch
from torchvision import transforms
from pytorch_lightning.metrics.functional.classification import iou

from cloudseg.models import get_model


class CloudSegmentation(pl.LightningModule):
    def __init__(self, **kwargs):
        super(CloudSegmentation, self).__init__()
        self.save_hyperparameters()

        self.model = get_model(self.hparams.model_name, **kwargs)

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(
            reduction="mean",
            ignore_index=-1 if kwargs.get("ignore_background") else -100,
            weight=torch.tensor([1.0, 1.0 * kwargs.get("cloud_weight", 1.0)]),
        )

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
        mask = outputs["labels"] != -1
        val_iou = iou(
            torch.argmax(outputs["preds"], 1)[mask],
            outputs["labels"][mask],
        )
        self.log("val_iou", val_iou, prog_bar=True)

    def test_step(self, batch, batch_idx):
        batch_input = batch["irc"]
        batch_labels = batch["label"].squeeze(1)

        pred_labels = self.model(batch_input)

        loss = self.cross_entropy_loss(pred_labels, batch_labels)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
