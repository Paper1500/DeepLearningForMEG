from typing import Any, List

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.metrics.functional import accuracy
from torch.nn import functional as F

import utils


class NullModel(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes=2):
        super(NullModel, self).__init__()
        self.fc = nn.Linear(1, n_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        device = next(self.fc.parameters()).device
        out = torch.ones((batch_size, 1)).to(device)
        out = self.fc(out)
        return out


class ReplicationModule(pl.LightningModule):
    def __init__(self, hparams, datamodule, model, optim):
        super(ReplicationModule, self).__init__()
        self.hparams = hparams
        self.dm = datamodule

        self.model = model
        self.optim = optim
        self.sub_acc = utils.SubjectAccuracy()

    def forward(self, x):
        return self.model(x)

    def _run_step(self, x, y):
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        _, predicted = torch.max(y_hat, dim=1)
        return loss, y_hat, predicted

    def training_step(self, batch, batch_idx):
        if len(batch) == 2:
            x, y = batch
        else:
            subjects, x, y = batch
        loss, y_hat, predicted = self._run_step(x, y)
        acc = accuracy(predicted, y)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=False)
        return loss

    def eval_step(self, batch, batch_idx, prefix):
        if len(batch) == 2:
            x, y = batch
            subjects = y
        else:
            subjects, x, y = batch
        loss, y_hat, predicted = self._run_step(x, y)
        self.sub_acc(subjects, y, predicted)
        acc = accuracy(predicted, y)

        self.log(f"{prefix}_loss", loss, prog_bar=True)
        self.log(f"{prefix}_acc", acc, prog_bar=True)

        return loss

    def eval_dl_end(self, result, prefix, training_end=False):
        target = result.y.cpu().numpy()
        if self.logger:
            probs = F.softmax(torch.exp(result.y_hat), dim=-1).cpu().numpy()
            p, (expected, actual) = utils.get_ecdfs(probs, target)
            curve = utils.plot_to_wandb(utils.plot_ecdfs(p, expected, actual))
            new_prefix = f"best_{prefix}" if training_end else prefix
            self.logger.experiment.log({f"{new_prefix}_prob_curve": curve})

        return result

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        if dataloader_idx is None or dataloader_idx == 0:
            return self.eval_step(batch, batch_idx, "val")
        else:
            return self.eval_step(batch, batch_idx, "dt")

    def validation_epoch_end(self, outputs: List[Any]):
        sub_acc, (lb, ub) = self.sub_acc.compute()
        self.sub_acc.reset()
        acc = {i: float(sub_acc[i]) for i in range(len(sub_acc))}
        self.log("val_sub_acc", acc, prog_bar=False)
        self.log("val_sub_lb", lb, prog_bar=True)
        self.log("val_sub_ub", ub, prog_bar=False)

    def test_epoch_end(self, outputs: List[Any]):
        sub_acc, (lb, ub) = self.sub_acc.compute()
        acc = {i: float(sub_acc[i]) for i in range(len(sub_acc))}
        self.log("test_sub_acc", acc, prog_bar=False)
        self.log("test_sub_lb", lb, prog_bar=False)
        self.log("test_sub_ub", ub, prog_bar=False)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        if dataloader_idx is None:
            return self.eval_step(batch, batch_idx, "test")
        if dataloader_idx == 0:
            return self.eval_step(batch, batch_idx, "val")
        else:
            return self.eval_step(batch, batch_idx, "dt")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.optim.lr,
            eps=self.optim.eps,
            weight_decay=self.optim.weight_decay,
            betas=self.optim.betas,
        )

    def prepare_data(self):
        self.dm.prepare_data()

    def train_dataloader(self):
        return self.dm.train_dataloader()

    def val_dataloader(self):
        return self.dm.val_dataloader()

    def test_dataloader(self):
        return self.dm.test_dataloader()

    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument("--optimizer", default="adam", type=str)
        parser.add_argument("--patience", default=8, type=int)
        parser.add_argument("--n_classes", default=2, type=int)
