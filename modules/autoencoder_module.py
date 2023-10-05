import hydra
import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from modules.resnet import Decoder, Encoder


class AutoEncoderModule(pl.LightningModule):
    def __init__(
        self,
        hparams,
        datamodule: pl.LightningDataModule,
        n_filters: int = 32,
        n_blocks: int = 4,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_filters = n_filters
        # self.save_hyperparameters()
        self.datamodule = datamodule

        self.encoder = Encoder(self.n_filters, self.n_blocks)
        self.decoder = Decoder(self.n_filters, self.n_blocks)

    def forward(self, x):
        return self.encoder(x)

    def _run_step(self, x):
        z = self(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x, x_hat)
        return loss

    def training_step(self, batch, batch_idx):
        if len(batch) == 2:
            x, y = batch
        else:
            subjects, x, y = batch

        loss = self._run_step(x)

        self.log("train_loss", loss, on_epoch=True)
        return loss

    def eval_step(self, batch, batch_idx, prefix):
        if len(batch) == 2:
            x, y = batch
        else:
            subjects, x, y = batch
        loss = self._run_step(x)

        self.log(f"{prefix}_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        if dataloader_idx is None or dataloader_idx == 0:
            return self.eval_step(batch, batch_idx, "val")
        else:
            return self.eval_step(batch, batch_idx, "dt")

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        if dataloader_idx is None:
            return self.eval_step(batch, batch_idx, "test")
        if dataloader_idx == 0:
            return self.eval_step(batch, batch_idx, "val")
        else:
            return self.eval_step(batch, batch_idx, "dt")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def prepare_data(self):
        self.datamodule.prepare_data()

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()
