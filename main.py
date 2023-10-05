import hydra
import numpy as np
import torch

import conf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig

from modules.data import MegDataModule

from train import train

from modules import ReplicationModule, AutoEncoderModule

from modules.autoencoder_module import Encoder

import seaborn as sns

sns.set()


def get_project(branch):
    if branch == "":
        return "dev"
    return ".".join([w for w in branch.split("/") if w not in ["refs", "heads"]])


def conf_init(cfg, key):
    return hydra.utils.instantiate(getattr(cfg, key)) if hasattr(cfg, key) else None


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    dm = hydra.utils.instantiate(cfg.datamodule)

    _, n_channels, n_samples = dm.dims

    if cfg.downsampler.enabled:
        if cfg.downsampler.path:
            downsampler = AutoEncoderModule.load_from_checkpoint(
                cfg.downsampler.path, datamodule=dm, hparams=cfg
            )
            if cfg.downsampler.frozen:
                downsampler.freeze()
            downsampler = downsampler.encoder
        else:
            downsampler = Encoder(32, 4)

        batch_size = dm.batch_size // 4
        dm.batch_size = batch_size
        cfg.datamodule.batch_size = batch_size
        cfg.trainer.accumulate_grad_batches *= 4

        n_samples = n_samples // 2
    else:
        downsampler = None

    accumulate_grad_batches = cfg.trainer.accumulate_grad_batches
    batch_size = dm.batch_size

    best_loss = np.inf
    ckpt = None
    for i in range(cfg.trainer.n_models):
        model = hydra.utils.instantiate(
            cfg.model,
            n_samples=n_samples,
            n_channels=n_channels,
            n_classes=dm.num_classes,
        )
        model.downsampler = downsampler

        if model.accumulate_grad_batches > 1:
            dm.batch_size = batch_size // model.accumulate_grad_batches
            cfg.datamodule.batch_size = dm.batch_size
            cfg.trainer.accumulate_grad_batches = (
                accumulate_grad_batches * model.accumulate_grad_batches
            )

        module = ReplicationModule(cfg, dm, model, cfg.optim)

        trainer = train(cfg, module)
        loss = trainer.checkpoint_callback.kth_value
        if loss < best_loss:
            best_loss = loss
            ckpt = trainer.checkpoint_callback.best_model_path

    # trainer.test(test_dataloaders=dm.val_dataloader(), ckpt_path=ckpt)
    trainer.test(test_dataloaders=dm.test_dataloader(), ckpt_path=ckpt)


if __name__ == "__main__":
    main()
