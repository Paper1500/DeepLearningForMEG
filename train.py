from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import conf
import hydra
import pytorch_lightning as pl
import seaborn as sns
import torch

sns.set()

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer


def get_project(branch):
    if branch == "":
        return "dev"
    return ".".join([w for w in branch.split("/") if w not in ["refs", "heads"]])


def train(cfg: DictConfig, module: pl.LightningModule) -> Trainer:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.trainer.logger and (not cfg.trainer.fast_dev_run):
        logger_params = {"project": cfg.trainer.logger.project}

        if cfg.build.id:
            build_id = str(cfg.build.id)
            logger_params["tags"] = [build_id]
            logger_params["group"] = build_id

        logger = hydra.utils.instantiate(cfg.trainer.logger, **logger_params)
    else:
        logger = None

    use_gpu = cfg.trainer.use_gpu and torch.cuda.is_available()
    es = EarlyStopping(monitor="val_loss")
    mc = ModelCheckpoint(monitor="val_loss", save_top_k=1)
    trainer = Trainer(
        gpus=1 if use_gpu else None,
        auto_select_gpus=use_gpu,
        max_epochs=cfg.trainer.max_epochs,
        min_epochs=cfg.trainer.min_epochs,
        progress_bar_refresh_rate=cfg.trainer.progress_bar_refresh_rate,
        logger=logger,
        fast_dev_run=cfg.trainer.fast_dev_run,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        callbacks=[es, mc],
    )

    trainer.fit(module)

    return trainer
