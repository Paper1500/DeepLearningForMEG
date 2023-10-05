import hydra
import torch

import conf
from omegaconf import DictConfig

from modules.data import MegDataModule
from train import train

from modules import AutoEncoderModule

from pipeline import export_variable

import seaborn as sns

sns.set()


def get_project(branch):
    if branch == "":
        return "dev"
    return ".".join([w for w in branch.split("/") if w not in ["refs", "heads"]])


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    dm = hydra.utils.instantiate(cfg.datamodule)
    module = AutoEncoderModule(cfg, dm)

    trainer = train(cfg, module)

    export_variable("checkpoint_path", trainer.checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()
