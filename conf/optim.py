from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore
from hydra.types import TargetConf

cs = ConfigStore.instance()


@dataclass
class AdamBaseConf(TargetConf):
    betas: tuple = (0.9, 0.999)
    lr: float = 1e-3
    eps: float = 1e-8
    weight_decay: float = 0


@dataclass
class AdamConf(AdamBaseConf):
    # _target_ = "torch.optim.Adam"
    amsgrad: bool = False


# @dataclass
# class AdamWConf(AdamConf):
#     _target_ = "torch.optim.AdamW"


cs.store(group="optim", name="adam", node=AdamConf)
# cs.store(group="optim", name="adamw", node=AdamWConf)
