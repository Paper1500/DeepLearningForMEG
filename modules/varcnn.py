from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from torch.nn import functional as F

cs = ConfigStore.instance()

from modules.base import BaseModelConf


@dataclass
class VarCnnConf(BaseModelConf):
    _target_: str = "modules.varcnn.VarCnn"

@dataclass
class LfCnnConf(BaseModelConf):
    _target_: str = "modules.varcnn.LfCnn"

cs.store(group="model", name="varcnn", node=VarCnnConf)
cs.store(group="model", name="lfcnn", node=LfCnnConf)

class AdaptiveBase(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int=2, downsampler: nn.Module=None):
        super(AdaptiveBase, self).__init__()
        dense_in = 32 * n_samples // 2
        self.downsampler = downsampler
        self.head = torch.nn.Sequential(
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(dense_in, n_classes)
        )

    @property
    def accumulate_grad_batches(self):
        return 1

    def forward(self, x):
        if self.downsampler is not None:
            x = self.downsampler(x)
        
        x = self.conv(x)
        return self.head(x)


class LfCnn(AdaptiveBase):
    
    def __init__(self, n_channels: int, n_samples: int, n_classes: int=2, downsampler: nn.Module=None):
        super(LfCnn, self).__init__(n_channels, n_samples, n_classes, downsampler)
        self.conv = torch.nn.Sequential(
            nn.Conv2d(1, 32, (n_channels, 1)),
            nn.Conv2d(32, 32, (1, 7), padding=(0, 3), groups=32),
            nn.Flatten(1, 2),
        )


class VarCnn(AdaptiveBase):
    
    def __init__(self, n_channels: int, n_samples: int, n_classes: int=2, downsampler: nn.Module=None):
        super(VarCnn, self).__init__(n_channels, n_samples, n_classes, downsampler)
        self.conv = torch.nn.Sequential(
            nn.Conv2d(1, 32, (n_channels, 1)),
            nn.Flatten(1, 2),
            nn.Conv1d(32, 32, 7, padding=3),
        )
        

