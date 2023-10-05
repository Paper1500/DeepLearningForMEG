from dataclasses import dataclass
from typing import Any

import torch.nn as nn
from hydra.types import TargetConf


@dataclass
class BaseModelConf(TargetConf):
    n_classes: int = 2
    downsampler: Any = None


class SpatialEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(1, out_channels, (in_channels, 1))

    def forward(self, x):
        x = self.conv(x)
        return x.transpose(1, 2)
