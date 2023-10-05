from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as models
from hydra.core.config_store import ConfigStore
from torch.nn import functional as F

cs = ConfigStore.instance()

from modules.base import BaseModelConf


@dataclass
class GoogLeNetConf(BaseModelConf):
    _target_: str = "modules.vision.GoogLeNet"


@dataclass
class ResNetConf(BaseModelConf):
    _target_: str = "modules.vision.ResNet"


@dataclass
class VGGConf(BaseModelConf):
    _target_: str = "modules.vision.VGG"

cs.store(group="model", name="googlenet", node=GoogLeNetConf)
cs.store(group="model", name="resnet", node=ResNetConf)
cs.store(group="model", name="vgg", node=VGGConf)


class ResNet(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int=2, downsampler: nn.Module=None):
        super().__init__()
        self.body = nn.Conv2d(1,3, 1)
        self.head = models.resnet18(num_classes=n_classes)


    @property
    def accumulate_grad_batches(self):
        return 4

    def forward(self, x):
        if self.downsampler is not None:
            x = self.downsampler(x)
        
        x = self.body(x)
        return self.head(x)


class VGG(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int=2, downsampler: nn.Module=None):
        super().__init__()
        self.body = nn.Conv2d(1,3, 1)
        self.head = models.vgg19(num_classes=n_classes)


    @property
    def accumulate_grad_batches(self):
        return 4

    def forward(self, x):
        if self.downsampler is not None:
            x = self.downsampler(x)
        
        x = self.body(x)
        return self.head(x)



class GoogLeNet(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int=2, downsampler: nn.Module=None):
        super().__init__()
        self.body = nn.Conv2d(1,3, 1)
        self.head = models.GoogLeNet(num_classes=n_classes, aux_logits=False)

    @property
    def accumulate_grad_batches(self):
        return 4
        
    def forward(self, x):
        if self.downsampler is not None:
            x = self.downsampler(x)
        
        x = self.body(x)
        return self.head(x)



