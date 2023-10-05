from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from torch.nn import functional as F

from modules.base import BaseModelConf, SpatialEncoder

cs = ConfigStore.instance()


@dataclass
class BaseTimeResModelConf(BaseModelConf):
    n_filters: int = 16
    layers_per_block: int = 4
    embedding_dim: int = 16
    dropout: float = 0.5


@dataclass
class TimeResConf(BaseTimeResModelConf):
    _target_: str = "modules.resnet.TimeResModel"


@dataclass
class TimeSpatialResModelConf(BaseTimeResModelConf):
    _target_: str = "modules.resnet.TimeSpatialResModel"


@dataclass
class TimeSpatialSeperableResModelConf(BaseTimeResModelConf):
    _target_: str = "modules.resnet.TimeSpatialSeparableResModel"


@dataclass
class TimeResLiteConf(BaseModelConf):
    _target_: str = "modules.resnet.TimeResModel"
    n_filters: int = 8
    layers_per_block: int = 2
    embedding_dim: int = 8
    dropout: float = 0.5


@dataclass
class TimeResHeavyConf(BaseModelConf):
    _target_: str = "modules.resnet.TimeResModel"
    n_filters: int = 32
    layers_per_block: int = 4
    embedding_dim: int = 32
    dropout: float = 0.5


cs.store(group="model", name="timeres", node=TimeResConf)
cs.store(group="model", name="timespatialres", node=TimeSpatialResModelConf)
cs.store(group="model", name="timespatialsepres", node=TimeSpatialSeperableResModelConf)
cs.store(group="model", name="timereslite", node=TimeResLiteConf)
cs.store(group="model", name="timeresheavy", node=TimeResHeavyConf)


class ResBlock(torch.nn.Module):
    def __init__(self, n_filters: int, kernel_size: int, dilation: int):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation = dilation

        conv_params = {
            "kernel_size": (1, kernel_size),
            "dilation": (1, dilation),
            "padding": (0, dilation),
        }

        self.conv1 = nn.Conv2d(self.n_filters, self.n_filters, **conv_params)
        self.conv2 = nn.Conv2d(self.n_filters, self.n_filters, **conv_params)

        self.bn = nn.BatchNorm2d(self.n_filters)

    def forward(self, x):
        x = self.bn(x)
        identity = x

        out = self.conv1(x)
        out = F.relu(out)

        out = self.conv2(out)

        out += identity
        out = F.relu(out)

        return out


class Encoder(torch.nn.Module):
    def __init__(self, n_filters: int, n_blocks: int):
        super().__init__()
        self.n_filters = n_filters
        self.n_blocks = n_blocks

        self.blocks = nn.Sequential(
            *[ResBlock(self.n_filters, 3, 3) for _ in range(n_blocks)]
        )
        self.conv_in = nn.Conv2d(
            1,
            self.n_filters,
            kernel_size=(1, 3),
            dilation=(1, 3),
            padding=(0, 3),
        )
        self.conv_out = nn.Conv2d(
            self.n_filters,
            1,
            kernel_size=(1, 2),
            stride=(1, 2),
        )

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv_out(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, n_filters: int, n_blocks: int):
        super().__init__()
        self.n_filters = n_filters
        self.n_blocks = n_blocks

        self.deconv_in = nn.ConvTranspose2d(
            1,
            self.n_filters,
            kernel_size=(1, 3),
            dilation=(1, 3),
            padding=(0, 3),
        )
        self.deconv_out = nn.ConvTranspose2d(
            self.n_filters,
            1,
            kernel_size=(1, 2),
            stride=(1, 2),
        )

        self.blocks = nn.Sequential(
            *[ResBlock(self.n_filters, 3, 3) for _ in range(n_blocks)]
        )

    def forward(self, x):
        x = self.deconv_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.deconv_out(x)
        return x


class BaseResModel(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int = 2,
        downsampler: nn.Module = None,
        n_filters: int = 16,
        layers_per_block: int = 4,
        embedding_dim: int = 16,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.layers_per_block = layers_per_block
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.downsampler = downsampler

        self.encoder = SpatialEncoder(n_channels, self.embedding_dim)

        self.b1 = self.create_block(
            1,
            n_filters,
            self.layers_per_block,
            dropout=dropout,
        )

        self.b2 = self.create_block(
            n_filters,
            n_filters,
            self.layers_per_block,
            dropout=dropout,
        )

        self.b3 = self.create_block(
            n_filters,
            n_filters,
            self.layers_per_block,
            dropout=dropout,
        )

        self.conv = nn.Sequential(
            self.encoder,
            self.b1,
            self.b2,
            self.b3,
        )

        self.head = self.create_head()

    def create_block(
        self,
        in_filters: int,
        filters: int,
        n_layers: int,
        dropout: float = 0.5,
    ):
        in_conv = nn.Conv2d(
            in_filters,
            filters,
            kernel_size=(1, 3),
            stride=(1, 2),
            padding=(0, 1),
        )
        return nn.Sequential(
            nn.Dropout(dropout),
            in_conv,
            *[ResBlock(filters, 3, 3) for _ in range(n_layers)],
        )

    @property
    def accumulate_grad_batches(self):
        return 1

    def forward(self, x):
        if self.downsampler is not None:
            x = self.downsampler(x)

        x = self.conv(x)
        return self.head(x)


class TimeResModel(BaseResModel):
    def create_head(self):
        dense_in = self.n_samples * self.embedding_dim // 8
        return torch.nn.Sequential(
            nn.Conv2d(self.n_filters, 1, 1),
            nn.Flatten(),
            nn.Linear(dense_in, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_classes),
        )


class TimeSpatialResModel(BaseResModel):
    def create_head(self):
        dense_in = self.n_filters * self.n_samples // 8
        # dense_in = self.embedding_dim * self.n_samples // 8
        return torch.nn.Sequential(
            nn.Conv2d(self.n_filters, 1, 1),
            nn.Conv2d(1, self.n_filters, (self.embedding_dim, 1)),
            # nn.Conv2d(1, self.embedding_dim, (self.embedding_dim, 1)),
            nn.Flatten(),
            nn.Linear(dense_in, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_classes),
        )


class TimeSpatialSeparableResModel(BaseResModel):
    def create_head(self):
        dense_in = self.n_filters * self.n_samples // 8
        return torch.nn.Sequential(
            nn.Conv2d(
                self.n_filters,
                self.n_filters,
                (self.embedding_dim, 1),
                groups=self.n_filters,
            ),
            nn.Flatten(),
            nn.Linear(dense_in, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_classes),
        )
