from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore

from modules.base import BaseModelConf

cs = ConfigStore.instance()


@dataclass
class TimeConvConf(BaseModelConf):
    _target_: str = "modules.time.TimeConvModel"
    kernel_size: int = 5
    stride: int = 2
    dilation: int = 3
    batch_norm: bool = False
    l1_kernels: int = 32
    l2_kernels: int = 64
    l3_kernels: int = 32
    dropout: float = 0.25


@dataclass
class TimeConvOldConf(TimeConvConf):
    _target_: str = "modules.time.TimeConvModel"
    stride: int = 3


@dataclass
class TimeConvSmallConf(TimeConvConf):
    _target_: str = "modules.time.TimeConvModel"
    dilation: int = 2


@dataclass
class TimeConvPlusModelConf(BaseModelConf):
    _target_: str = "modules.time.TimeConvPlusModel"
    kernel_size: int = 5


cs.store(group="model", name="timeconv", node=TimeConvConf)
cs.store(group="model", name="timeconvold", node=TimeConvOldConf)
cs.store(group="model", name="timeconvsmall", node=TimeConvSmallConf)
cs.store(group="model", name="timeconvplus", node=TimeConvPlusModelConf)


cs.store(group="model", name="timeconv", node=TimeConvConf)


class TimeConvModel(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int = 2,
        downsampler: nn.Module = None,
        kernel_size: int = 5,
        stride: int = 2,
        dilation: int = 3,
        batch_norm: bool = False,
        l1_kernels: int = 32,
        l2_kernels: int = 64,
        l3_kernels: int = 32,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.downsampler = downsampler
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.batch_norm = batch_norm
        self.l1_kernels = l1_kernels
        self.l2_kernels = l2_kernels
        self.l3_kernels = l3_kernels
        self.dropout = dropout
        dense_in = self.l3_kernels * self.n_channels

        self.conv = torch.nn.Sequential(
            self.create_layer(1, self.l1_kernels),
            self.create_layer(self.l1_kernels, self.l2_kernels),
            self.create_layer(self.l2_kernels, self.l3_kernels),
            nn.AdaptiveMaxPool2d((n_channels, 1)),
            nn.ReLU(),
        )
        self.head = torch.nn.Sequential(
            nn.Flatten(), nn.Linear(dense_in, 128), nn.ReLU(), nn.Linear(128, n_classes)
        )

    def create_layer(self, in_f, out_f):
        layer_params = {
            "kernel_size": (1, self.kernel_size),
            "padding": (0, (self.dilation * (self.kernel_size - 1)) // 2),
            "stride": (1, self.stride),
            "dilation": (1, self.dilation),
        }
        if self.batch_norm:
            layers = [nn.BatchNorm2d(in_f)]
        else:
            layers = list()

        return nn.Sequential(
            *layers,
            nn.Dropout(self.dropout),
            nn.Conv2d(in_f, out_f, **layer_params),
            nn.ReLU()
        )

    @property
    def accumulate_grad_batches(self):
        return 1

    def forward(self, x):
        if self.downsampler is not None:
            x = self.downsampler(x)

        x = self.conv(x)
        return self.head(x)


class TimeConvPlusModel(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int = 2,
        downsampler: nn.Module = None,
        kernel_size: int = 5,
    ):
        super().__init__()

        self.downsampler = downsampler
        self.kernel_size = kernel_size

        layer_params = {
            "kernel_size": (1, self.kernel_size),
            "padding": (0, self.kernel_size // 2),
            "dilation": (1, 2),
        }

        self.time_conv = torch.nn.Sequential(
            nn.Dropout(0.5),
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 32, **layer_params),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, **layer_params),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, **layer_params),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((n_channels, 1)),
        )

        self.spatial_conv = torch.nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_channels, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # self.conv = torch.nn.Sequential(self.time_conv, self.spatial_conv)
        self.head = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    @property
    def accumulate_grad_batches(self):
        return 4

    def forward(self, x):
        if self.downsampler is not None:
            x = self.downsampler(x)

        # x = self.conv(x)
        x = self.time_conv(x)
        x = x.flatten(-2)
        x = self.spatial_conv(x)
        return self.head(x)
