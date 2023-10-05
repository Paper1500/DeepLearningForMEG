from dataclasses import dataclass

import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore

from modules.base import BaseModelConf, SpatialEncoder

cs = ConfigStore.instance()


@dataclass
class TransformerConf(BaseModelConf):
    _target_: str = "modules.transformer.TransformerModel"
    d_model: int = 16
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.5


@dataclass
class TransformerLargeConf(BaseModelConf):
    _target_: str = "modules.transformer.TransformerModel"
    d_model: int = 32
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.5


@dataclass
class TransformerHugeConf(BaseModelConf):
    _target_: str = "modules.transformer.TransformerModel"
    d_model: int = 96
    n_heads: int = 16
    n_layers: int = 4
    dropout: float = 0.5


cs.store(group="model", name="transformer", node=TransformerConf)
cs.store(group="model", name="transformerlarge", node=TransformerLargeConf)
cs.store(group="model", name="transformerhuge", node=TransformerHugeConf)


class TransformerModel(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int = 2,
        downsampler: nn.Module = None,
        d_model: int = 16,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.downsampler = downsampler
        self.dropout = dropout
        self.n_samples = n_samples
        self.encoder = SpatialEncoder(n_channels, self.d_model)
        self.do = nn.Dropout(dropout)
        self.pos = nn.Embedding(self.d_model, self.n_samples)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers
        )

        self.head = torch.nn.Sequential(
            nn.Flatten(), nn.Linear(self.d_model * n_samples, n_classes)
        )

    @property
    def accumulate_grad_batches(self):
        return 1

    def forward(self, x):
        if self.downsampler is not None:
            x = self.downsampler(x)
        x = self.encoder(x)
        x = self.do(x)
        pos = self.pos(torch.arange(self.d_model, device=x.device))
        x += pos[None]
        x = x.flatten(1, 2)
        x = x.permute(2, 0, 1)
        x = self.transformer(x)
        x = x.permute(1, 2, 0)
        return self.head(x)
