import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from hydra.core.config_store import ConfigStore
from hydra.types import TargetConf
from torch.utils.data import DataLoader, Dataset, random_split

cs = ConfigStore.instance()


@dataclass
class BaseDataModuleConf(TargetConf):
    batch_size: int = 128
    data_dir: str = "${system.data_dir}"
    cache_dir: Optional[str] = None
    num_workers: int = 8
    seed: int = 42
    normalize: bool = False


@dataclass
class MegDataModuleConf(BaseDataModuleConf):
    _target_: str = "modules.data.MegDataModule"
    downsampling: str = "8"
    variant: str = "baseline"
    num_samples: int = 64
    subset: Optional[Any] = None


@dataclass
class CamCanDataModuleConf(MegDataModuleConf):
    dataset: str = "camcan"
    num_channels: int = 204
    target_field: str = "is_audio"


@dataclass
class MousDataModuleConf(MegDataModuleConf):
    dataset: str = "mous"
    num_channels: int = 298
    target_field: str = "sentence"


cs.store(group="datamodule", name="camcan", node=CamCanDataModuleConf)
cs.store(group="datamodule", name="mous", node=MousDataModuleConf)


class NumpyDataset(Dataset):
    """Numpy dataset."""

    def __init__(
        self,
        csv_file,
        target_field="event",
        subset=None,
        transform=None,
        downsampling=8,
    ):
        """
        Args:
            root_dir (string): Directory with all the files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.csv_fn = Path(csv_file)
        self.csv = pd.read_csv(self.csv_fn)
        self.split = self.csv_fn.name.split(".")[0]
        self.root = Path(csv_file).parent / str(downsampling)
        self.target_field = target_field
        self.subset = subset

        mask = ~self.csv[self.target_field].isna()
        self.csv = self.csv[mask]

        if self.target_field == "cognitive_impairment":
            mask = self.csv["subject_class"].isin({"control", "mci", "dementia"})
            self.csv = self.csv[mask]

        if self.target_field == "event":
            mask = self.csv["event"].isin({6, 7, 8, 9})
            self.csv = self.csv[mask]

        if self.target_field == "tone":
            mask = self.csv["tone"].isin({6, 7, 8, 9})
            self.csv = self.csv[mask]

        if self.target_field == "relative_clause":
            mask = self.csv["sentence"]
            self.csv = self.csv[mask]

        if self.subset is not None:
            if self.target_field == "cognitive_impairment":
                mask = self.csv["recording_site"] == subset
            else:
                mask = self.csv["is_audio"] == subset
            self.csv = self.csv[mask]

        self.transform = transform
        class_values = set(self.csv[self.target_field])
        self.class_labels = {c: i for i, c in enumerate(class_values)}
        subject_values = sorted(set(self.csv["subject"]))
        self.subject_labels = {c: i for i, c in enumerate(subject_values)}

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.csv.iloc[idx]
        fn = self.root / sample.filename

        data = np.load(fn).astype(np.float32)[None]
        label = self.class_labels[sample[self.target_field]]

        if self.transform:
            data = self.transform(data)

        subject = self.subject_labels[sample["subject"]]
        return subject, data, label


class Crop(object):
    """Crop the image in a sample from the start of the window.

    Args:
        output_size (int): Desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, data):
        return data[:, :, : self.output_size]


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (int): Desired output size.
    """

    def __init__(self, output_size, max_offset):
        assert isinstance(output_size, int)
        self.output_size = output_size
        self.max_offset = max_offset

    def __call__(self, data):
        max_offset = np.min([data.shape[-1] - self.output_size, self.max_offset])
        offset = np.random.randint(0, max_offset)

        return data[:, :, offset : offset + self.output_size]


class Pad(object):
    """Pads the input by appending zeros to the end of the window.

    Args:
        output_size (int): Desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, data):
        x = np.zeros(shape=(*data.shape[:-1], self.output_size))
        x[:, :, : data.shape[-1]] = data
        return x


class Normalize(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (int): Desired output size.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data = data - self.mean
        return data / self.std


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        return torch.from_numpy(data.astype(np.float32))


class MegDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        data_dir: str = "./",
        cache_dir: Optional[str] = None,
        num_workers: int = 16,
        seed: int = 42,
        normalize: bool = False,
        dataset: str = "camcan",
        downsampling: int = 8,
        target_field: str = "is_audio",
        num_channels: int = 204,
        num_samples: int = 64,
        variant: str = "baseline",
        subset: Optional[bool] = None,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.seed = seed
        self.batch_size = batch_size
        self.normalize = normalize
        self.dims = (1, num_channels, num_samples)
        self.num_channels = num_channels
        self.num_samples = num_samples
        self.subset = subset

        self.downsampling = downsampling
        self.variant = variant
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.target_field = target_field
        self.full_name = f"{dataset}-{variant}"

        ds = NumpyDataset(
            self.get_root(self.data_dir) / "all.csv", target_field, subset=subset
        )

        self.num_classes = len(ds.class_labels)

    def validate_cache(self, dataset_root, cache_root):
        ds_fn = dataset_root / "build"
        c_fn = cache_root / "build"
        if (not ds_fn.exists()) or (not c_fn.exists()):
            return False
        with ds_fn.open() as f:
            ds_build = f.read()
        with c_fn.open() as f:
            c_build = f.read()

        return ds_build == c_build

    def get_root(self, root):
        return Path(root) / self.full_name / "develop"

    def prepare_data(self):
        dataset_root = self.get_root(self.data_dir)

        if self.cache_dir is not None:
            cache_root = self.get_root(self.cache_dir)
            if not self.validate_cache(dataset_root, cache_root):
                if cache_root.exists():
                    shutil.rmtree(cache_root)
                shutil.copytree(dataset_root, cache_root)
            dataset_root = cache_root

        self.root = dataset_root

        cs = self.num_samples
        tfms = [Crop(cs), Pad(cs), ToTensor()]

        if self.normalize:
            with (self.root / "stats.json").open() as f:
                stats = json.loads(f.read())
                stats = stats[str(self.downsampling)]
            mean = stats["mean"]
            std = np.sqrt(stats["variance"])
            tfms.append(Normalize(mean, std))

        self.tfms = transforms.Compose(tfms)
        target_field = self.target_field
        ds_params = {
            "target_field": target_field,
            "transform": self.tfms,
            "downsampling": self.downsampling,
            "subset": self.subset,
        }
        self.train_ds = NumpyDataset(self.root / "train.csv", **ds_params)
        self.dev_ds = NumpyDataset(self.root / "valid.csv", **ds_params)
        n = len(self.dev_ds)
        split = n // 2
        self.valid_ds, self.dev_test_dataset = random_split(
            self.dev_ds, [n - split, split], generator=torch.Generator().manual_seed(42)
        )
        self.test_ds = NumpyDataset(self.root / "test.csv", **ds_params)

    def train_dataloader(self):
        dataset = self.train_ds
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return loader

    def val_dataloader(self):
        val_dl = DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return val_dl

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
