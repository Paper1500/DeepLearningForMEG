import json
import numpy as np
import pandas as pd
import shutil

from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import datasets.utils as utils
from datasets.megio import CamCanFile, MegFile, MousFile


class MegDataset(ABC):
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        dataset: str = "camcan",
        variant: str = "baseline",
        window_size: int = 600,
        filter_type: str = "mne",
        window_normalization: str = "baseline",
        sample_generation: str = "numpy",
        include_baseline: bool = True,
        random_seed: int = 42,
        test_run: bool = False,
        downsampling: List[int] = [8],
        max_subjects: int = 1000,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path("tmp")
        self.random_seed = random_seed
        self.test_run = test_run
        self.downsampling = downsampling
        self.max_subjects = max_subjects

        self.dataset = dataset
        self.variant = variant
        self.filter_type = filter_type
        self.window_normalization = window_normalization
        self.sample_generation = sample_generation
        self.include_baseline = include_baseline
        self.window_size = window_size

        np.random.seed(self.random_seed)

    def generate(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True)

        for ds in self.downsampling:
            ds_dir = self.temp_dir / str(ds)
            ds_dir.mkdir()

        data_fn = self.temp_dir / "all.csv"
        stats_fn = self.temp_dir / "stats.json"

        meta = list()
        global_stats = dict()
        for raw_file in self.get_files():
            print(raw_file.filename)

            for i, ((baseline, window), info) in enumerate(raw_file.generate_samples()):
                output_fn = info["filename"]

                if self.window_normalization == "baseline":
                    baseline, window, stats = utils.baseline_normalize(baseline, window)
                elif self.window_normalization == "window":
                    baseline, window, stats = utils.window_normalize(baseline, window)
                elif self.window_normalization == "all":
                    baseline, window, stats = utils.all_normalize(baseline, window)
                elif self.window_normalization == "scale":
                    baseline, window, stats = utils.scale_normalize(baseline, window)
                elif self.window_normalization == "raw":
                    stats = utils.get_normalization_stats(window)
                else:
                    raise ValueError(
                        f"{self.window_normalization} window normalization is not supported"
                    )

                info = {**info, **stats}

                if self.include_baseline:
                    data = np.hstack([baseline, window])
                else:
                    data = window
                for ds in self.downsampling:
                    ds_dir = self.temp_dir / str(ds)
                    ds_data = raw_file.downsample(data, ds)
                    ds_info = utils.SSInfo.from_array(ds_data)

                    if ds in global_stats:
                        global_stats[ds] = global_stats[ds].combine(ds_info)
                    else:
                        global_stats[ds] = ds_info

                    np.save(ds_dir / output_fn, ds_data)

                meta.append(info)

        global_stats = {ds: global_stats[ds].to_dict() for ds in global_stats}

        df = pd.DataFrame(meta)
        df.to_csv(data_fn, index=False)
        subjects = utils.arbitrary_order(list(set(df.subject)))
        np.random.seed(self.random_seed)
        np.random.shuffle(subjects)

        with stats_fn.open("w") as f:
            f.write(json.dumps(global_stats))

        n_test = max(len(subjects) // 5, 1)
        train = subjects[: -2 * n_test]
        valid = subjects[-2 * n_test : -n_test]
        test = subjects[-n_test:]
        for split, subs in [("train", train), ("valid", valid), ("test", test)]:
            mask = df.subject.isin(subs)
            df[mask].to_csv(self.temp_dir / f"{split}.csv", index=False)

        if self.test_run:
            shutil.rmtree(self.temp_dir)
            return

        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        shutil.move(self.temp_dir, self.output_dir)

    def get_files(self) -> List[MegFile]:
        file_args = {
            "filter_type": self.filter_type,
            "window_size": self.window_size,
            "sample_generation": self.sample_generation,
        }
        if self.dataset == "camcan":
            return self.get_camcan_files(file_args)
        elif self.dataset == "mous":
            return self.get_mous_files(file_args)
        else:
            raise ValueError(f"{self.dataset} is not supported")

    def get_camcan_files(self, file_args) -> List[CamCanFile]:
        subjects = list(sorted(self.input_dir.glob("CC*")))
        np.random.shuffle(subjects)
        subjects = subjects[: self.max_subjects]

        for sub in subjects:
            fn = Path(sub, "passive", "passive_raw.fif")
            yield CamCanFile(filename=fn, subject=sub.name, **file_args)

    def get_mous_files(self, file_args) -> List[MousFile]:
        subjects = list(sorted(self.input_dir.glob("sub*")))
        np.random.shuffle(subjects)
        subjects = subjects[: self.max_subjects]
        for sub in subjects:
            paths = [
                path
                for path in sub.glob("*/*.ds")
                if "visual" in path.name or "auditory" in path.name
            ]
            for fn in paths:
                yield MousFile(filename=fn, subject=sub.name, **file_args)
