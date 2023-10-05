from dataclasses import dataclass

import numpy as np


def sum_of_squares(data: np.array, mean: float, axis=None):
    squares = (data - mean) ** 2
    return np.sum(squares, axis=axis)


@dataclass
class SSInfo(object):
    n: int
    mean: float
    ss: float

    @staticmethod
    def from_array(data: np.array) -> "SSInfo":
        n = np.prod(data.shape)
        mean = data.mean()
        ss = sum_of_squares(data, mean)
        return SSInfo(n, mean, ss)

    def combine(self, other: "SSInfo") -> "SSInfo":
        n = self.n + other.n
        delta = other.mean - self.mean
        mean = self.mean + delta * other.n / n
        ss = self.ss + other.ss + delta**2 * self.n * other.n / n
        return SSInfo(n, mean, ss)

    @property
    def variance(self) -> float:
        return self.ss / self.n

    def to_dict(self):
        return {"mean": self.mean, "variance": self.variance}


def mean_variance(data_iter):
    infos = list(map(SSInfo.from_array, data_iter))
    info = infos.pop()
    while len(infos) > 0:
        info = info.combine(infos.pop())
    return info.mean, info.variance


def get_normalization_stats(x):
    return {"mean": x.mean(), "std": x.std()}


def _normalize(baseline, window, stats):
    window -= stats["mean"]
    window /= stats["std"]
    baseline -= stats["mean"]
    baseline /= stats["std"]
    return baseline, window, stats


def baseline_normalize(baseline, window):
    stats = get_normalization_stats(baseline)
    return _normalize(baseline, window, stats)


def window_normalize(baseline, window):
    stats = get_normalization_stats(window)
    return _normalize(baseline, window, stats)


def all_normalize(baseline, window):
    data = np.hstack([baseline, window])
    stats = get_normalization_stats(data)
    return _normalize(baseline, window, stats)


def scale_normalize(baseline, window):
    data = np.hstack([baseline, window])
    lb = np.percentile(data, 1)
    ub = np.percentile(data, 99)
    data = np.clip(data, lb, ub)

    data -= data.min()
    data /= data.max()
    data = (data * 2) - 1

    stats = get_normalization_stats(data)

    baseline = data[:, : baseline.shape[-1]]
    window = data[:, baseline.shape[-1] :]

    return baseline, window, stats


def arbitrary_order(data, seed=42):
    """
    Puts the input in an arbitrary order.
    First sorts and then deterministically shuffles the input.

    determinstic_shuffle(data) == determinstic_shuffle(determinstic_shuffle((data))
    """
    data = list(np.sort(data))
    rng = np.random.RandomState(seed)
    rng.shuffle(data)
    return data
