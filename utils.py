import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from torchmetrics import Metric

import wandb


def confidence_interval(data, p=0.05):
    data = data[~np.isnan(data)]
    n = len(data)
    df = n - 1
    mean = np.mean(data)
    stderr = np.std(data, ddof=1) / np.sqrt(n)
    t_stat = stats.t.ppf(1 - (p / 2), df)
    error = t_stat * stderr
    return mean - error, mean + error


def tukey(models, data):
    data = np.vstack(data).T
    df = (
        pd.DataFrame(data, columns=models.index)
        .unstack()
        .reset_index()
        .drop(columns=["level_1"])
    )
    df.columns = ["module", "accuracy"]
    results = pairwise_tukeyhsd(df["accuracy"], df["module"])
    df = pd.DataFrame(results.summary())

    df.columns = df.iloc[0].map(str)
    df = df.iloc[1:].reset_index().drop(columns=["index"])
    df = pd.DataFrame(df.applymap(lambda x: x.data))
    df["module1"] = df.group1.map(lambda x: models.loc[x].module)
    df["module2"] = df.group2.map(lambda x: models.loc[x].module)
    return df


def anova(data):
    data = np.vstack(data)
    k = data.shape[0]
    n = np.prod(data.shape)
    me = np.mean(data, axis=1)
    va = np.var(data, axis=1, ddof=1)
    mse = va.mean()
    msb = me.var(ddof=1) * data.shape[1]
    return 1 - stats.f.cdf(msb / mse, k - 1, n - k)


def grouped_accuracy(pred, target, groups):
    mask = torch.zeros(groups.max() + 1, len(target))
    mask[groups, torch.arange(len(target))] = 1
    return (mask * (pred == target)).sum(axis=-1) / mask.sum(axis=-1)


def get_ecdfs(predicted, ground_truth):
    probs = list()
    correct = list()
    for i in range(predicted.shape[-1]):
        probs.append(predicted[:, i])
        correct.append(ground_truth == i)

    probs = np.hstack(probs)
    correct = np.hstack(correct)

    n = correct.sum()
    bins = 100
    expected = np.zeros((bins,))
    actual = np.zeros((bins,))
    totals = np.zeros((bins,))
    for i, p in enumerate(np.linspace(0, 1, bins)):
        mask = probs <= p
        expected[i] = probs[mask].sum()
        actual[i] = correct[mask].sum()
        totals[i] = mask.sum()

    p = np.linspace(0, 1, bins)

    expected /= n
    actual /= n

    return p, (expected, actual)


def plot_ecdfs(x, expected, actual, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.plot(x, expected, label="expected", color="b")
    ax.plot(x, actual, label="actual", c="r")
    return fig


def plot_to_wandb(fig):
    image = wandb.Image(fig)
    plt.close(fig)
    return image


class SubjectAccuracy(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(compute_on_step=False, dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.Tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.Tensor(0), dist_reduce_fx="sum")

    def reset(self):
        self.correct *= 0
        self.total *= 0

    def update(
        self, subjects: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor
    ):
        n_subjects = subjects.max() + 1
        bs = subjects.shape[0]
        if self.correct.shape[0] < n_subjects:
            ext = torch.zeros(
                n_subjects - len(self.correct), device=self.correct.device
            )
            self.correct = torch.cat((self.correct, ext))
            self.total = torch.cat((self.total, ext))

        data = torch.zeros(n_subjects, bs, device=self.correct.device)
        correct = targets == preds
        data[subjects, torch.arange(bs)] = correct.to(torch.float)
        self.correct[:n_subjects] += data.sum(axis=-1)
        data[subjects, torch.arange(bs)] = 1
        self.total[:n_subjects] += data.sum(axis=-1)

    def compute(self):
        result = self.correct / self.total
        return result, confidence_interval(result.detach().cpu().numpy())
