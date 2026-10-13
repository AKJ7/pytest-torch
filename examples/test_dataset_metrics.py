"""Dataset evaluation: run a loader, aggregate metrics, catch accuracy regressions.

Run: pytest examples/test_dataset_metrics.py -v
Record baselines:  pytest examples/test_dataset_metrics.py --torch-update-metrics
(stored in .torch-metric-baselines.json, per-sample values in .torch-metric-baselines/;
commit both so CI compares against them)

A baseline records what it measured: the `context` (dataset, split, preprocessing), a
fingerprint of the data, the sample selection, the metric configuration and the checkpoint.
A run that measures something else fails instead of being compared. Per-sample metrics are
compared paired, with a confidence interval: noise on a small set isn't a regression.
"""
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytest_torch import evaluate


def make_data(n=512):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 2, generator=g)
    y = (x[:, 0] + x[:, 1] > 0).long()  # linearly separable
    return TensorDataset(x, y)


@pytest.fixture(scope="module")
def val_loader():
    return DataLoader(make_data(), batch_size=50)  # uneven last batch: weighting matters


@pytest.fixture(scope="module")
def trained_model():
    torch.manual_seed(0)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    for x, y in DataLoader(make_data(), batch_size=64, shuffle=False):
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        optimizer.step()
    return model.eval()


def accuracy(output, batch):
    # One value per sample (no .mean()): enables paired comparisons with the baseline.
    return (output.argmax(-1) == batch[1]).float()


def nll(output, batch):
    return nn.functional.cross_entropy(output, batch[1])


def test_accuracy_does_not_regress(trained_model, val_loader, metric_baseline):
    # The minimum always applies; the baseline comparison applies once recorded
    # (until then this test still runs, because a threshold is given).
    result = metric_baseline(
        trained_model,
        val_loader,
        {"accuracy": accuracy, "nll": nll},
        # What was evaluated. "checkpoint" is recorded for interpretation, never enforced;
        # the other keys must match the baseline's. Try changing "split" after recording a
        # baseline: the test fails with "context 'split': baseline 'validation', now ...".
        context={"dataset": "synthetic-2d", "split": "validation", "preprocessing": "v1",
                 "checkpoint": "linear-epoch1"},
        tolerance={"accuracy": 0.01, "nll": 0.05},
        higher_is_better={"accuracy": True, "nll": False},
        minimum={"accuracy": 0.95},
    )
    print(result)  # with `pytest -s`: accuracy=... (baseline ...) drop +0.002 (95% CI ...), nll=...


def test_metrics_without_a_baseline(trained_model, val_loader):
    # evaluate() alone: the same loop and aggregation, for your own assertions.
    metrics = evaluate(trained_model, val_loader, {"accuracy": accuracy}, max_batches=4)
    # Try `max_batches=0`: evaluate() raises because the loader produced no batches.
    assert metrics["accuracy"] > 0.9


class ConfusionMatrix:
    """Metrics that need every prediction (F1, AUC, ...) use update()/compute()."""

    def __init__(self, classes=2):
        self.matrix = torch.zeros(classes, classes, dtype=torch.long)

    def update(self, output, batch):
        for truth, pred in zip(batch[1], output.argmax(-1)):
            self.matrix[truth, pred] += 1

    def compute(self):
        tp = self.matrix.diag().float()
        precision = tp / self.matrix.sum(0).clamp_min(1)
        recall = tp / self.matrix.sum(1).clamp_min(1)
        return float((2 * precision * recall / (precision + recall).clamp_min(1e-9)).mean())


def test_macro_f1(trained_model, val_loader):
    assert evaluate(trained_model, val_loader, {"f1": ConfusionMatrix()})["f1"] > 0.9
