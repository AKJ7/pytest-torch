import pytest

torch = pytest.importorskip("torch")
from torch import nn

from pytest_torch import evaluate


def accuracy(output, batch):
    return (output.argmax(-1) == batch[1]).float().mean()


@pytest.fixture
def loader():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(10, 4, generator=g)
    y = torch.arange(10) % 2
    # uneven batches: 4 + 4 + 2, so a plain mean of batch means would be wrong
    return [(x[i:i + 4], y[i:i + 4]) for i in range(0, 10, 4)]


class Oracle(nn.Module):
    """Predicts class 0 for the first 7 samples (rows are matched by value)."""

    def __init__(self, x_all):
        super().__init__()
        self.register_buffer("x_all", x_all)

    def forward(self, x):
        index = (x[:, None, :] == self.x_all[None]).all(-1).float().argmax(-1)
        return torch.stack([(index < 7).float(), (index >= 7).float()], -1)


def test_metrics_are_weighted_by_batch_size(loader):
    x_all = torch.cat([b[0] for b in loader])
    y_all = torch.cat([b[1] for b in loader])
    model = Oracle(x_all)
    expected = float((model(x_all).argmax(-1) == y_all).float().mean())
    assert evaluate(model, loader, {"accuracy": accuracy})["accuracy"] == pytest.approx(expected)


def test_update_compute_metrics_and_max_batches(loader):
    class Count:
        def __init__(self):
            self.n = 0

        def update(self, output, batch):
            self.n += output.shape[0]

        def compute(self):
            return self.n

    model = nn.Linear(4, 2)
    assert evaluate(model, loader, {"n": Count()})["n"] == 10
    assert evaluate(model, loader, {"n": Count()}, max_batches=2)["n"] == 8


def test_training_mode_is_restored_and_dict_batches(loader):
    class Kw(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 2)

        def forward(self, features):
            return self.lin(features)

    model = Kw().train()
    batches = [{"features": x, "labels": y} for x, y in loader]
    out = evaluate(
        model, batches, {"acc": lambda o, b: (o.argmax(-1) == b["labels"]).float().mean()},
        inputs=lambda b: {"features": b["features"]},
    )
    assert 0 <= out["acc"] <= 1
    assert model.training


def test_empty_loader_and_no_metrics():
    with pytest.raises(ValueError, match="no batches"):
        evaluate(nn.Linear(4, 2), [], {"acc": accuracy})
    with pytest.raises(ValueError, match="at least one metric"):
        evaluate(nn.Linear(4, 2), [], {})


BASELINE_TEST = """
import torch
from torch import nn

SHIFT = {shift}

def accuracy(output, batch):
    return (output.argmax(-1) == batch[1]).float().mean()

def test_accuracy(metric_baseline):
    torch.manual_seed(0)
    x = torch.randn(64, 4)
    y = (x[:, 0] > SHIFT).long()
    model = nn.Linear(4, 2)
    with torch.no_grad():
        model.weight.zero_(); model.bias.zero_()
        model.weight[1, 0] = 1.0
    loader = [(x[i:i + 16], y[i:i + 16]) for i in range(0, 64, 16)]
    metric_baseline(model, loader, {{"accuracy": accuracy}}, {options})
"""


def run(pytester, shift=0.0, options="", *args):
    pytester.makepyfile(test_metric=BASELINE_TEST.format(shift=shift, options=options))
    return pytester.runpytest_subprocess("-p", "no:cacheprovider", "-rs", *args)


def test_missing_baseline_skips_then_records_then_compares(pytester):
    run(pytester).assert_outcomes(skipped=1)
    run(pytester, 0.0, "", "--torch-metrics-strict").assert_outcomes(failed=1)
    run(pytester, 0.0, "", "--torch-update-metrics").assert_outcomes(passed=1)
    assert (pytester.path / ".torch-metric-baselines.json").exists()
    run(pytester).assert_outcomes(passed=1)
    worse = run(pytester, 0.8)  # labels shift: the same model gets worse
    worse.assert_outcomes(failed=1)
    worse.stdout.fnmatch_lines(["*accuracy dropped from 1 to *allowed 0.01*", "*--torch-update-metrics*"])
    run(pytester, 0.8, "tolerance=0.5").assert_outcomes(passed=1)


def test_thresholds_apply_without_a_baseline(pytester):
    run(pytester, 0.0, 'minimum={"accuracy": 0.9}').assert_outcomes(passed=1)
    result = run(pytester, 0.8, 'minimum={"accuracy": 0.9}')
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*accuracy = * is below the minimum 0.9*"])


def test_lower_is_better(pytester):
    body = 'higher_is_better=False, maximum={"accuracy": 2.0}'
    run(pytester, 0.8, body, "--torch-update-metrics").assert_outcomes(passed=1)
    run(pytester, 0.0, body).assert_outcomes(failed=1)  # "accuracy" rose: worse when lower is better
