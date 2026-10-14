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

CUTOFF = {cutoff}        # the model's decision boundary: 0.0 is perfect, larger is worse
SAMPLES = {samples}
DATA_SEED = {data_seed}


def accuracy(output, batch):
    return (output.argmax(-1) == batch[1]).float(){reduce}


def test_accuracy(metric_baseline):
    g = torch.Generator().manual_seed(DATA_SEED)
    x = torch.randn(SAMPLES, 4, generator=g)
    y = (x[:, 0] > 0).long()  # fixed labels: only the model changes between runs
    model = nn.Linear(4, 2)
    with torch.no_grad():
        model.weight.zero_(); model.bias.zero_()
        model.weight[1, 0] = 1.0
        model.bias[1] = -CUTOFF
    loader = [(x[i:i + 16], y[i:i + 16]) for i in range(0, SAMPLES, 16)]
    metric_baseline(model, loader, {{"accuracy": accuracy}}, {options})
"""


@pytest.fixture(autouse=True)
def no_stale_bytecode(monkeypatch):
    # The tests below rewrite test_metric.py with same-length content within one second; a
    # cached .pyc (validated by mtime and size) would then run the previous version.
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")


def run(pytester, cutoff=0.0, options="", *args, samples=64, data_seed=0, per_sample=False):
    pytester.makepyfile(test_metric=BASELINE_TEST.format(
        cutoff=cutoff, options=options, samples=samples, data_seed=data_seed,
        reduce="" if per_sample else ".mean()"))
    return pytester.runpytest_subprocess("-p", "no:cacheprovider", "-rs", *args)


def test_missing_baseline_skips_then_records_then_compares(pytester):
    run(pytester).assert_outcomes(skipped=1)
    run(pytester, 0.0, "", "--torch-metrics-strict").assert_outcomes(failed=1)
    run(pytester, 0.0, "", "--torch-update-metrics").assert_outcomes(passed=1)
    assert (pytester.path / ".torch-metric-baselines.json").exists()
    run(pytester).assert_outcomes(passed=1)
    worse = run(pytester, 0.8)  # a worse model on the same data
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


# --------------------------------------------------------------------------- #
# Provenance: a baseline is only compared with an evaluation of the same thing
# --------------------------------------------------------------------------- #
def test_context_mismatch_fails_and_names_the_field(pytester):
    record = 'context={"dataset": "toy", "split": "validation", "preprocessing": "v1"}'
    run(pytester, 0.0, record, "--torch-update-metrics").assert_outcomes(passed=1)
    run(pytester, 0.0, record).assert_outcomes(passed=1)
    other = run(pytester, 0.0, 'context={"dataset": "toy", "split": "test", "preprocessing": "v1"}')
    other.assert_outcomes(failed=1)
    other.stdout.fnmatch_lines(["*the baseline measured something else*", "*context 'split': baseline 'validation', now 'test'*"])


def test_changed_data_or_sample_count_is_detected(pytester):
    run(pytester, 0.0, "", "--torch-update-metrics").assert_outcomes(passed=1)
    changed = run(pytester, 0.0, "", data_seed=1)  # same size, different samples
    changed.assert_outcomes(failed=1)
    changed.stdout.fnmatch_lines(["*input fingerprint: baseline *, now * (the data, its order or its preprocessing changed)*"])
    fewer = run(pytester, 0.0, "", samples=48)
    fewer.assert_outcomes(failed=1)
    fewer.stdout.fnmatch_lines(["*sample count: baseline 64, now 48*"])


def test_checkpoint_is_recorded_but_not_enforced(pytester):
    run(pytester, 0.0, 'context={"checkpoint": "v1"}', "--torch-update-metrics").assert_outcomes(passed=1)
    run(pytester, 0.0, 'context={"checkpoint": "v2"}').assert_outcomes(passed=1)  # a new checkpoint: compared
    worse = run(pytester, 0.8, 'context={"checkpoint": "v2"}')
    worse.assert_outcomes(failed=1)
    worse.stdout.fnmatch_lines(["*baseline: v1 / *, torch *, recorded *; now: v2 / *"])


def test_baseline_file_records_provenance(pytester):
    import json

    run(pytester, 0.0, 'context={"dataset": "toy", "checkpoint": "v1"}', "--torch-update-metrics",
        per_sample=True).assert_outcomes(passed=1)
    entry = json.loads((pytester.path / ".torch-metric-baselines.json").read_text())["entries"]["test_metric.py::test_accuracy"]
    prov = entry["provenance"]
    assert prov["context"] == {"dataset": "toy"}  # checkpoint is recorded separately
    assert prov["checkpoint"]["label"] == "v1" and len(prov["checkpoint"]["fingerprint"]) == 16
    assert prov["samples"] == 64 and prov["batches"] == 4 and prov["max_batches"] is None
    assert prov["metrics"]["accuracy"]["kind"] == "per-sample"
    assert (pytester.path / entry["per_sample_file"]).is_file()


# --------------------------------------------------------------------------- #
# Paired comparisons with confidence intervals
# --------------------------------------------------------------------------- #
from pytest_torch import paired_difference
from pytest_torch.evaluation import Evaluation, check_metrics
from pytest_torch.perf import PerfStore


def correctness(n, wrong):
    values = torch.ones(n)
    values[:wrong] = 0
    return values


def evaluation(per_sample, provenance=None):
    prov = provenance or {"context": {}, "samples": per_sample.numel(), "max_batches": None,
                          "inputs_fingerprint": "same", "metrics": {"accuracy": {"kind": "per-sample", "config": None}}}
    return Evaluation({"accuracy": float(per_sample.mean())}, {"accuracy": per_sample}, prov)


def test_paired_difference_is_deterministic_and_brackets_the_drop():
    base, current = correctness(400, 0), correctness(400, 40)
    first = paired_difference(current, base)
    assert first == paired_difference(current, base)  # seeded bootstrap
    assert first.drop == pytest.approx(0.1) and first.low < 0.1 < first.high and first.low > 0
    improved = paired_difference(base, current)
    assert improved.drop == pytest.approx(-0.1) and improved.high < 0


def test_small_dataset_noise_is_not_a_regression(tmp_path):
    store = PerfStore(tmp_path / "m.json")
    check_metrics(store, "k", evaluation(correctness(40, 0)), update=True)
    # 2 of 40 samples flipped: a 5% drop, above the 1% tolerance, but within noise.
    result = check_metrics(store, "k", evaluation(correctness(40, 2)))
    paired = result.paired["accuracy"]
    assert paired.drop == pytest.approx(0.05) and paired.low <= 0


def test_same_drop_on_a_large_dataset_is_a_regression(tmp_path):
    store = PerfStore(tmp_path / "m.json")
    check_metrics(store, "k", evaluation(correctness(4000, 0)), update=True)
    with pytest.raises(AssertionError, match=r"accuracy regressed from 1 to 0\.95: drop \+0\.05 \(95% CI \+0\.0\d+"):
        check_metrics(store, "k", evaluation(correctness(4000, 200)))


def test_significant_but_tiny_drop_within_tolerance_passes(tmp_path):
    store = PerfStore(tmp_path / "m.json")
    check_metrics(store, "k", evaluation(correctness(100_000, 0)), update=True)
    result = check_metrics(store, "k", evaluation(correctness(100_000, 500)))  # 0.5% < 1% tolerance
    assert result.paired["accuracy"].low > 0  # real, but too small to matter


def test_metric_config_change_blocks_comparison(tmp_path):
    store = PerfStore(tmp_path / "m.json")
    prov = {"context": {}, "samples": 10, "max_batches": None, "inputs_fingerprint": "x",
            "metrics": {"f1": {"kind": "update/compute", "config": {"average": "macro"}}}}
    check_metrics(store, "k", Evaluation({"f1": 0.8}, {}, prov), update=True)
    changed = {**prov, "metrics": {"f1": {"kind": "update/compute", "config": {"average": "micro"}}}}
    with pytest.raises(AssertionError, match=r"metric 'f1' config: baseline \{'average': 'macro'\}, now \{'average': 'micro'\}"):
        check_metrics(store, "k", Evaluation({"f1": 0.8}, {}, changed))


def test_evaluate_detailed_records_config_and_per_sample_values():
    from pytest_torch import evaluate_detailed

    class MacroF1:
        config = {"average": "macro", "ignore_index": -100}

        def __init__(self):
            self.n = 0

        def update(self, output, batch):
            self.n += 1

        def compute(self):
            return 0.5

    x, y = torch.randn(10, 4), torch.arange(10) % 2
    result = evaluate_detailed(nn.Linear(4, 2), [(x[:6], y[:6]), (x[6:], y[6:])],
                               {"acc": lambda o, b: (o.argmax(-1) == b[1]).float(), "f1": MacroF1()},
                               context={"split": "dev", "checkpoint": "ckpt-7"})
    assert result.per_sample["acc"].shape == (10,)
    f1 = result.provenance["metrics"]["f1"]
    assert f1["kind"] == "update/compute" and f1["config"] == MacroF1.config
    assert f1["function"].endswith("MacroF1")  # informational only, never enforced
    assert result.provenance["context"] == {"split": "dev"}
    assert result.provenance["checkpoint"]["label"] == "ckpt-7"
    # the same samples, batched differently, have the same fingerprint
    rebatched = evaluate_detailed(nn.Linear(4, 2), [(x[:3], y[:3]), (x[3:], y[3:])], {"acc": lambda o, b: (o.argmax(-1) == b[1]).float()})
    assert rebatched.provenance["inputs_fingerprint"] == result.provenance["inputs_fingerprint"]


def test_per_sample_metric_must_match_the_batch():
    from pytest_torch import evaluate_detailed

    with pytest.raises(ValueError, match="returned 3 values for a batch of 4"):
        evaluate_detailed(nn.Linear(4, 2), [(torch.randn(4, 4), torch.zeros(4))], {"bad": lambda o, b: torch.ones(3)})
