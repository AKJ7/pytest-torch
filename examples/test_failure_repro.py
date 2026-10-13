"""Failure reproduction bundles: capture what a failing test saw, replay it anywhere.

Run with recording (nothing is written unless a test fails):
    pytest examples/test_failure_repro.py --torch-repro-dir=repro
When a test fails, its report ends with:
    bundle: repro/<test>.pt
    replay: pytest --torch-repro=repro/<test>.pt
The replay runs only that test, with the recorded seed and exactly the recorded inputs,
even on another machine or torch version. Add --torch-device=cpu to replay a GPU failure
on the CPU, and --torch-repro-max-weights-mb=100 to also store model weights.

Try changing `< 0.95` to `< 0.8` in test_scores_are_calibrated, run with --torch-repro-dir,
then replay the bundle: the failure reproduces with the same numbers.
"""
import torch
from torch import nn

from pytest_torch import load_repro, record_repro


def test_scores_are_calibrated(torch_seed, make_tensor):
    model = nn.Sequential(nn.Linear(8, 1), nn.Sigmoid()).eval()
    # Recorded automatically: the seed, the device, and every make_tensor() result.
    # record_repro() adds anything else; on replay it returns the recorded value.
    model = record_repro("model", model)  # weights stored only with --torch-repro-max-weights-mb
    x = make_tensor(256, 8)
    with torch.inference_mode():
        scores = model(x)
    stats = record_repro("stats", {"mean": scores.mean(), "max": scores.max()})
    assert 0.2 < float(stats["mean"]) < 0.8
    assert float(stats["max"]) < 0.95


def test_bundles_can_be_inspected(tmp_path):
    # Bundles are plain torch files (loaded with weights_only=True): open one in a notebook
    # to look at the exact inputs, the failure message and the environment.
    bundle = {
        "format": 1, "nodeid": "t.py::test[cpu]", "seed": 0, "device": "cpu",
        "inputs": [torch.zeros(2)], "recorded": {}, "state_dicts": {}, "failure": "AssertionError",
    }
    torch.save(bundle, tmp_path / "b.pt")
    loaded = load_repro(tmp_path / "b.pt")
    assert loaded["nodeid"] == "t.py::test[cpu]" and torch.equal(loaded["inputs"][0], torch.zeros(2))
