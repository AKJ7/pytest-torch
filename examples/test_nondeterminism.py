"""Flaky outputs: find out whether the model, the RNG or the kernels are to blame.

Run: pytest examples/test_nondeterminism.py -v -s   (-s prints the diagnoses)
"""
import pytest
from torch import nn

from pytest_torch import assert_deterministic, check_inference_contract, diagnose_nondeterminism


def build():
    return nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 4))


class MCDropout(nn.Module):
    """Monte-Carlo dropout: sampling at inference is intended, so outputs vary on purpose."""

    def __init__(self):
        super().__init__()
        self.net = build()

    def forward(self, x):
        self.net[2].train()  # keep dropout active for uncertainty estimates
        return self.net(x)


@pytest.fixture
def x(make_tensor):
    return make_tensor(8, 16)


def test_eval_model_is_bit_exact(torch_seed, torch_device, x):
    assert_deterministic(build().to(torch_device).eval(), x, runs=5)


def test_forgetting_eval_is_caught_by_the_contract(torch_seed, torch_device, x):
    report = check_inference_contract(build().to(torch_device), x, raise_on_failure=False)
    print(report)
    failed = {check.name for check in report.failures}
    assert {"eval_mode", "deterministic"} <= failed


def test_mc_dropout_variance_comes_from_the_rng(torch_seed, torch_device, x):
    report = diagnose_nondeterminism(MCDropout().to(torch_device), x, runs=5)
    print(report.verdict)
    assert not report.is_deterministic
    # Reseeding before each run removes the variance: it is RNG-driven, not a kernel issue,
    # so a seeded run is reproducible.
    assert_deterministic(MCDropout().to(torch_device), x, runs=5, reseed=0)
