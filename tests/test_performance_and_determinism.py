"""Performance, determinism and compile-health checks.

Record baselines once per machine:   pytest examples/test_performance_and_determinism.py --torch-update-perf
Then every normal run compares against them (a missing baseline skips the test).
"""
import pytest
from torch import nn

from pytest_torch import (
    assert_deterministic,
    assert_no_graph_breaks,
    assert_no_recompiles,
    diagnose_nondeterminism,
)


@pytest.fixture
def model(torch_seed, torch_device):
    net = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
    return net.to(torch_device).eval()


def test_latency_and_memory_do_not_regress(model, make_tensor, perf_baseline):
    perf_baseline(model, make_tensor(32, 64), latency_tolerance=0.3)


def test_is_deterministic(model, make_tensor):
    # On failure, run diagnose_nondeterminism(...) to tell RNG usage from kernel variance.
    assert_deterministic(model, make_tensor(32, 64))


def test_diagnosis_is_clean(model, make_tensor):
    assert diagnose_nondeterminism(model, make_tensor(32, 64), runs=3).is_deterministic


def test_compiles_as_one_graph_without_recompiles(model, make_tensor):
    x = make_tensor(32, 64)
    assert_no_graph_breaks(model, x)
    assert_no_recompiles(model, x)


@pytest.mark.torch_gpu_exclusive  # one heavy test at a time per GPU, even under pytest -n auto
def test_big_batch_fits_in_memory(model, make_tensor):
    model(make_tensor(4096, 64))
