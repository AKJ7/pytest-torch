import json

import pytest

torch = pytest.importorskip("torch")

from pytest_torch import BenchmarkResult, BenchmarkSuite
from pytest_torch.benchmark import format_table

SUITE = """
import time
import torch
from torch import nn
from pytest_torch import BenchmarkSuite

SLEEP = {sleep}


class Slowable(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(16, 4)

    def forward(self, x):
        if SLEEP:
            time.sleep(SLEEP)
        return self.lin(x)


class TestSlowable(BenchmarkSuite):
    batch_sizes = (1, 8)
    dtypes = (torch.float32,)
    modes = ("eager",)
    warmup = 1
    iters = 5
    rounds = 1

    def make_model(self):
        return Slowable()

    def make_inputs(self, batch_size, make_tensor):
        return make_tensor(batch_size, 16)


def test_ordinary():
    pass
"""


@pytest.fixture(autouse=True)
def no_stale_bytecode(monkeypatch):
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")  # the file is rewritten within a second


def run(pytester, *args, sleep=0.0):
    pytester.makepyfile(test_bench=SUITE.format(sleep=sleep))
    return pytester.runpytest_subprocess("-p", "no:cacheprovider", "--torch-device=cpu", "-W", "ignore", "-v", *args)


def test_benchmarks_are_deselected_without_the_flag(pytester):
    result = run(pytester)
    result.assert_outcomes(passed=1, deselected=2)
    assert "pytest-torch benchmarks" not in result.stdout.str()


def test_matrix_report_and_json(pytester):
    result = run(pytester, "--torch-benchmark", "--torch-benchmark-json=out/bench.json")
    result.assert_outcomes(passed=3)  # no baseline yet: measured and reported, not compared
    result.stdout.fnmatch_lines([
        "*TestSlowable::test_benchmark[[]cpu-float32-eager-b1[]] PASSED*",
        "*pytest-torch benchmarks*",
        "*suite*device*dtype*mode*batch*median ms*p90 ms*samples/s*peak MB*compile s*vs baseline*",
        "*TestSlowable*cpu*float32*eager*8*",
        "*benchmark results written to out/bench.json*",
    ])
    rows = json.loads((pytester.path / "out" / "bench.json").read_text())
    assert [(r["batch_size"], r["mode"], r["dtype"]) for r in rows] == [(1, "eager", "float32"), (8, "eager", "float32")]
    assert all(r["median_ms"] > 0 and r["throughput"] > 0 and r["status"] == "measured" for r in rows)


def test_recorded_baseline_then_regression(pytester):
    run(pytester, "--torch-benchmark", "--torch-update-perf").assert_outcomes(passed=3)
    assert (pytester.path / ".torch-perf-baselines.json").exists()
    run(pytester, "--torch-benchmark").assert_outcomes(passed=3)
    slow = run(pytester, "--torch-benchmark", sleep=0.01)  # 10 ms per call: far beyond tolerance
    slow.assert_outcomes(passed=1, failed=2)
    slow.stdout.fnmatch_lines(["*latency regressed*", "*REGRESSED*"])


def test_strict_mode_fails_without_a_baseline(pytester):
    run(pytester, "--torch-benchmark", "--torch-perf-strict").assert_outcomes(passed=1, failed=2)


@pytest.mark.parametrize(
    ("attrs", "message"),
    [
        ({"modes": ("eager", "turbo")}, "unknown mode 'turbo'"),
        ({"batch_sizes": (0,)}, "batch_sizes must be positive integers"),
        ({"batch_sizes": ()}, "batch_sizes must be positive integers"),
    ],
)
def test_invalid_matrices_are_rejected_at_class_definition(attrs, message):
    with pytest.raises(TypeError, match=message):
        type("TestBad", (BenchmarkSuite,), dict(attrs))


def result(**overrides):
    base = {"suite": "S", "device": "cpu", "hardware": "cpu:x", "dtype": "float32", "mode": "eager",
            "batch_size": 8, "median_ms": 1.0, "p90_ms": 1.1, "min_ms": 0.9, "throughput": 8000.0,
            "peak_memory_mb": None, "compile_s": None}
    return BenchmarkResult(**{**base, **overrides})


def test_report_formatting():
    lines = format_table([
        result(baseline_ms=0.5, status="regressed"),
        result(mode="compile", compile_s=1.234, baseline_ms=400.0),
        result(status="baseline updated", noisy=True, p90_ms=9.0),
    ])
    assert "+100.0% REGRESSED" in lines[1]
    assert "1.23" in lines[2] and "-99.8%" in lines[2]
    assert "recorded (noisy)" in lines[3]
