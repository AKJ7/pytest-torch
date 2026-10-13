"""pytest-torch plugin: options, markers, fixtures.

Keep module-level imports torch-free; import torch lazily inside hooks/fixtures
so projects that don't use torch pay nothing at collection time.
"""
from __future__ import annotations

import importlib.metadata
import random
import statistics
import time

import pytest

# from src.pytest_torch.golden import Golden
from pytest_torch.golden import Golden


# --------------------------------------------------------------------------- #
# Options, markers, report header
# --------------------------------------------------------------------------- #
def pytest_addoption(parser):
    group = parser.getgroup("torch", "pytest-torch")
    group.addoption(
        "--torch-device",
        action="append",
        dest="torch_devices",
        default=None,
        help="Device(s) for the `torch_device` fixture (repeatable). "
        "Default: all available (cpu/cuda/mps).",
    )
    group.addoption(
        "--torch-seed", type=int, default=0, help="Seed used by `torch_seed`."
    )
    group.addoption(
        "--torch-deterministic",
        action="store_true",
        help="Enable torch.use_deterministic_algorithms in `torch_seed`.",
    )
    group.addoption(
        "--torch-update-golden",
        action="store_true",
        help="Rewrite golden files instead of comparing. "
        "Use with --torch-device=cpu for a portable reference.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "torch_golden(name): name of the golden file used by `golden`"
    )
    config.addinivalue_line(
        "markers", "torch_max_latency(ms): median latency budget enforced by `latency`"
    )


def pytest_report_header(config):
    try:
        return f"pytest-torch: torch {importlib.metadata.version('torch')}"
    except importlib.metadata.PackageNotFoundError:
        return "pytest-torch: torch not installed"


# --------------------------------------------------------------------------- #
# Device handling
# --------------------------------------------------------------------------- #
def _mps_available() -> bool:
    import torch

    mps = getattr(torch.backends, "mps", None)
    return mps is not None and mps.is_available()


def _device_available(device: str) -> bool:
    import torch

    kind = device.split(":")[0]
    if kind == "cpu":
        return True
    if kind == "cuda":
        return torch.cuda.is_available()
    if kind == "mps":
        return _mps_available()
    return False


def _devices(config) -> list[str]:
    requested = config.getoption("torch_devices")
    if requested:
        return requested
    import torch

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if _mps_available():
        devices.append("mps")
    return devices


def _sync(device: str) -> None:
    import torch

    kind = device.split(":")[0]
    if kind == "cuda":
        torch.cuda.synchronize(device)
    elif kind == "mps":
        torch.mps.synchronize()


def pytest_generate_tests(metafunc):
    if "torch_device" in metafunc.fixturenames:
        metafunc.parametrize("torch_device", _devices(metafunc.config), indirect=True)


@pytest.fixture
def torch_device(request) -> str:
    device = request.param
    if not _device_available(device):
        pytest.skip(f"{device} is not available")
    return device


# --------------------------------------------------------------------------- #
# Reproducibility / inference context
# --------------------------------------------------------------------------- #
@pytest.fixture
def torch_seed(request):
    import torch

    seed = request.config.getoption("torch_seed")
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if request.config.getoption("torch_deterministic"):
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        yield seed
        torch.use_deterministic_algorithms(previous)
    else:
        yield seed


@pytest.fixture
def inference_mode():
    """Run the test body under ``torch.inference_mode()``."""
    import torch

    with torch.inference_mode():
        yield


# --------------------------------------------------------------------------- #
# Golden files
# --------------------------------------------------------------------------- #
@pytest.fixture
def golden(request) -> Golden:
    marker = request.node.get_closest_marker("torch_golden")
    if marker and marker.args:
        name = marker.args[0]
    else:
        name = getattr(request.node, "originalname", request.node.name)
    path = request.node.path.parent / "__golden__" / f"{name}.pt"
    return Golden(path, update=request.config.getoption("torch_update_golden"))


# --------------------------------------------------------------------------- #
# Latency
# --------------------------------------------------------------------------- #
@pytest.fixture
def latency(request, torch_device):
    """Return ``measure(fn, *args, warmup=5, iters=20)`` -> median latency in ms.

    Handles warmup and device synchronization. If the test carries
    ``@pytest.mark.torch_max_latency(ms=...)`` the budget is enforced.
    """
    import torch

    marker = request.node.get_closest_marker("torch_max_latency")
    budget = marker.kwargs.get("ms") if marker else None

    def measure(fn, *args, warmup: int = 5, iters: int = 20, **kwargs) -> float:
        times = []
        with torch.inference_mode():
            for _ in range(warmup):
                fn(*args, **kwargs)
            _sync(torch_device)
            for _ in range(iters):
                start = time.perf_counter()
                fn(*args, **kwargs)
                _sync(torch_device)
                times.append((time.perf_counter() - start) * 1e3)
        median = statistics.median(times)
        if budget is not None and median > budget:
            pytest.fail(
                f"median latency {median:.2f} ms exceeds budget {budget} ms "
                f"(min {min(times):.2f}, max {max(times):.2f}, n={iters})",
                pytrace=False,
            )
        return median

    return measure
