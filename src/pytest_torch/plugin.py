"""pytest-torch plugin: options, markers, fixtures.

Keep module-level imports torch-free; import torch lazily inside hooks/fixtures
so projects that don't use torch pay nothing at collection time.
"""
from __future__ import annotations

import importlib.metadata
import statistics
import time
import zlib

import pytest

from .golden import Golden


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
        "--torch-seed", type=int, default=0, help="Base seed used by `torch_seed`."
    )
    group.addoption(
        "--torch-seed-mode",
        choices=("fixed", "per-test"),
        default="fixed",
        help="'fixed': every test gets --torch-seed. 'per-test': a stable seed derived "
        "from --torch-seed and the test id, so tests are independent but reproducible.",
    )
    group.addoption(
        "--torch-autoseed",
        action="store_true",
        help="Seed before every test, even if it does not request `torch_seed`.",
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
    config.addinivalue_line(
        "markers", "torch_seed(n): seed used by `torch_seed` for this test"
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
# Seeding
# --------------------------------------------------------------------------- #
def _resolve_seed(request) -> int:
    marker = request.node.get_closest_marker("torch_seed")
    if marker and marker.args:
        return int(marker.args[0])
    config = request.config
    seed = config.getoption("torch_seed")
    if config.getoption("torch_seed_mode") == "per-test":
        # Derive from the test id *without* the parametrization suffix, so every
        # device (and every param) of one test sees identical inputs.
        base_id = request.node.nodeid.split("[")[0]
        seed = (seed + zlib.crc32(base_id.encode())) % (2**31)
    return seed


@pytest.fixture
def torch_seed(request):
    """Seed ``random``, ``numpy`` and ``torch``; yields the seed that was used.

    Priority: ``@pytest.mark.torch_seed(n)`` > ``--torch-seed-mode=per-test`` >
    ``--torch-seed``. The seed is shown in the report of a failing test.
    """
    from .utils import deterministic_algorithms, seed_everything

    seed = _resolve_seed(request)
    seed_everything(seed)
    request.node.add_report_section("call", "torch-seed", f"torch seed: {seed}")
    if request.config.getoption("torch_deterministic"):
        with deterministic_algorithms(True):
            yield seed
    else:
        yield seed


@pytest.fixture(autouse=True)
def _torch_autoseed(request):
    if request.config.getoption("torch_autoseed"):
        request.getfixturevalue("torch_seed")


@pytest.fixture
def torch_generator(torch_seed):
    """A seeded **CPU** ``torch.Generator``.

    It lives on the CPU on purpose: the same seed then yields identical values no
    matter which device the test runs on, so cross-device comparisons are fair.
    """
    import torch

    return torch.Generator().manual_seed(torch_seed)


@pytest.fixture
def make_tensor(torch_generator, torch_device):
    """Factory for reproducible tensors, generated on the CPU and moved to ``torch_device``.

    ``make_tensor(2, 3)``                          -> normal(0, 1), float32
    ``make_tensor(2, 3, kind="uniform", low=0, high=1)``
    ``make_tensor(4, 8, dtype=torch.long, low=1, high=50)``   -> integers in [1, 50)
    """
    import torch

    def make(*shape, dtype=None, kind=None, low=None, high=None):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = tuple(shape[0])
        dtype = torch.float32 if dtype is None else dtype
        is_int = not (dtype.is_floating_point or dtype.is_complex)
        kind = kind or ("randint" if is_int else "normal")
        if kind == "normal":
            data = torch.randn(shape, generator=torch_generator, dtype=torch.float64)
        elif kind == "uniform":
            lo = -1.0 if low is None else float(low)
            hi = 1.0 if high is None else float(high)
            data = lo + (hi - lo) * torch.rand(
                shape, generator=torch_generator, dtype=torch.float64
            )
        elif kind == "randint":
            lo = 0 if low is None else int(low)
            hi = (2 if dtype == torch.bool else 100) if high is None else int(high)
            data = torch.randint(lo, hi, shape, generator=torch_generator)
        else:
            raise ValueError(f"unknown kind {kind!r}; use normal, uniform or randint")
        return data.to(device=torch_device, dtype=dtype)

    return make


# --------------------------------------------------------------------------- #
# Inference context
# --------------------------------------------------------------------------- #
@pytest.fixture
def inference_mode():
    """Run the test body under ``torch.inference_mode()``."""
    import torch

    with torch.inference_mode():
        yield


# --------------------------------------------------------------------------- #
# Model zoo (Hugging Face / timm / torchvision), cached per session
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def model_loader():
    """``ModelLoader`` with a per-session cache: ``model_loader.hf(...)``, ``.timm(...)``,
    ``.torchvision(...)``. Cached models are shared between tests, so don't mutate them."""
    from .integrations import ModelLoader

    return ModelLoader()


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
