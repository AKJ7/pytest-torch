"""pytest-torch plugin: options, markers, fixtures, hooks.

Keep module-level imports torch-free; import torch lazily inside hooks/fixtures
so projects that don't use torch pay nothing at collection time.
"""
from __future__ import annotations

import importlib.metadata
import zlib
from pathlib import Path

import pytest

from . import report as _artifacts
from .golden import Golden


# --------------------------------------------------------------------------- #
# Options, markers, hooks
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
    # performance baselines
    group.addoption(
        "--torch-update-perf",
        action="store_true",
        help="Record performance baselines (`perf_baseline`) instead of comparing.",
    )
    group.addoption(
        "--torch-perf-strict",
        action="store_true",
        help="Fail (instead of skip) when a performance baseline is missing.",
    )
    group.addoption(
        "--torch-perf-file",
        default=None,
        help="Baseline file (default: <rootdir>/.torch-perf-baselines.json).",
    )
    # pytest-xdist
    group.addoption(
        "--torch-xdist-gpus",
        default="auto",
        help="Under pytest-xdist: spread workers over these GPUs. 'auto' (all), 'none', "
        "or a comma-separated list such as 0,1.",
    )
    group.addoption(
        "--torch-gpu-memory-fraction",
        default="none",
        help="Per-process CUDA memory cap under xdist: 'none', 'auto' (split each GPU "
        "between the workers sharing it) or a float in (0, 1].",
    )
    # failure output
    group.addoption(
        "--torch-failure-artifacts",
        default=None,
        metavar="DIR",
        help="Save a heatmap PNG for every tensor mismatch into DIR (needs matplotlib).",
    )
    group.addoption(
        "--torch-github-annotations",
        action="store_true",
        help="Print GitHub Actions ::error annotations for failing tests.",
    )


class _GithubAnnotations:
    def __init__(self, config):
        self.config = config

    def pytest_runtest_logreport(self, report):
        if not report.failed or report.when not in ("setup", "call"):
            return
        path, lineno, _ = report.location
        crash = getattr(report.longrepr, "reprcrash", None)
        message = crash.message if crash is not None else str(report.longrepr)
        command = _artifacts.github_annotation(
            title=report.nodeid,
            message=message[:2000],
            file=path,
            line=(lineno + 1) if lineno is not None else None,
        )
        terminal = self.config.pluginmanager.getplugin("terminalreporter")
        if terminal is not None:
            terminal.ensure_newline()
            terminal.write_line(command)
        else:
            print(command)


def pytest_configure(config):
    for line in (
        "torch_golden(name): name of the golden file used by `golden`",
        "torch_max_latency(ms): median latency budget enforced by `latency`",
        "torch_seed(n): seed used by `torch_seed` for this test",
        "torch_gpu_exclusive(timeout=1800): serialize this test with other exclusive "
        "tests on the same GPU (useful under pytest-xdist)",
    ):
        config.addinivalue_line("markers", line)
    _artifacts.configure(config.getoption("torch_failure_artifacts"))
    # Workers forward their reports to the controller, which prints the annotations.
    if config.getoption("torch_github_annotations") and not hasattr(config, "workerinput"):
        config.pluginmanager.register(_GithubAnnotations(config), "torch-github-annotations")


def pytest_runtest_setup(item):
    _artifacts.set_current_test(item.nodeid)


def pytest_runtest_teardown(item, nextitem):
    _artifacts.set_current_test(None)


def pytest_report_header(config):
    try:
        return f"pytest-torch: torch {importlib.metadata.version('torch')}"
    except importlib.metadata.PackageNotFoundError:
        return "pytest-torch: torch not installed"


# --------------------------------------------------------------------------- #
# Device handling (+ GPU assignment under pytest-xdist)
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


def _assigned_gpu(config) -> int | None:
    """GPU index for the current xdist worker (``None`` outside xdist workers)."""
    from .parallel import assign_gpu, parse_gpu_option, worker_index

    worker = worker_index()
    if worker is None:
        return None
    import torch

    gpus = parse_gpu_option(config.getoption("torch_xdist_gpus"), torch.cuda.device_count)
    return assign_gpu(worker, gpus)


_configured_gpus: set[int] = set()


def _apply_memory_fraction(config, gpu: int) -> None:
    import torch

    from .parallel import memory_fraction, parse_gpu_option, worker_count, worker_index

    option = str(config.getoption("torch_gpu_memory_fraction")).strip().lower()
    if option in ("", "none", "off"):
        return
    if option == "auto":
        gpus = parse_gpu_option(config.getoption("torch_xdist_gpus"), torch.cuda.device_count)
        fraction = memory_fraction(worker_index(), worker_count(), gpus)
    else:
        fraction = float(option)
    if fraction is not None:
        torch.cuda.set_per_process_memory_fraction(fraction, gpu)


def _assign_worker_gpu(config, device: str) -> str:
    """Map plain ``"cuda"`` to this xdist worker's GPU (test ids stay ``cuda``)."""
    if device != "cuda":
        return device
    gpu = _assigned_gpu(config)
    if gpu is None:
        return device
    import torch

    torch.cuda.set_device(gpu)
    if gpu not in _configured_gpus:
        _configured_gpus.add(gpu)
        _apply_memory_fraction(config, gpu)
    return f"cuda:{gpu}"


def pytest_generate_tests(metafunc):
    if "torch_device" in metafunc.fixturenames:
        metafunc.parametrize("torch_device", _devices(metafunc.config), indirect=True)


@pytest.fixture
def torch_device(request) -> str:
    device = request.param
    if not _device_available(device):
        pytest.skip(f"{device} is not available")
    return _assign_worker_gpu(request.config, device)


@pytest.fixture(autouse=True)
def _torch_gpu_exclusive(request):
    """``@pytest.mark.torch_gpu_exclusive``: one such test at a time per GPU, across xdist workers."""
    marker = request.node.get_closest_marker("torch_gpu_exclusive")
    if marker is None:
        yield
        return
    from .parallel import FileLock

    factory = request.getfixturevalue("tmp_path_factory")
    gpu = _assigned_gpu(request.config)
    name = f"pytest-torch-gpu-{'any' if gpu is None else gpu}.lock"
    # basetemp's parent is shared by all xdist workers of one session.
    lock = FileLock(
        Path(factory.getbasetemp()).parent / name,
        timeout=float(marker.kwargs.get("timeout", 1800.0)),
    )
    with lock:
        yield


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
# Inference context, model zoo
# --------------------------------------------------------------------------- #
@pytest.fixture
def inference_mode():
    """Run the test body under ``torch.inference_mode()``."""
    import torch

    with torch.inference_mode():
        yield


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
# Latency and performance baselines
# --------------------------------------------------------------------------- #
@pytest.fixture
def latency(request, torch_device):
    """Return ``measure(fn, *args, warmup=5, iters=20)`` -> median latency in ms.

    Handles warmup and device synchronization. If the test carries
    ``@pytest.mark.torch_max_latency(ms=...)`` the budget is enforced.
    """
    from .perf import measure_latency

    marker = request.node.get_closest_marker("torch_max_latency")
    budget = marker.kwargs.get("ms") if marker else None

    def measure(fn, *args, warmup: int = 5, iters: int = 20, **kwargs) -> float:
        stats = measure_latency(
            fn, *args, device=torch_device, warmup=warmup, iters=iters, **kwargs
        )
        if budget is not None and stats.median_ms > budget:
            pytest.fail(
                f"median latency {stats.median_ms:.2f} ms exceeds budget {budget} ms "
                f"(min {stats.min_ms:.2f}, max {stats.max_ms:.2f}, n={iters})",
                pytrace=False,
            )
        return stats.median_ms

    return measure


@pytest.fixture
def perf_baseline(request, torch_device):
    """Return ``check(fn, *args, name=None, **options)``: measure latency (and CUDA peak
    memory) and compare with the recorded baseline for this machine.

    Record baselines with ``--torch-update-perf``. Options: ``latency_tolerance``,
    ``latency_slack_ms``, ``memory_tolerance``, ``memory_slack_mb``, ``warmup``, ``iters``.
    Wrap functions that need keyword arguments in a lambda.
    """
    from .perf import PerfStore, check_perf

    config = request.config
    path = config.getoption("torch_perf_file") or Path(config.rootpath) / ".torch-perf-baselines.json"
    store = PerfStore(path)
    update = config.getoption("torch_update_perf")
    strict = config.getoption("torch_perf_strict")
    default_name = request.node.nodeid.split("[")[0]

    def check(fn, *args, name=None, **options):
        return check_perf(
            store, name or default_name, fn, *args,
            device=torch_device, update=update, strict=strict, **options,
        )

    return check
