"""pytest-torch plugin: options, markers, fixtures, hooks.

Keep module-level imports torch-free; import torch lazily inside hooks/fixtures
so projects that don't use torch pay nothing at collection time.
"""
from __future__ import annotations

import importlib.metadata
import zlib
from pathlib import Path

import pytest

from . import report as _artifacts, repro as _repro, tolerances as _tolerances
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
        "Default: all available (cpu/cuda/xpu/mps).",
    )
    group.addoption(
        "--torch-dtype",
        action="append",
        dest="torch_dtypes",
        default=None,
        help="Dtype(s) for the `torch_dtype` fixture (repeatable), e.g. float32, bfloat16. "
        "Default: float32; @pytest.mark.torch_dtypes(...) overrides per test.",
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
    # benchmarks
    group.addoption(
        "--torch-benchmark",
        action="store_true",
        help="Run benchmarks (BenchmarkSuite and @pytest.mark.torch_benchmark tests); "
        "they are deselected otherwise.",
    )
    group.addoption(
        "--torch-benchmark-json",
        default=None,
        metavar="PATH",
        help="Write the benchmark results of the run to PATH as JSON.",
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
    # dataset metric baselines
    group.addoption(
        "--torch-update-metrics",
        action="store_true",
        help="Record metric baselines (`metric_baseline`) instead of comparing.",
    )
    group.addoption(
        "--torch-metrics-strict",
        action="store_true",
        help="Fail (instead of skip) when a metric baseline is missing.",
    )
    group.addoption(
        "--torch-metrics-file",
        default=None,
        help="Metric baseline file (default: <rootdir>/.torch-metric-baselines.json).",
    )
    # pytest-xdist
    group.addoption(
        "--torch-xdist-gpus",
        default="auto",
        help="Under pytest-xdist: spread workers over these CUDA or XPU GPUs. 'auto' (all), "
        "'none', or a comma-separated list such as 0,1.",
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
    # zero-code testing
    group.addoption(
        "--torch-model",
        action="append",
        dest="torch_models",
        default=None,
        metavar="SPEC",
        help="Test a model with no test file (repeatable): hf:NAME[@AutoClass], timm:NAME, "
        "torchvision:NAME, package.module:Factory or path/to/file.py:Factory.",
    )
    group.addoption(
        "--torch-model-checks",
        default=None,
        help="Comma-separated checks for --torch-model: contract, invariance, checkpoint, "
        "gradients (the default), export, compile.",
    )
    group.addoption(
        "--torch-model-inputs",
        default=None,
        metavar="INPUTS",
        help="Inputs for --torch-model when they can't be inferred: e.g. 4x3x32x32, or "
        "input_ids=long:4x16:1000,attention_mask=mask:4x16.",
    )
    group.addoption(
        "--torch-model-pretrained",
        action="store_true",
        help="Load pretrained weights for --torch-model (downloads them); default: random init.",
    )
    # failure reproduction bundles
    group.addoption(
        "--torch-repro-dir",
        default=None,
        metavar="DIR",
        help="Write a reproduction bundle (seed, device, inputs, environment) for every "
        "failing test into DIR.",
    )
    group.addoption(
        "--torch-repro",
        default=None,
        metavar="BUNDLE",
        help="Replay a reproduction bundle: run only its test, with its seed and inputs. "
        "Add --torch-device to replay on another device.",
    )
    group.addoption(
        "--torch-repro-max-weights-mb",
        type=float,
        default=0.0,
        help="Also store weights passed to record_repro() up to this size (default 0: never).",
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


_SESSION_FIELDS = ("selected_devices", "model_specs", "model_checks", "model_inputs",
                   "model_pretrained", "zero_code_only")


def _save_global_state() -> dict:
    """Module-level state this plugin sets per session. A nested session (pytester running
    in-process) must not leave its settings behind for the outer one."""
    from . import _session

    return {
        "session": {name: getattr(_session, name) for name in _SESSION_FIELDS},
        "tolerances": dict(_tolerances._config),
        "repro": dict(_repro._state),
        "artifacts": dict(_artifacts._state),
        "benchmarks": list(_benchmark_results()),
    }


def _restore_global_state(saved: dict) -> None:
    from . import _session

    for name, value in saved["session"].items():
        setattr(_session, name, value)
    _tolerances.configure(saved["tolerances"])
    _repro._state.clear()
    _repro._state.update(saved["repro"])
    _artifacts._state.clear()
    _artifacts._state.update(saved["artifacts"])
    _benchmark_results()[:] = saved["benchmarks"]


def _benchmark_results() -> list:
    from . import benchmark

    return benchmark.results


def pytest_unconfigure(config):
    saved = getattr(config, "_pytest_torch_saved_state", None)
    if saved is not None:
        _restore_global_state(saved)


def pytest_configure(config):
    config._pytest_torch_saved_state = _save_global_state()
    from . import _session

    # Pinned devices also bound what library checks (e.g. the contract's device_parity) use.
    _session.selected_devices = config.getoption("torch_devices")
    for line in (
        "torch_golden(name): name of the golden file used by `golden`",
        "torch_max_latency(ms): median latency budget enforced by `latency`",
        "torch_seed(n): seed used by `torch_seed` for this test",
        "torch_gpu_exclusive(timeout=1800): serialize this test with other exclusive "
        "tests on the same GPU (useful under pytest-xdist)",
        "torch_tolerance(**entries): override default tolerances, keys as in "
        "[tool.pytest_torch.tolerance], e.g. bfloat16=(2e-2, 1e-2)",
        "torch_dtypes(*dtypes): dtypes the `torch_dtype` fixture runs this test with",
        "torch_benchmark: a benchmark; runs only with --torch-benchmark",
    ):
        config.addinivalue_line("markers", line)
    _configure_tolerances(config)
    _configure_repro(config)
    _configure_zero_code(config)
    _artifacts.configure(config.getoption("torch_failure_artifacts"))
    # Workers forward their reports to the controller, which prints the annotations.
    if config.getoption("torch_github_annotations") and not hasattr(config, "workerinput"):
        config.pluginmanager.register(_GithubAnnotations(config), "torch-github-annotations")


def _configure_tolerances(config) -> None:
    """Load ``[tool.pytest_torch.tolerance]`` from the project's ``pyproject.toml``."""
    inipath = getattr(config, "inipath", None)
    path = inipath if inipath is not None and inipath.name == "pyproject.toml" else None
    path = path or Path(config.rootpath) / "pyproject.toml"
    entries = {}
    if path.is_file():
        try:
            entries = _tolerances.load_pyproject(path)
        except ValueError as exc:
            raise pytest.UsageError(str(exc)) from None
    _tolerances.configure(entries)


def _configure_repro(config) -> None:
    replay = config.getoption("torch_repro")
    try:
        _repro.configure(
            config.getoption("torch_repro_dir"),
            config.getoption("torch_repro_max_weights_mb"),
            replay,
        )
    except Exception as exc:  # missing file, not a torch file, not a bundle, ...
        first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        raise pytest.UsageError(f"--torch-repro: cannot load {replay}: {first_line}") from None
    bundle = _repro.replay_bundle()
    if bundle is not None and _no_explicit_paths(config):
        # Collect only the bundle's test file: faster, and independent of testpaths/patterns.
        test_file = Path(config.rootpath) / bundle["nodeid"].split("::")[0]
        if test_file.is_file():
            config.args[:] = [str(test_file)]
    if bundle is not None and not config.getoption("torch_devices") and bundle.get("params", {}).get("torch_device"):
        # Collect the recorded device's variant (it skips if that device is missing here).
        config.option.torch_devices = [bundle["params"]["torch_device"]]


def _configure_zero_code(config) -> None:
    from . import _session

    specs = config.getoption("torch_models") or []
    _session.model_specs = list(specs)
    if not specs:
        return
    from . import zero_code_tests

    requested = config.getoption("torch_model_checks")
    checks = tuple(c.strip() for c in requested.split(",") if c.strip()) if requested else zero_code_tests.DEFAULT_CHECKS
    unknown = sorted(set(checks) - set(zero_code_tests.CHECKS))
    if unknown:
        raise pytest.UsageError(f"--torch-model-checks: unknown {unknown}; choose from {list(zero_code_tests.CHECKS)}")
    _session.model_checks = checks
    _session.model_inputs = config.getoption("torch_model_inputs")
    _session.model_pretrained = config.getoption("torch_model_pretrained")
    # Without file arguments, run only the zero-code tests: regular collection is skipped
    # (the rootdir and its pyproject.toml stay the project's).
    _session.zero_code_only = _no_explicit_paths(config)


def pytest_ignore_collect(collection_path, config):
    from . import _session

    return True if _session.zero_code_only else None


def _zero_code_items(session) -> list:
    """The zero-code test module, collected under the readable node id ``torch-model``."""
    from . import zero_code_tests

    module = pytest.Module.from_parent(session, path=Path(zero_code_tests.__file__), nodeid="torch-model")
    items: list[pytest.Item] = []
    pending: list = list(module.collect())
    while pending:
        node = pending.pop(0)
        if isinstance(node, pytest.Item):
            items.append(node)
        else:
            pending[:0] = node.collect()  # a class or other collector: expand in place
    return items


def _no_explicit_paths(config) -> bool:
    """True when pytest was given no file/directory arguments."""
    explicit = getattr(getattr(type(config), "ArgsSource", None), "ARGS", None)
    if explicit is not None:  # pytest >= 7.1 records where its arguments came from
        return getattr(config, "args_source", explicit) != explicit
    # pytest 7.0: look for a positional argument naming an existing file or directory.
    return not any(
        not arg.startswith("-") and Path(arg.split("::")[0]).exists()
        for arg in config.invocation_params.args
    )


def _callspec_params(item) -> dict[str, str]:
    callspec = getattr(item, "callspec", None)
    return {k: str(v) for k, v in callspec.params.items()} if callspec else {}


def _matches_bundle(item, bundle: dict) -> bool:
    if item.nodeid.split("[")[0] != bundle["nodeid"].split("[")[0]:
        return False
    recorded = {k: v for k, v in bundle.get("params", {}).items() if k != "torch_device"}
    current = {k: v for k, v in _callspec_params(item).items() if k != "torch_device"}
    return recorded == current


def pytest_collection_modifyitems(session, config, items):
    from . import _session

    if not config.getoption("torch_benchmark"):  # benchmarks are slow: opt in
        benchmarks = [i for i in items if i.get_closest_marker("torch_benchmark") is not None]
        if benchmarks:
            config.hook.pytest_deselected(items=benchmarks)
            items[:] = [i for i in items if i.get_closest_marker("torch_benchmark") is None]

    if _session.model_specs:  # zero-code tests: test_builds and the selected checks
        from . import zero_code_tests

        selected = {"test_builds"} | {zero_code_tests.CHECKS[c] for c in _session.model_checks}
        generated = _zero_code_items(session)
        dropped = [i for i in generated if i.originalname not in selected]
        if dropped:
            config.hook.pytest_deselected(items=dropped)
        items.extend(i for i in generated if i.originalname in selected)
    bundle = _repro.replay_bundle()
    if bundle is None:
        return
    keep = [item for item in items if _matches_bundle(item, bundle)]
    if not keep:
        raise pytest.UsageError(
            f"--torch-repro: test {bundle['nodeid']!r} was not collected. Run pytest from the "
            "same root directory, and include the test's file in the arguments if you pass any."
        )
    config.hook.pytest_deselected(items=[item for item in items if item not in keep])
    items[:] = keep


def pytest_runtest_setup(item):
    _artifacts.set_current_test(item.nodeid)
    _repro.start_test(item.nodeid, _callspec_params(item), replaying=_repro.replay_bundle() is not None)


def pytest_runtest_teardown(item, nextitem):
    _artifacts.set_current_test(None)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.failed:
        path = _repro.write_bundle(report.longreprtext)
        if path is not None:
            report.sections.append(
                ("torch-repro", f"bundle: {path}\nreplay: pytest --torch-repro={path}")
            )
    if report.when == "teardown":
        _repro.finish_test()


def pytest_report_header(config):
    try:
        lines = [f"pytest-torch: torch {importlib.metadata.version('torch')}"]
    except importlib.metadata.PackageNotFoundError:
        lines = ["pytest-torch: torch not installed"]
    bundle = _repro.replay_bundle()
    if bundle is not None:
        lines += ["pytest-torch: " + line for line in _repro.describe(bundle)]
    from . import _session

    if _session.model_specs:
        weights = "pretrained" if _session.model_pretrained else "random-init"
        lines.append(f"pytest-torch: testing {', '.join(_session.model_specs)} ({weights}); "
                     f"checks: {', '.join(_session.model_checks)}")
    return lines


# --------------------------------------------------------------------------- #
# Device handling (+ GPU assignment under pytest-xdist)
# --------------------------------------------------------------------------- #
def _device_available(device: str) -> bool:
    from .devices import is_available

    try:
        return is_available(device)
    except RuntimeError:  # not a device type torch knows
        return False


def _devices(config) -> list[str]:
    requested = config.getoption("torch_devices")
    if requested:
        return requested
    from .devices import available_devices

    return available_devices()


def _gpu_kind(device: str | None) -> str | None:
    """``cuda`` / ``xpu`` for a GPU device string, else ``None``."""
    from .devices import GPU_KINDS

    kind = (device or "").split(":")[0]
    return kind if kind in GPU_KINDS else None


def _assigned_gpu(config, kind: str) -> int | None:
    """``kind`` GPU index for the current xdist worker (``None`` outside xdist workers)."""
    from .parallel import assign_gpu, parse_gpu_option, worker_index

    worker = worker_index()
    if worker is None:
        return None
    from .devices import device_count

    gpus = parse_gpu_option(config.getoption("torch_xdist_gpus"), lambda: device_count(kind))
    return assign_gpu(worker, gpus)


_configured_gpus: set[tuple[str, int]] = set()


def _apply_memory_fraction(config, kind: str, gpu: int) -> None:
    import warnings

    from .devices import device_count, set_memory_fraction
    from .parallel import memory_fraction, parse_gpu_option, worker_count, worker_index

    option = str(config.getoption("torch_gpu_memory_fraction")).strip().lower()
    if option in ("", "none", "off"):
        return
    if option == "auto":
        gpus = parse_gpu_option(config.getoption("torch_xdist_gpus"), lambda: device_count(kind))
        fraction = memory_fraction(worker_index(), worker_count(), gpus)
    else:
        fraction = float(option)
    if fraction is not None and not set_memory_fraction(kind, fraction, gpu):
        warnings.warn(f"--torch-gpu-memory-fraction: this torch cannot cap {kind} memory", stacklevel=1)


def _assign_worker_gpu(config, device: str) -> str:
    """Map plain ``"cuda"`` / ``"xpu"`` to this xdist worker's GPU (test ids stay unchanged)."""
    kind = _gpu_kind(device)
    if kind is None or device != kind:
        return device
    gpu = _assigned_gpu(config, kind)
    if gpu is None:
        return device
    from .devices import set_device

    set_device(f"{kind}:{gpu}")
    if (kind, gpu) not in _configured_gpus:
        _configured_gpus.add((kind, gpu))
        _apply_memory_fraction(config, kind, gpu)
    return f"{kind}:{gpu}"


_FLOAT_DTYPES = ("float64", "float32", "float16", "bfloat16")


def _dtype_name(value) -> str:
    name = str(value).replace("torch.", "")
    if name not in _FLOAT_DTYPES:
        raise pytest.UsageError(f"unsupported torch dtype {value!r}; use one of {list(_FLOAT_DTYPES)}")
    return name


def _dtypes(metafunc) -> list[str]:
    marker = metafunc.definition.get_closest_marker("torch_dtypes")
    if marker is not None and marker.args:
        return [_dtype_name(d) for d in marker.args]
    requested = metafunc.config.getoption("torch_dtypes")
    return [_dtype_name(d) for d in requested] if requested else ["float32"]


def pytest_generate_tests(metafunc):
    configs = list(getattr(metafunc.cls, "configs", None) or ()) if metafunc.cls is not None else []
    if "model_config" in metafunc.fixturenames and configs:
        # first, so ids read [config-device]
        metafunc.parametrize("model_config", configs, ids=[metafunc.cls.config_id(c) for c in configs],
                             indirect=True)
    if "torch_device" in metafunc.fixturenames:
        metafunc.parametrize("torch_device", _devices(metafunc.config), indirect=True)
    if "benchmark_case" in metafunc.fixturenames and metafunc.cls is not None:
        from .benchmark import cases

        matrix = cases(metafunc.cls)
        metafunc.parametrize("benchmark_case", matrix, ids=[c.id for c in matrix])
    if "torch_dtype" in metafunc.fixturenames:
        metafunc.parametrize("torch_dtype", _dtypes(metafunc), indirect=True)


_dtype_support: dict[tuple[str, str], str | None] = {}


def _unsupported(device: str, dtype) -> str | None:
    """Why ``dtype`` can't run on ``device`` (a tiny matmul + reduction probe), or None."""
    import torch

    key = (device, str(dtype))
    if key not in _dtype_support:
        try:
            x = torch.ones(2, 2, dtype=dtype, device=device)
            float((x @ x).sum())
            _dtype_support[key] = None
        except Exception as exc:  # e.g. float64 on MPS
            _dtype_support[key] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"
    return _dtype_support[key]


@pytest.fixture
def torch_dtype(request):
    """Parametrizes the test over floating dtypes (default float32; ``--torch-dtype``,
    ``@pytest.mark.torch_dtypes(...)``). Skips combinations the test's device can't run.
    ``make_tensor``'s default float dtype follows it."""
    import torch

    dtype = getattr(torch, request.param)
    if "torch_device" in request.fixturenames:
        device = request.getfixturevalue("torch_device")
        reason = _unsupported(device, dtype)
        if reason:
            pytest.skip(f"{request.param} is not supported on {device}: {reason}")
    return dtype


@pytest.fixture
def model_config(request):
    """The current config of a ``ModelSuite`` (or any test class) with a ``configs``
    attribute, loaded and with ``config_overrides`` applied; ``None`` without configs."""
    if not hasattr(request, "param"):
        return None
    resolve = getattr(request.instance, "resolve_config", None)
    config = resolve(request.param) if resolve is not None else request.param
    if request.instance is not None:
        request.instance.config = config  # self.config works in hand-written tests too
    return config


@pytest.fixture
def torch_device(request):
    device = request.param
    if not _device_available(device):
        pytest.skip(f"{device} is not available")
    device = _assign_worker_gpu(request.config, device)
    _repro.note_device(device)
    # "<device>.<key>" tolerance entries apply to everything this test compares.
    with _tolerances.active_test_device(device):
        yield device


@pytest.fixture(autouse=True)
def _torch_tolerance(request):
    """``@pytest.mark.torch_tolerance(...)``: markers on module, class and test all apply,
    the closest one winning for the keys they share."""
    import contextlib

    markers = list(request.node.iter_markers("torch_tolerance"))  # closest first
    with contextlib.ExitStack() as stack:
        for marker in reversed(markers):
            entries = dict(*marker.args, **marker.kwargs)
            try:
                stack.enter_context(_tolerances.tolerance_overrides(**entries))
            except ValueError as exc:
                pytest.fail(f"@pytest.mark.torch_tolerance: {exc}", pytrace=False)
        yield


@pytest.fixture(autouse=True)
def _torch_gpu_exclusive(request):
    """``@pytest.mark.torch_gpu_exclusive``: one such test at a time per GPU, across xdist workers."""
    marker = request.node.get_closest_marker("torch_gpu_exclusive")
    if marker is None:
        yield
        return
    from .parallel import FileLock

    factory = request.getfixturevalue("tmp_path_factory")
    callspec = getattr(request.node, "callspec", None)
    kind = _gpu_kind(callspec.params.get("torch_device") if callspec else None)
    if kind is None:  # no GPU device param: lock the first GPU backend this machine has
        from .devices import device_count

        kind = next((k for k in ("cuda", "xpu") if device_count(k)), "cuda")
    gpu = _assigned_gpu(request.config, kind)
    name = f"pytest-torch-{kind}-{'any' if gpu is None else gpu}.lock"
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
    bundle = _repro.replay_bundle()
    if bundle is not None and bundle.get("seed") is not None:
        return int(bundle["seed"])
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
    _repro.note_seed(seed)
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
def make_tensor(request, torch_generator, torch_device):
    """Factory for reproducible tensors, generated on the CPU and moved to ``torch_device``.

    ``make_tensor(2, 3)``                          -> normal(0, 1), float32 (or ``torch_dtype``)
    ``make_tensor(2, 3, kind="uniform", low=0, high=1)``
    ``make_tensor(4, 8, dtype=torch.long, low=1, high=50)``   -> integers in [1, 50)
    """
    import torch

    def make(*shape, dtype=None, kind=None, low=None, high=None):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = tuple(shape[0])
        if dtype is None:  # follows torch_dtype when the test is parametrized over dtypes
            uses_dtype = "torch_dtype" in request.fixturenames
            dtype = request.getfixturevalue("torch_dtype") if uses_dtype else torch.float32
        is_int = not (dtype.is_floating_point or dtype.is_complex)
        kind = kind or ("randint" if is_int else "normal")
        if kind not in ("normal", "uniform", "randint"):
            raise ValueError(f"unknown kind {kind!r}; use normal, uniform or randint")

        def generate():
            if kind == "normal":
                data = torch.randn(shape, generator=torch_generator, dtype=torch.float64)
            elif kind == "uniform":
                lo = -1.0 if low is None else float(low)
                hi = 1.0 if high is None else float(high)
                data = lo + (hi - lo) * torch.rand(shape, generator=torch_generator, dtype=torch.float64)
            else:
                lo = 0 if low is None else int(low)
                hi = (2 if dtype == torch.bool else 100) if high is None else int(high)
                data = torch.randint(lo, hi, shape, generator=torch_generator)
            return data.to(device=torch_device, dtype=dtype)

        # Recorded into failure bundles, and replayed bit for bit under --torch-repro.
        return _repro.make_tensor_hook(tuple(shape), dtype, torch_device, generate)

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


# --------------------------------------------------------------------------- #
# Dataset metric baselines
# --------------------------------------------------------------------------- #
@pytest.fixture
def metric_baseline(request):
    """Return ``check(model, loader, metrics, *, name=None, inputs=..., max_batches=None,
    context=None, fingerprint_model=True, tolerance=0.01, higher_is_better=True,
    minimum=None, maximum=None, confidence=0.95, resamples=2000)``.

    Evaluates the model (see :func:`pytest_torch.evaluation.evaluate_detailed`), then
    compares with the baseline recorded by ``--torch-update-metrics`` (one per test, shared
    by all devices): only if it measured the same thing (``context``, data, sample
    selection, metric configuration), paired with confidence intervals for per-sample
    metrics. Absolute ``minimum``/``maximum`` thresholds always apply. Returns a
    ``MetricResult``.
    """
    from .evaluation import check_metrics, default_inputs, evaluate_detailed
    from .perf import PerfStore

    config = request.config
    path = config.getoption("torch_metrics_file") or Path(config.rootpath) / ".torch-metric-baselines.json"
    store = PerfStore(path)
    default_name = request.node.nodeid.split("[")[0]

    def check(model, loader, metrics, *, name=None, inputs=default_inputs, max_batches=None,
              context=None, fingerprint_model=True, **options):
        evaluation = evaluate_detailed(model, loader, metrics, inputs=inputs, max_batches=max_batches,
                                       context=context, fingerprint_model=fingerprint_model)
        result = check_metrics(
            store,
            name or default_name,
            evaluation,
            update=config.getoption("torch_update_metrics"),
            strict=config.getoption("torch_metrics_strict"),
            **options,
        )
        request.node.add_report_section("call", "torch-metrics", str(result))
        return result

    return check


# --------------------------------------------------------------------------- #
# Benchmark report
# --------------------------------------------------------------------------- #
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    from .benchmark import format_table, results

    if not results:
        return
    terminalreporter.section("pytest-torch benchmarks")
    for line in format_table(results):
        terminalreporter.write_line(line)
    path = config.getoption("torch_benchmark_json")
    if path:
        import json

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps([r.as_dict() for r in results], indent=2) + "\n", encoding="utf-8")
        terminalreporter.write_line(f"benchmark results written to {path}")
