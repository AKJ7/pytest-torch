"""Markers for requirements, per-device skips and expected failures, test modes and
memory budgets. Imports torch only for tests that carry one of them."""
from __future__ import annotations

import contextlib
import gc
import importlib.metadata
import importlib.util
import sys
from collections.abc import Iterator
from typing import Any

import pytest

REQUIREMENTS = ("cuda", "xpu", "mps", "gpu", "torch", "python", "compile", "gpu_memory_gb",
                "cuda_capability", "packages")
MODE_MARKERS = ("torch_deterministic", "torch_default_dtype", "torch_matmul_precision",
                "torch_no_host_sync", "torch_detect_anomaly")
LINES = (
    "torch_requires(cuda=, xpu=, mps=, gpu=, torch=, python=, compile=, gpu_memory_gb=, "
    "cuda_capability=, packages=): skip unless every requirement holds",
    "torch_skip_on(*devices, dtype=None, reason=): skip the test's runs on these devices",
    "torch_xfail_on(*devices, dtype=None, reason=, raises=None, strict=True): expected failure on these devices",
    "torch_deterministic: deterministic algorithms during this test",
    "torch_default_dtype(dtype): torch's default dtype during this test",
    "torch_matmul_precision(precision): float32 matmul precision ('highest' disables TF32) during this test",
    "torch_no_host_sync: fail on host-device synchronizations during this test (CUDA)",
    "torch_detect_anomaly: autograd anomaly detection during this test",
    "torch_max_memory(mb): peak memory budget of the test body (CUDA, XPU)",
    "torch_memory_leak(iterations=5): fail when repeated runs keep accumulating tensors or memory",
)


def item_device(item: Any) -> str | None:
    """The test's ``torch_device`` parameter (``"cuda"``, ``"cpu"``, ...), if any."""
    callspec = getattr(item, "callspec", None)
    return callspec.params.get("torch_device") if callspec is not None else None


def _kind(device: Any) -> str:
    return str(device).split(":")[0]


def _dtype_name(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")


# --------------------------------------------------------------------------- #
# torch_requires
# --------------------------------------------------------------------------- #
def _version_ok(installed: str, spec: str) -> bool:
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    return SpecifierSet(spec).contains(Version(installed), prereleases=True)


def _gpu_device(item: Any) -> str | None:
    """The GPU a requirement applies to: the test's own, or the first available one."""
    from .devices import is_available

    device = item_device(item)
    if device is not None and _kind(device) in ("cuda", "xpu"):
        return device if is_available(device) else None
    for kind in ("cuda", "xpu"):
        if is_available(kind):
            return kind
    return None


def unmet_requirement(item: Any, requirements: dict[str, Any]) -> str | None:
    """Why ``@pytest.mark.torch_requires(**requirements)`` doesn't hold, or None."""
    unknown = sorted(set(requirements) - set(REQUIREMENTS))
    if unknown:
        raise pytest.UsageError(f"torch_requires: unknown requirement(s) {unknown}; use {list(REQUIREMENTS)}")
    if requirements.get("python") and not _version_ok(
            ".".join(map(str, sys.version_info[:3])), requirements["python"]):
        return f"needs Python {requirements['python']} (this is {sys.version.split()[0]})"
    if requirements.get("torch"):
        installed = importlib.metadata.version("torch")
        if not _version_ok(installed, requirements["torch"]):
            return f"needs torch {requirements['torch']} (installed: {installed})"
    packages = requirements.get("packages") or ()
    for package in [packages] if isinstance(packages, str) else packages:
        if importlib.util.find_spec(package) is None:
            return f"needs the {package!r} package"
    if not any(requirements.get(k) for k in ("cuda", "xpu", "mps", "gpu", "compile", "gpu_memory_gb",
                                             "cuda_capability")):
        return None

    from .devices import is_available

    for kind in ("cuda", "xpu", "mps"):
        if requirements.get(kind) and not is_available(kind):
            return f"needs {kind.upper() if kind != 'mps' else 'MPS'}"
    if requirements.get("gpu") and not any(is_available(k) for k in ("cuda", "xpu", "mps")):
        return "needs a GPU (CUDA, XPU or MPS)"
    if requirements.get("compile"):
        from .utils import is_compile_supported

        if not is_compile_supported():
            return "needs torch.compile, which isn't supported on this platform"
    if requirements.get("cuda_capability"):
        import torch

        wanted = tuple(requirements["cuda_capability"])
        device = _gpu_device(item)
        if device is None or _kind(device) != "cuda":
            return f"needs a CUDA GPU with compute capability {wanted}"
        found = torch.cuda.get_device_capability(device)
        if tuple(found) < wanted:
            return f"needs CUDA compute capability >= {wanted} (this GPU: {tuple(found)})"
    if requirements.get("gpu_memory_gb"):
        from .devices import _gpu_backend

        wanted_gb = float(requirements["gpu_memory_gb"])
        device = _gpu_device(item)
        if device is None:
            return f"needs a GPU with {wanted_gb:g} GB of memory"
        total = _gpu_backend(device).get_device_properties(device).total_memory / 2**30
        if total < wanted_gb:
            return f"needs {wanted_gb:g} GB of GPU memory ({device} has {total:.1f} GB)"
    return None


# --------------------------------------------------------------------------- #
# torch_skip_on / torch_xfail_on
# --------------------------------------------------------------------------- #
def _matches(marker: Any, item: Any) -> bool:
    device = item_device(item)
    if device is None:
        raise pytest.UsageError(f"{item.nodeid}: @pytest.mark.{marker.name} applies to tests that use the "
                                "torch_device fixture")
    if not marker.args:
        raise pytest.UsageError(f"{item.nodeid}: @pytest.mark.{marker.name} needs at least one device, "
                                f"e.g. {marker.name}('mps')")
    if _kind(device) not in {_kind(d) for d in marker.args} and device not in marker.args:
        return False
    dtype = marker.kwargs.get("dtype")
    if dtype is None:
        return True
    dtypes = dtype if isinstance(dtype, (list, tuple)) else [dtype]
    param = item.callspec.params.get("torch_dtype")
    return param is not None and param in {_dtype_name(d) for d in dtypes}


def apply_device_markers(items: list[Any]) -> None:
    """Turn ``torch_requires``, ``torch_skip_on`` and ``torch_xfail_on`` into skip/xfail marks
    on the matching runs (reported at the test's location)."""
    for item in items:
        for marker in item.iter_markers("torch_requires"):  # every one applies: module, class, test
            reason = unmet_requirement(item, dict(marker.kwargs))
            if reason is not None:
                item.add_marker(pytest.mark.skip(reason=reason))
                break
        for marker in item.iter_markers("torch_skip_on"):
            if _matches(marker, item):
                item.add_marker(pytest.mark.skip(reason=marker.kwargs.get("reason")
                                                 or f"skipped on {item_device(item)} (torch_skip_on)"))
        for marker in item.iter_markers("torch_xfail_on"):
            if _matches(marker, item):
                item.add_marker(pytest.mark.xfail(
                    reason=marker.kwargs.get("reason") or f"expected to fail on {item_device(item)}",
                    raises=marker.kwargs.get("raises"),
                    strict=marker.kwargs.get("strict", True),
                ))


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _default_dtype(dtype: Any) -> Iterator[None]:
    import torch

    dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@contextlib.contextmanager
def _matmul_precision(precision: str) -> Iterator[None]:
    import torch

    if precision not in ("highest", "high", "medium"):
        raise pytest.UsageError(f"torch_matmul_precision: use 'highest', 'high' or 'medium', got {precision!r}")
    previous = torch.get_float32_matmul_precision()
    previous_cudnn = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision(precision)
    torch.backends.cudnn.allow_tf32 = precision != "highest"  # convolutions too
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)
        torch.backends.cudnn.allow_tf32 = previous_cudnn


@contextlib.contextmanager
def _no_host_sync(request: Any) -> Iterator[None]:
    import torch

    device = item_device(request.node)
    on_cuda = _kind(device) == "cuda" if device is not None else torch.cuda.is_available()
    if not on_cuda:
        request.node.add_report_section("call", "torch_no_host_sync", "not checked: the test doesn't run on CUDA")
        yield
        return
    from .sync import forbid_host_sync

    with forbid_host_sync("error"):
        yield


def mode_contexts(request: Any, stack: contextlib.ExitStack) -> None:
    """Enter the contexts of the test's mode markers."""
    node = request.node
    if node.get_closest_marker("torch_deterministic") is not None:
        from .utils import deterministic_algorithms

        stack.enter_context(deterministic_algorithms(True))
    marker = node.get_closest_marker("torch_default_dtype")
    if marker is not None:
        stack.enter_context(_default_dtype(marker.args[0] if marker.args else marker.kwargs["dtype"]))
    marker = node.get_closest_marker("torch_matmul_precision")
    if marker is not None:
        stack.enter_context(_matmul_precision(marker.args[0] if marker.args else marker.kwargs.get("precision", "highest")))
    if node.get_closest_marker("torch_no_host_sync") is not None:
        stack.enter_context(_no_host_sync(request))
    if node.get_closest_marker("torch_detect_anomaly") is not None:
        import torch

        stack.enter_context(torch.autograd.set_detect_anomaly(True))


# --------------------------------------------------------------------------- #
# Memory budgets
# --------------------------------------------------------------------------- #
def _memory_device(item: Any) -> Any:
    """Where memory is measured: the test's device (as mapped for this worker) if it has
    allocator statistics; else the current CUDA/XPU device for tests without one."""
    from .devices import is_available, supports_memory_stats

    device = item.funcargs.get("torch_device") if hasattr(item, "funcargs") else None
    if device is not None:
        return device if supports_memory_stats(device) else None
    for kind in ("cuda", "xpu"):
        if is_available(kind) and supports_memory_stats(kind):
            return kind
    return None


def _allocated(device: Any) -> int:
    from .devices import _gpu_backend, synchronize

    synchronize(device)
    return int(_gpu_backend(device).memory_allocated(device))


def _run_with_budget(item: Any, run: Any, budget_mb: float) -> None:
    device = _memory_device(item)
    if device is None:
        item.add_report_section("call", "torch_max_memory",
                                f"not checked: no allocator statistics on {item_device(item) or 'this machine'}")
        run()
        return
    from .devices import _gpu_backend

    gpu = _gpu_backend(device)
    baseline = _allocated(device)
    gpu.reset_peak_memory_stats(device)
    run()
    _allocated(device)  # synchronize
    peak_mb = (gpu.max_memory_allocated(device) - baseline) / 2**20
    item.add_report_section("call", "torch_max_memory", f"peak {peak_mb:.1f} MB on {device} (budget {budget_mb:g} MB)")
    if peak_mb > budget_mb:
        pytest.fail(f"peak memory {peak_mb:.1f} MB exceeds the budget of {budget_mb:g} MB on {device} "
                    f"(memory allocated during the test, above the {baseline / 2**20:.1f} MB allocated "
                    "before it)", pytrace=False)


def _live_tensors() -> list[Any]:
    import torch

    # type(), not isinstance(): reading __class__ trips deprecation warnings on some objects
    return [obj for obj in gc.get_objects() if issubclass(type(obj), torch.Tensor)]


def _describe(tensor: Any, ours: tuple[Any, ...]) -> str:
    text = f"{tuple(tensor.shape)} {_dtype_name(tensor.dtype)} on {tensor.device}"
    if tensor.grad_fn is not None:
        text += f" with an autograd graph ({type(tensor.grad_fn).__name__})"
    for holder in gc.get_referrers(tensor):
        if any(holder is container for container in ours):  # the check's own lists
            continue
        if isinstance(holder, (list, dict, set, tuple)):
            text += f", held by a {type(holder).__name__} of {len(holder)} item(s)"
            break
    return text


def _run_leak_check(item: Any, run: Any, iterations: int) -> None:
    if iterations < 2:
        raise pytest.UsageError("torch_memory_leak: iterations must be >= 2")
    from . import repro

    device = _memory_device(item)
    counts: list[int] = []
    memory: list[int] = []
    first_ids: set[int] = set()
    recording = repro.current()  # failure bundles keep one run's make_tensor inputs, not all
    replay_start = recording.replay_index if recording is not None else 0
    recorded = 0
    for index in range(iterations + 1):  # one warmup run: lazy initialization and caches
        if recording is not None:
            recording.replay_index = replay_start  # every run replays the same inputs
        run()
        if recording is not None:
            if index == 0:
                recorded = len(recording.inputs)
            else:
                del recording.inputs[recorded:]
        gc.collect()
        scan = _live_tensors()
        counts.append(len(scan))
        if device is not None:
            memory.append(_allocated(device))
        if index == 0:
            first_ids = {id(t) for t in scan}
        if index < iterations:
            del scan  # holding it would keep tensors alive across the next run
    growing_tensors = all(b > a for a, b in zip(counts, counts[1:]))
    growing_memory = bool(memory) and all(b > a for a, b in zip(memory, memory[1:]))
    if not (growing_tensors or growing_memory):
        del scan
        return
    lines = [f"live tensors after each run: {', '.join(map(str, counts))}"]
    if memory:
        lines.append(f"{device} memory allocated after each run (MB): "
                     f"{', '.join(f'{m / 2**20:.2f}' for m in memory)}")
    new = [t for t in scan if id(t) not in first_ids]
    shown = new[:5]
    lines += [f"new: {_describe(t, (scan, new, shown))}" for t in shown]
    hints = []
    if any(t.grad_fn is not None for t in new):
        hints.append("tensors that still hold their autograd graph are being kept: store loss.item() or "
                     "loss.detach() instead of loss")
    hints.append("look for module-level lists, dicts or caches that grow on every call")
    del scan, new, shown
    pytest.fail("Memory leak: every run of the test leaves more behind\n  " + "\n  ".join(lines)
                + "".join(f"\nHint: {h}." for h in hints), pytrace=False)


def wrap_memory_markers(items: list[Any]) -> None:
    """Run the body of tests marked ``torch_max_memory``/``torch_memory_leak`` under the check."""
    for item in items:
        budget = item.get_closest_marker("torch_max_memory")
        leak = item.get_closest_marker("torch_memory_leak")
        if (budget is None and leak is None) or not hasattr(item, "runtest"):
            continue
        run = item.runtest
        if leak is not None:
            iterations = leak.args[0] if leak.args else leak.kwargs.get("iterations", 5)
            run = _bind(_run_leak_check, item, run, iterations)
        if budget is not None:
            mb = budget.args[0] if budget.args else budget.kwargs.get("mb")
            if not isinstance(mb, (int, float)) or mb <= 0:
                raise pytest.UsageError(f"{item.nodeid}: torch_max_memory needs a positive mb=, got {mb!r}")
            run = _bind(_run_with_budget, item, run, float(mb))
        item.runtest = run


def _bind(check: Any, item: Any, run: Any, arg: Any) -> Any:
    def runtest() -> None:
        check(item, run, arg)

    return runtest
