"""Device discovery and per-backend helpers: cpu, cuda, xpu (Intel GPUs) and mps.

``torch.cuda`` and ``torch.xpu`` share one API (``is_available``, ``device_count``,
``synchronize``, memory stats, ...), so GPU code paths look the backend up by device type
instead of hard-coding CUDA. Functions an older torch lacks are treated as unsupported.
"""
from __future__ import annotations

from types import ModuleType
from typing import Any

import torch

#: Accelerators in the order ``torch_device`` lists them after ``cpu``.
ACCELERATORS = ("cuda", "xpu", "mps")
#: Device types with numbered devices that pytest-xdist workers can be spread over.
GPU_KINDS = ("cuda", "xpu")


def _kind(device: Any) -> str:
    return torch.device(device).type


def backend(device: Any) -> ModuleType | None:
    """``torch.cuda`` / ``torch.xpu`` / ``torch.mps`` for a device (``None`` for cpu or unknown)."""
    kind = _kind(device)
    if kind not in ACCELERATORS:
        return None
    return getattr(torch, kind, None)


def _gpu_backend(device: Any) -> ModuleType:
    """``torch.cuda`` / ``torch.xpu``: callers only ask for GPU kinds this torch has."""
    module = backend(device)
    if module is None:
        raise RuntimeError(f"this torch has no backend for {_kind(device)!r}")
    return module


def is_available(device: Any) -> bool:
    kind = _kind(device)
    if kind == "cpu":
        return True
    if kind == "mps":
        mps = getattr(torch.backends, "mps", None)
        return mps is not None and mps.is_available()
    module = backend(kind)
    try:
        return module is not None and bool(module.is_available())
    except Exception:  # a backend whose runtime is broken counts as unavailable
        return False


def available_devices() -> list[str]:
    """``["cpu", ...]`` plus every accelerator usable in this process."""
    return ["cpu", *(kind for kind in ACCELERATORS if is_available(kind))]


def device_count(kind: str) -> int:
    if kind not in GPU_KINDS or not is_available(kind):
        return 0
    return int(_gpu_backend(kind).device_count())


def synchronize(device: Any) -> None:
    """Wait for all queued work on ``device`` (no-op on CPU)."""
    kind = _kind(device)
    if kind in GPU_KINDS:
        _gpu_backend(kind).synchronize(device)
    elif kind == "mps":
        torch.mps.synchronize()


def manual_seed_all(seed: int) -> None:
    """Seed every available GPU backend (``torch.manual_seed`` covers CPU and MPS)."""
    for kind in GPU_KINDS:
        if is_available(kind):
            _gpu_backend(kind).manual_seed_all(seed)


def device_name(device: Any) -> str | None:
    module = backend(device)
    get_name = getattr(module, "get_device_name", None) if _kind(device) in GPU_KINDS else None
    return get_name(torch.device(device)) if get_name is not None else None


def supports_memory_stats(device: Any) -> bool:
    """Whether peak-memory tracking works for ``device`` (CUDA, and XPU on recent torch)."""
    if _kind(device) not in GPU_KINDS:
        return False
    module = backend(device)
    return all(
        hasattr(module, name)
        for name in ("reset_peak_memory_stats", "memory_allocated", "max_memory_allocated")
    )


def set_device(device: Any) -> None:
    if _kind(device) in GPU_KINDS:
        _gpu_backend(device).set_device(torch.device(device))


def set_memory_fraction(kind: str, fraction: float, index: int) -> bool:
    """Cap this process's memory on ``kind:index``; ``False`` if the backend can't."""
    setter = getattr(backend(kind), "set_per_process_memory_fraction", None)
    if setter is None:
        return False
    setter(fraction, index)
    return True
