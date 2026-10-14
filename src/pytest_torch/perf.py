"""Performance measurement (latency, peak memory) and baseline tracking."""
from __future__ import annotations

import json
import os
import platform
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from . import devices


def sync(device: Any) -> None:
    """Wait for all queued work on ``device`` (no-op on CPU)."""
    devices.synchronize(device)


@dataclass
class LatencyStats:
    median_ms: float
    min_ms: float
    max_ms: float
    iters: int
    p90_ms: float = 0.0


def measure_latency(
    fn: Callable, *args, device: Any = "cpu", warmup: int = 5, iters: int = 20, **kwargs
) -> LatencyStats:
    """Median/p90/min/max wall time of ``fn(*args, **kwargs)`` with warmup and device sync."""
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            fn(*args, **kwargs)
        sync(device)
        for _ in range(iters):
            start = time.perf_counter()
            fn(*args, **kwargs)
            sync(device)
            times.append((time.perf_counter() - start) * 1e3)
    ordered = sorted(times)
    p90 = ordered[min(len(ordered) - 1, round(0.9 * (len(ordered) - 1)))]
    return LatencyStats(statistics.median(times), min(times), max(times), iters, p90)


def measure_peak_memory(fn: Callable, *args, device: Any = "cpu", **kwargs) -> float | None:
    """Peak extra allocated memory in MB during one call. CUDA and XPU; ``None`` elsewhere."""
    if not devices.supports_memory_stats(device):
        return None
    backend = devices._gpu_backend(device)  # supports_memory_stats: a GPU backend
    sync(device)
    backend.reset_peak_memory_stats(device)
    baseline = backend.memory_allocated(device)
    with torch.inference_mode():
        fn(*args, **kwargs)
    sync(device)
    return (backend.max_memory_allocated(device) - baseline) / 2**20


def hardware_id(device: Any) -> str:
    """Fingerprint that keys baselines, so numbers from different machines never mix."""
    dev = torch.device(device)
    name = devices.device_name(dev)
    if name is not None:
        return f"{dev.type}:{name}"
    cpu = platform.processor() or platform.machine()
    return f"{dev.type}:{platform.system()}-{platform.machine()}-{cpu}-{os.cpu_count()}c"


class PerfStore:
    """JSON file with one ``{latency_ms, peak_memory_mb}`` entry per ``name|hardware``."""

    VERSION = 1

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)

    def _load(self) -> dict:
        if not self.path.exists():
            return {"version": self.VERSION, "entries": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, key: str) -> dict | None:
        return self._load()["entries"].get(key)

    def set(self, key: str, entry: dict) -> None:
        data = self._load()
        data["entries"][key] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.write("\n")
            tmp.replace(self.path)  # atomic: readers never see a half-written file
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise


@dataclass
class PerfResult:
    latency_ms: float
    peak_memory_mb: float | None
    baseline_latency_ms: float | None = None
    baseline_peak_memory_mb: float | None = None
    updated: bool = False

    def __str__(self) -> str:
        parts = [f"latency {self.latency_ms:.3f} ms"]
        if self.baseline_latency_ms is not None:
            parts[0] += f" (baseline {self.baseline_latency_ms:.3f} ms)"
        if self.peak_memory_mb is not None:
            parts.append(f"peak memory {self.peak_memory_mb:.1f} MB")
        return ", ".join(parts) + (" [baseline updated]" if self.updated else "")


def check_perf(
    store: PerfStore,
    key: str,
    fn: Callable,
    *args,
    device: Any = "cpu",
    update: bool = False,
    strict: bool = False,
    latency_tolerance: float = 0.25,
    latency_slack_ms: float = 0.05,
    memory_tolerance: float = 0.10,
    memory_slack_mb: float = 1.0,
    warmup: int = 5,
    iters: int = 20,
) -> PerfResult:
    """Measure ``fn(*args)`` and fail if it regressed against the stored baseline.

    A regression is ``new > baseline * (1 + tolerance) + slack`` (the slack keeps tiny
    absolute differences from failing). Faster/leaner is never a failure. A missing
    baseline skips the test (or fails with ``strict=True``); ``update=True`` records one.
    Peak memory is tracked on CUDA and XPU only.
    """
    stats = measure_latency(fn, *args, device=device, warmup=warmup, iters=iters)
    peak = measure_peak_memory(fn, *args, device=device)
    return compare_perf(
        store, key, hardware_id(device), stats.median_ms, peak, update=update, strict=strict,
        latency_tolerance=latency_tolerance, latency_slack_ms=latency_slack_ms,
        memory_tolerance=memory_tolerance, memory_slack_mb=memory_slack_mb,
    )


def compare_perf(
    store: PerfStore,
    key: str,
    hardware: str,
    latency_ms: float,
    peak_memory_mb: float | None,
    *,
    update: bool = False,
    strict: bool = False,
    latency_tolerance: float = 0.25,
    latency_slack_ms: float = 0.05,
    memory_tolerance: float = 0.10,
    memory_slack_mb: float = 1.0,
) -> PerfResult:
    """Compare measured numbers with the baseline for ``key`` on ``hardware`` (see
    :func:`check_perf`), or record them with ``update=True``."""
    full_key = f"{key}|{hardware}"
    result = PerfResult(latency_ms, peak_memory_mb)
    peak = peak_memory_mb

    if update:
        store.set(
            full_key,
            {
                "latency_ms": round(latency_ms, 4),
                "peak_memory_mb": None if peak is None else round(peak, 3),
                "torch": torch.__version__,
            },
        )
        result.updated = True
        return result

    entry = store.get(full_key)
    if entry is None:
        message = (
            f"no performance baseline for {full_key!r}; "
            "record one with --torch-update-perf"
        )
        if strict:
            pytest.fail(message, pytrace=False)
        pytest.skip(message)

    base_latency = float(entry["latency_ms"])
    base_memory = entry.get("peak_memory_mb")
    result.baseline_latency_ms = base_latency
    result.baseline_peak_memory_mb = base_memory

    problems = []
    if latency_ms > base_latency * (1 + latency_tolerance) + latency_slack_ms:
        change = f"+{(latency_ms / base_latency - 1) * 100:.0f}%" if base_latency > 0 else "n/a"
        problems.append(
            f"latency regressed: {latency_ms:.3f} ms vs baseline {base_latency:.3f} ms "
            f"({change}; allowed +{latency_tolerance * 100:.0f}% and {latency_slack_ms:g} ms slack)"
        )
    if (
        peak is not None
        and base_memory is not None
        and peak > float(base_memory) * (1 + memory_tolerance) + memory_slack_mb
    ):
        problems.append(
            f"peak memory regressed: {peak:.1f} MB vs baseline {float(base_memory):.1f} MB "
            f"(allowed +{memory_tolerance * 100:.0f}% and {memory_slack_mb:g} MB slack)"
        )
    if problems:
        raise AssertionError(
            f"Performance regression in {key!r} on {hardware}:\n  "
            + "\n  ".join(problems)
            + "\n  (re-record with --torch-update-perf if the change is intended)"
        )
    return result
