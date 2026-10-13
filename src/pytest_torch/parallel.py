"""pytest-xdist awareness: GPU assignment, memory caps and an exclusive-access file lock.

Torch-free on purpose: everything here is plain Python so it is cheap to import.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable

if sys.platform == "win32":
    import msvcrt

    def _lock(handle) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(handle) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(handle) -> None:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(handle) -> None:
        fcntl.flock(handle, fcntl.LOCK_UN)


def worker_index() -> int | None:
    """Index of the current xdist worker (``gw3`` -> 3); ``None`` outside xdist workers."""
    worker = os.environ.get("PYTEST_XDIST_WORKER", "")
    if worker.startswith("gw") and worker[2:].isdigit():
        return int(worker[2:])
    return None


def worker_count() -> int:
    try:
        return max(1, int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")))
    except ValueError:
        return 1


def parse_gpu_option(value: str, device_count: Callable[[], int]) -> list[int]:
    """``'auto'`` -> all GPUs, ``'none'`` -> [], ``'0,2'`` -> [0, 2]."""
    value = str(value).strip().lower()
    if value in ("", "none", "off"):
        return []
    if value == "auto":
        return list(range(device_count()))
    return [int(item) for item in value.split(",") if item.strip()]


def assign_gpu(worker: int | None, gpus: list[int]) -> int | None:
    """Round-robin: worker i gets ``gpus[i % len(gpus)]``."""
    if worker is None or not gpus:
        return None
    return gpus[worker % len(gpus)]


def workers_sharing(worker: int, workers: int, gpus: list[int]) -> int:
    """How many workers (including this one) end up on the same GPU."""
    slot = worker % len(gpus)
    return sum(1 for other in range(workers) if other % len(gpus) == slot)


def memory_fraction(
    worker: int | None, workers: int, gpus: list[int], headroom: float = 0.95
) -> float | None:
    """Memory share per process when several workers share a GPU (``None``: no cap needed)."""
    if worker is None or not gpus:
        return None
    sharing = workers_sharing(worker, workers, gpus)
    return None if sharing <= 1 else headroom / sharing


class FileLock:
    """Cross-process, cross-platform exclusive lock on a file (blocks until acquired)."""

    def __init__(self, path: str | os.PathLike, timeout: float = 1800.0, poll: float = 0.05):
        self.path = Path(path)
        self.timeout = timeout
        self.poll = poll
        self._handle = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+b")
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                _lock(handle)
                self._handle = handle
                return
            except OSError:
                if time.monotonic() >= deadline:
                    handle.close()
                    raise TimeoutError(
                        f"could not acquire {self.path} within {self.timeout:g}s"
                    ) from None
                time.sleep(self.poll)

    def release(self) -> None:
        if self._handle is not None:
            try:
                _unlock(self._handle)
            finally:
                self._handle.close()
                self._handle = None

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
