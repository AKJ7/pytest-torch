"""Detect unexpected host-device synchronization (CUDA)."""
from __future__ import annotations

import contextlib
from typing import Any, Callable

import pytest
import torch


@contextlib.contextmanager
def forbid_host_sync(mode: str = "error"):
    """Inside the block, synchronizing CUDA ops (``.item()``, ``.cpu()``, ``if tensor:``)
    raise (``mode="error"``) or warn (``"warn"``). Skips the test when CUDA is unavailable.
    """
    if mode not in ("error", "warn"):
        raise ValueError("mode must be 'error' or 'warn'")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available: host-device sync detection needs CUDA")
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode(mode)
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(previous)


def assert_no_host_sync(fn: Callable, *args, **kwargs) -> Any:
    """Call ``fn`` and fail if it triggers a host-device synchronization.

    The original error (with the offending line in its traceback) is chained.
    Returns ``fn``'s result.
    """
    with forbid_host_sync("error"):
        try:
            with torch.inference_mode():
                return fn(*args, **kwargs)
        except RuntimeError as exc:
            if "synchroniz" in str(exc).lower():
                raise AssertionError(
                    "unexpected host-device synchronization (it stalls the GPU pipeline):\n"
                    + str(exc)
                ) from exc
            raise
