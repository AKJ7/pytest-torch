"""Shared helpers: pytrees, batching, and model device/dtype inspection."""
from __future__ import annotations

import contextlib
import dataclasses
import itertools
import os
import random
from collections.abc import Mapping
from typing import Any, Callable

import torch


def _is_namedtuple(obj: Any) -> bool:
    return isinstance(obj, tuple) and hasattr(obj, "_fields")


def _is_dataclass_instance(obj: Any) -> bool:
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


# --------------------------------------------------------------------------- #
# Pytrees (dict/Mapping, list, tuple, namedtuple, dataclass). Everything else
# (tensors, numbers, strings, None, arbitrary objects) is a leaf.
# --------------------------------------------------------------------------- #
def flatten(obj: Any, path: str = "output") -> list[tuple[str, Any]]:
    """Flatten nested containers into ``(path, leaf)`` pairs.

    Paths look like ``output['logits']``, ``output['aux'][0]`` or ``output.hidden``.
    Mappings include Hugging Face ``ModelOutput`` objects.
    """
    if isinstance(obj, torch.Tensor):
        return [(path, obj)]
    out: list[tuple[str, Any]] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            out.extend(flatten(value, f"{path}[{key!r}]"))
        return out
    if _is_namedtuple(obj):
        for field in obj._fields:
            out.extend(flatten(getattr(obj, field), f"{path}.{field}"))
        return out
    if isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            out.extend(flatten(value, f"{path}[{i}]"))
        return out
    if _is_dataclass_instance(obj):
        for field in dataclasses.fields(obj):
            out.extend(flatten(getattr(obj, field.name), f"{path}.{field.name}"))
        return out
    return [(path, obj)]


def tree_map(fn: Callable[[torch.Tensor], Any], obj: Any) -> Any:
    """Apply ``fn`` to every tensor in a nested structure, preserving the structure."""
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, Mapping):
        mapped = {k: tree_map(fn, v) for k, v in obj.items()}
        try:
            return type(obj)(mapped)
        except Exception:
            return mapped
    if _is_namedtuple(obj):
        return type(obj)(*[tree_map(fn, v) for v in obj])
    if isinstance(obj, (list, tuple)):
        return type(obj)(tree_map(fn, v) for v in obj)
    if _is_dataclass_instance(obj):
        changes = {
            f.name: tree_map(fn, getattr(obj, f.name))
            for f in dataclasses.fields(obj)
            if f.init
        }
        return dataclasses.replace(obj, **changes)
    return obj


def call_model(model: Callable, inputs: Any) -> Any:
    """Call ``model`` with a tensor, a tuple of positional args, or a dict of kwargs."""
    if isinstance(inputs, Mapping):
        return model(**inputs)
    if isinstance(inputs, (list, tuple)):
        return model(*inputs)
    return model(inputs)


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #
def model_device(model: Any) -> torch.device:
    if hasattr(model, "parameters"):
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            return tensor.device
    return torch.device("cpu")


def model_dtype(model: Any) -> torch.dtype | None:
    if hasattr(model, "parameters"):
        for param in model.parameters():
            if param.is_floating_point():
                return param.dtype
    return None


def to_device(obj: Any, device: Any) -> Any:
    return tree_map(lambda t: t.to(device), obj)


def cast_floating(obj: Any, dtype: torch.dtype) -> Any:
    return tree_map(lambda t: t.to(dtype) if t.is_floating_point() else t, obj)


# --------------------------------------------------------------------------- #
# Batching. Heuristic: a tensor is "batch-aligned" if it has >= 1 dim and its
# leading dim equals the batch size (taken from the first such input tensor).
# --------------------------------------------------------------------------- #
def find_batch_size(inputs: Any) -> int | None:
    for _, leaf in flatten(inputs, "inputs"):
        if isinstance(leaf, torch.Tensor) and leaf.dim() >= 1:
            return int(leaf.shape[0])
    return None


def slice_batch(inputs: Any, index: int, batch_size: int) -> Any:
    def _slice(t: torch.Tensor) -> torch.Tensor:
        if t.dim() >= 1 and t.shape[0] == batch_size:
            return t[index : index + 1]
        return t

    return tree_map(_slice, inputs)


def batch_aligned(output: Any, batch_size: int) -> dict[str, torch.Tensor]:
    return {
        path: leaf
        for path, leaf in flatten(output)
        if isinstance(leaf, torch.Tensor)
        and leaf.dim() >= 1
        and leaf.shape[0] == batch_size
    }


def concat_batch_outputs(outputs: list[Any]) -> dict[str, torch.Tensor]:
    """Concatenate per-sample outputs path by path (tensor leaves with leading dim 1)."""
    per_path: dict[str, list[Any]] = {}
    for out in outputs:
        for path, leaf in flatten(out):
            per_path.setdefault(path, []).append(leaf)
    result: dict[str, torch.Tensor] = {}
    for path, leaves in per_path.items():
        if len(leaves) == len(outputs) and all(
            isinstance(leaf, torch.Tensor) and leaf.dim() >= 1 and leaf.shape[0] == 1
            for leaf in leaves
        ):
            result[path] = torch.cat(leaves, dim=0)
    return result


# --------------------------------------------------------------------------- #
# Seeding and inference context
# --------------------------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    """Seed ``random``, ``numpy`` (if installed) and ``torch`` (CPU and all CUDA devices)."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def deterministic_algorithms(enabled: bool = True):
    """Temporarily toggle ``torch.use_deterministic_algorithms``; restores the old value."""
    previous = torch.are_deterministic_algorithms_enabled()
    if enabled:
        # Needed by some cuBLAS ops; only effective if set before CUDA initialises.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(enabled)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def run_inference(model: Callable, inputs: Any) -> Any:
    """``call_model`` under ``torch.inference_mode()``."""
    with torch.inference_mode():
        return call_model(model, inputs)
