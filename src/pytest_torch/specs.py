"""Declarative output specifications: shapes, dtypes, finiteness, ranges, probabilities.

    spec = {
        "logits": TensorSpec(("B", 10), dtype=torch.float32),
        "probs": Probabilities(("B", 10)),
        "hidden": TensorSpec(("B", "T", 64)),
    }
    assert_output_spec(model(x), spec, B=8)

Shape entries: an ``int`` (exact size), a ``str`` (a symbolic dim: every use must have the
same size), ``None`` (any size) or one ``...`` (any number of dims). Symbols are bound by
keyword arguments or by their first occurrence. The spec mirrors the output's structure
(dicts, lists/tuples, dataclasses, namedtuples, Hugging Face ``ModelOutput``s).
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .assertions import _unravel, format_problems


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    """What one output tensor must look like. Unset fields are not checked.

    ``finite`` (default ``True``) rejects NaN and Inf. ``min``/``max`` bound every value
    (inclusive). ``dtype`` may be a dtype or a tuple of allowed dtypes.
    """

    shape: Sequence[int | str | None | type(...)] | None = None
    dtype: torch.dtype | tuple[torch.dtype, ...] | None = None
    finite: bool = True
    min: float | None = None
    max: float | None = None

    def _value_problems(self, t: torch.Tensor) -> list[str]:
        problems = []
        if not (t.is_floating_point() or t.is_complex()):
            values = t
        else:
            values = t.detach()
            if self.finite and not bool(torch.isfinite(values).all()):
                nan = int(torch.isnan(values).sum())
                inf = int(torch.isinf(values).sum())
                problems.append(f"{nan} NaN and {inf} Inf values")
                values = values[torch.isfinite(values)]
        if values.numel() and self.min is not None and float(values.min()) < self.min:
            problems.append(f"values below min {self.min:g} (smallest {float(values.min()):.6g})")
        if values.numel() and self.max is not None and float(values.max()) > self.max:
            problems.append(f"values above max {self.max:g} (largest {float(values.max()):.6g})")
        return problems


@dataclasses.dataclass(frozen=True)
class Probabilities(TensorSpec):
    """Probabilities along ``dim``: values in ``[0, 1]`` summing to 1 (within ``atol``).

    With ``log=True``: log-probabilities, values ``<= 0`` whose ``logsumexp`` is 0.
    """

    dim: int = -1
    atol: float = 1e-4
    log: bool = False

    def _value_problems(self, t: torch.Tensor) -> list[str]:
        problems = super()._value_problems(t)
        if not t.is_floating_point():
            return problems + [f"probabilities must be floating point, got {t.dtype}"]
        x = t.detach().double()
        if x.numel() == 0:
            return problems
        if self.log:
            if float(x.max()) > self.atol:
                problems.append(f"log-probabilities above 0 (largest {float(x.max()):.6g})")
            total = x.logsumexp(self.dim)
            what = "logsumexp"
            target = 0.0
        else:
            if float(x.min()) < -self.atol or float(x.max()) > 1 + self.atol:
                problems.append(
                    f"probabilities outside [0, 1] (range {float(x.min()):.6g} .. {float(x.max()):.6g})"
                )
            total = x.sum(self.dim)
            what = "sum"
            target = 1.0
        error = (total - target).abs()
        finite = torch.isfinite(error)
        if bool(finite.any()) and float(error[finite].max()) > self.atol:
            worst = int(error.masked_fill(~finite, -1).argmax())
            index = _unravel(worst, error.shape)
            problems.append(
                f"{what} along dim {self.dim} is not {target:g}: "
                f"{int((error > self.atol).sum())}/{error.numel()} rows off, worst {float(total.reshape(-1)[worst]):.6g} "
                f"at {index} (atol={self.atol:g})"
            )
        return problems


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #
class _Dims:
    def __init__(self, bound: Mapping[str, int]):
        self.values = {k: int(v) for k, v in bound.items()}
        self.sources = {k: f"argument {k}={v}" for k, v in self.values.items()}

    def check(self, symbol: str, size: int, where: str) -> str | None:
        if symbol not in self.values:
            self.values[symbol] = size
            self.sources[symbol] = where
            return None
        if self.values[symbol] != size:
            return f"{symbol}={size}, but {symbol}={self.values[symbol]} from {self.sources[symbol]}"
        return None


def _shape_problems(shape: tuple[int, ...], spec_shape: Sequence, dims: _Dims, path: str) -> list[str]:
    spec_shape = list(spec_shape)
    if spec_shape.count(...) > 1:
        raise ValueError(f"{path}: a shape spec may contain at most one '...'")
    if ... in spec_shape:
        split = spec_shape.index(...)
        head, tail = spec_shape[:split], spec_shape[split + 1:]
        if len(shape) < len(head) + len(tail):
            return [f"shape {shape} has fewer than {len(head) + len(tail)} dims (spec {_fmt(spec_shape)})"]
        pairs = list(zip(head, shape[: len(head)])) + list(zip(tail, shape[len(shape) - len(tail):]))
        offsets = list(range(len(head))) + list(range(len(shape) - len(tail), len(shape)))
    else:
        if len(shape) != len(spec_shape):
            return [f"shape {shape} has {len(shape)} dims, spec {_fmt(spec_shape)} has {len(spec_shape)}"]
        pairs = list(zip(spec_shape, shape))
        offsets = list(range(len(shape)))
    problems = []
    for (expected, actual), dim in zip(pairs, offsets):
        if expected is None:
            continue
        if isinstance(expected, str):
            conflict = dims.check(expected, actual, f"{path} dim {dim}")
            if conflict:
                problems.append(f"dim {dim}: {conflict}")
        elif int(expected) != actual:
            problems.append(f"dim {dim} is {actual}, expected {expected}")
    if problems:
        problems[0] = f"shape {shape} vs spec {_fmt(spec_shape)}: " + problems[0]
    return problems


def _fmt(spec_shape: Sequence) -> str:
    return "(" + ", ".join("..." if s is ... else "*" if s is None else repr(s) if isinstance(s, str) else str(s)
                           for s in spec_shape) + ")"


def _check(output: Any, spec: Any, path: str, dims: _Dims, problems: list[str]) -> None:
    if spec is None:  # explicitly unchecked (it may also be absent)
        return
    if isinstance(spec, TensorSpec):
        if not isinstance(output, torch.Tensor):
            problems.append(f"{path}: expected a tensor, got {type(output).__name__}")
            return
        local = []
        if spec.shape is not None:
            local += _shape_problems(tuple(output.shape), spec.shape, dims, path)
        if spec.dtype is not None:
            allowed = spec.dtype if isinstance(spec.dtype, tuple) else (spec.dtype,)
            if output.dtype not in allowed:
                local.append(f"dtype {output.dtype}, expected {' or '.join(map(str, allowed))}")
        local += spec._value_problems(output)
        problems.extend(f"{path}: {p}" for p in local)
        return
    if isinstance(spec, Mapping):
        for key, child in spec.items():
            if isinstance(output, Mapping):
                if key in output:
                    _check(output[key], child, f"{path}[{key!r}]", dims, problems)
                elif child is not None:
                    problems.append(f"{path}: missing key {key!r} (has {list(output)})")
            elif hasattr(output, str(key)):
                _check(getattr(output, str(key)), child, f"{path}.{key}", dims, problems)
            elif child is not None:
                problems.append(f"{path}: {type(output).__name__} has no attribute {key!r}")
        return
    if isinstance(spec, (list, tuple)):
        if not isinstance(output, (list, tuple)):
            problems.append(f"{path}: expected a sequence, got {type(output).__name__}")
        elif len(output) != len(spec):
            problems.append(f"{path}: expected {len(spec)} items, got {len(output)}")
        else:
            for i, (child_out, child_spec) in enumerate(zip(output, spec)):
                _check(child_out, child_spec, f"{path}[{i}]", dims, problems)
        return
    raise TypeError(f"{path}: spec entries must be TensorSpec, dict, list/tuple or None, got {spec!r}")


def assert_output_spec(output: Any, spec: Any, **dims: int) -> dict[str, int]:
    """Assert ``output`` matches ``spec`` and return the bound symbolic dims.

    Keys present in the output but absent from the spec are ignored; ``None`` in the
    spec skips an entry. Every problem is reported, each with its output path.
    """
    bound = _Dims(dims)
    problems: list[str] = []
    _check(output, spec, "output", bound, problems)
    if problems:
        raise AssertionError(format_problems(f"Output does not match its spec: {len(problems)} problem(s)", problems))
    return dict(bound.values)
