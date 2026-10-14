"""Tensor/pytree assertions with readable failure messages."""
from __future__ import annotations

import dataclasses
import math
import numbers
from collections.abc import Mapping
from typing import Any

import torch

from . import tolerances
from .utils import (
    _is_dataclass_instance,
    _is_namedtuple,
    batch_aligned,
    call_model,
    concat_batch_outputs,
    find_batch_size,
    flatten,
    slice_batch,
)

# Python floats in nested outputs are compared like float32 tensors.
_FALLBACK_TOLERANCE = tolerances.FALLBACK


class BatchInvarianceNotApplicable(ValueError):
    """Raised when inputs/outputs don't allow a batch-invariance comparison."""


def _unravel(flat_index: int, shape) -> tuple:
    idx = []
    for dim in reversed(tuple(shape)):
        flat_index, rem = divmod(flat_index, dim)
        idx.append(rem)
    return tuple(reversed(idx))


def format_problems(header: str, problems: list[str], limit: int = 10) -> str:
    shown = problems[:limit]
    text = header + "\n" + "\n".join("  " + p.replace("\n", "\n    ") for p in shown)
    if len(problems) > limit:
        text += f"\n  ... and {len(problems) - limit} more"
    return text


# --------------------------------------------------------------------------- #
# Single tensors
# --------------------------------------------------------------------------- #
def _fmt(value: Any) -> str:
    if isinstance(value, complex):
        return f"({value.real:.6g}{value.imag:+.6g}j)"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _exact_diff(a: torch.Tensor, e: torch.Tensor, rtol: float | None, atol: float | None) -> str | None:
    """Integers and bools: exact (no float64 round trip, so values above 2**53 stay distinct),
    unless the caller passed a tolerance explicitly."""
    ai, ei = a.to(torch.int64), e.to(torch.int64)
    diff = (ai - ei).abs()
    if rtol is None and atol is None:
        bad = ai != ei
        how = "exact comparison"
    else:
        allowed = (atol or 0.0) + (rtol or 0.0) * ei.abs().to(torch.float64)
        bad = diff.to(torch.float64) > allowed
        how = f"rtol={rtol or 0:g}, atol={atol or 0:g}"
    if not bool(bad.any()):
        return None
    idx = _unravel(int(torch.where(bad, diff, torch.zeros_like(diff)).reshape(-1).argmax()), diff.shape)
    return (
        f"{int(bad.sum())}/{a.numel()} elements differ ({how})\n"
        f"max abs error: {int(diff[idx])} at index {idx} "
        f"(actual={int(ai[idx])}, expected={int(ei[idx])})"
    )


def _tensor_diff(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float | None,
    atol: float | None,
    check_dtype: bool,
    check_device: bool,
    equal_nan: bool = True,
) -> str | None:
    """Return a description of the mismatch, or None if the tensors are close."""
    if actual.shape != expected.shape:
        return f"Shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}"
    if check_dtype and actual.dtype != expected.dtype:
        return f"Dtype mismatch: {actual.dtype} != {expected.dtype}"
    if check_device and actual.device != expected.device:
        return f"Device mismatch: {actual.device} != {expected.device}"
    if actual.numel() == 0:
        return None

    a = actual.detach().cpu().resolve_conj()
    e = expected.detach().cpu().resolve_conj()
    inexact = any(t.is_floating_point() or t.is_complex() for t in (a, e))
    if not inexact:
        return _exact_diff(a, e, rtol, atol)

    # Defaults per dtype: built-in, overridable in pyproject.toml or with a marker.
    rtol, atol = tolerances.for_dtype(actual.dtype, actual.device, rtol, atol)
    # complex128 keeps the imaginary part; |a - e| is then the complex magnitude.
    work = torch.complex128 if (a.is_complex() or e.is_complex()) else torch.float64
    a, e = a.to(work), e.to(work)

    nan_a, nan_e = torch.isnan(a), torch.isnan(e)
    if not torch.equal(nan_a, nan_e):
        return (
            f"NaN positions differ: {int(nan_a.sum())} in actual, "
            f"{int(nan_e.sum())} in expected"
        )
    if not equal_nan and bool(nan_a.any()):
        return f"{int(nan_a.sum())} NaN in actual and {int(nan_e.sum())} in expected (equal_nan=False)"
    inf_a, inf_e = torch.isinf(a), torch.isinf(e)
    if not torch.equal(inf_a, inf_e) or not torch.equal(a[inf_a], e[inf_e]):
        return "Inf positions or signs differ"

    finite = torch.isfinite(a) & torch.isfinite(e)
    diff = (a - e).abs()
    e_abs = e.abs()
    bad = finite & (diff > atol + rtol * e_abs)
    if not bad.any():
        return None

    tiny = torch.finfo(torch.float64).tiny
    rel = torch.where(bad, diff / e_abs.clamp_min(tiny), torch.zeros_like(diff))
    masked = torch.where(finite, diff, torch.full_like(diff, -1.0))
    idx = _unravel(int(masked.reshape(-1).argmax()), diff.shape)

    return (
        f"{int(bad.sum())}/{a.numel()} elements differ beyond tolerance "
        f"(rtol={rtol:g}, atol={atol:g})\n"
        f"max abs error: {diff[idx].item():.3e} at index {idx} "
        f"(actual={_fmt(a[idx].item())}, expected={_fmt(e[idx].item())})\n"
        f"max rel error: {rel.max().item():.3e}"
    )


def assert_tensors_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    check_dtype: bool = True,
    check_device: bool = True,
    equal_nan: bool = True,
) -> None:
    """Assert two tensors are close, reporting *where* and *how badly* they differ.

    Floating-point tensors use dtype-aware default tolerances; complex tensors compare the
    complex difference (real and imaginary parts); integer and bool tensors compare exactly
    unless ``rtol``/``atol`` is given. NaNs must sit at the same positions, and count as
    equal unless ``equal_nan=False``.
    """
    message = _tensor_diff(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        check_dtype=check_dtype,
        check_device=check_device,
        equal_nan=equal_nan,
    )
    if message is not None:
        raise AssertionError(message)


# --------------------------------------------------------------------------- #
# Pytrees
# --------------------------------------------------------------------------- #
def _leaf_diff(
    actual: Any,
    expected: Any,
    *,
    rtol: float | None,
    atol: float | None,
    check_dtype: bool,
    check_device: bool,
    equal_nan: bool = True,
) -> str | None:
    a_is_tensor = isinstance(actual, torch.Tensor)
    e_is_tensor = isinstance(expected, torch.Tensor)
    if a_is_tensor and e_is_tensor:
        return _tensor_diff(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            check_dtype=check_dtype,
            check_device=check_device,
            equal_nan=equal_nan,
        )
    if a_is_tensor != e_is_tensor:
        return f"Type mismatch: {type(actual).__name__} != {type(expected).__name__}"

    both_real = (
        isinstance(actual, numbers.Real)
        and isinstance(expected, numbers.Real)
        and not isinstance(actual, bool)
        and not isinstance(expected, bool)
    )
    if both_real and (isinstance(actual, float) or isinstance(expected, float)):
        if math.isnan(actual) and math.isnan(expected):
            return None if equal_nan else f"{actual!r} != {expected!r} (equal_nan=False)"
        r = _FALLBACK_TOLERANCE[0] if rtol is None else rtol
        a = _FALLBACK_TOLERANCE[1] if atol is None else atol
        if math.isclose(actual, expected, rel_tol=r, abs_tol=a):
            return None
        return f"{actual!r} != {expected!r} (rtol={r:g}, atol={a:g})"

    try:
        equal = bool(actual == expected)
    except Exception:
        equal = False
    return None if equal else f"{actual!r} != {expected!r}"


def leaves_diff(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    rtol: float | None = None,
    atol: float | None = None,
    check_dtype: bool = True,
    check_device: bool = True,
    equal_nan: bool = True,
) -> list[str]:
    """Compare two ``{path: leaf}`` mappings; return a list of problem descriptions."""
    problems: list[str] = []
    missing = [p for p in expected if p not in actual]
    extra = [p for p in actual if p not in expected]
    if missing:
        problems.append("missing in actual: " + ", ".join(missing))
    if extra:
        problems.append("unexpected in actual: " + ", ".join(extra))
    for path, exp in expected.items():
        if path not in actual:
            continue
        message = _leaf_diff(
            actual[path],
            exp,
            rtol=rtol,
            atol=atol,
            check_dtype=check_dtype,
            check_device=check_device,
            equal_nan=equal_nan,
        )
        if message is not None:
            problems.append(f"{path}: {message}")
    return problems


# Output structure. Leaves alone miss empty containers ([] vs {}) and container kinds.
def _kind(obj: Any) -> str:
    if isinstance(obj, torch.Tensor):
        return "tensor"
    if isinstance(obj, Mapping):
        return "mapping"
    if _is_namedtuple(obj) or _is_dataclass_instance(obj):
        return type(obj).__name__
    if isinstance(obj, (list, tuple)):
        return "sequence"  # tuple vs list is not a structure change
    return "value"


def _describe(obj: Any) -> str:
    kind = _kind(obj)
    return f"empty {kind}" if kind in ("mapping", "sequence") and len(obj) == 0 else kind


def _children(obj: Any, path: str) -> list[tuple[Any, Any, str]]:
    """``(key, child, child_path)`` with the same paths as ``flatten``."""
    if isinstance(obj, Mapping):
        return [(k, v, f"{path}[{k!r}]") for k, v in obj.items()]
    if _is_namedtuple(obj):
        return [(f, getattr(obj, f), f"{path}.{f}") for f in obj._fields]
    if _is_dataclass_instance(obj):
        return [(f.name, getattr(obj, f.name), f"{path}.{f.name}") for f in dataclasses.fields(obj)]
    return [(i, v, f"{path}[{i}]") for i, v in enumerate(obj)]


def structure_problems(actual: Any, expected: Any, path: str = "output") -> list[tuple[str, str]]:
    """``(path, problem)`` for structure changes that comparing leaves can't see: a
    container kind that changed (sequence vs mapping, tensor vs container) and entries
    present on one side only that hold no values (empty containers)."""
    kind_a, kind_e = _kind(actual), _kind(expected)
    if kind_a != kind_e:
        if {kind_a, kind_e} <= {"tensor", "value"}:
            return []  # two leaves: the leaf comparison reports it
        return [(path, f"{_describe(actual)} vs {_describe(expected)}")]
    if kind_a in ("tensor", "value"):
        return []
    a_children = {key: (child, child_path) for key, child, child_path in _children(actual, path)}
    e_children = {key: (child, child_path) for key, child, child_path in _children(expected, path)}
    problems = []
    for key, (child, child_path) in a_children.items():
        if key not in e_children and not flatten(child):
            problems.append((child_path, f"only in actual ({_describe(child)})"))
    for key, (child, child_path) in e_children.items():
        if key not in a_children and not flatten(child):
            problems.append((child_path, f"only in expected ({_describe(child)})"))
    for key, (child, child_path) in a_children.items():
        if key in e_children:
            problems += structure_problems(child, e_children[key][0], child_path)
    return problems


def _under(path: str, prefixes: list[str]) -> bool:
    return any(path == p or path.startswith(p + "[") or path.startswith(p + ".") for p in prefixes)


def assert_outputs_close(
    actual: Any,
    expected: Any,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    check_dtype: bool = True,
    check_device: bool = True,
    equal_nan: bool = True,
) -> None:
    """Pytree-aware ``assert_tensors_close`` for dicts, tuples, dataclasses, ModelOutputs.

    Reports the path of every mismatching entry (e.g. ``output['aux'][0]``), missing and
    unexpected entries, and structure changes, including empty containers (``[]`` vs
    ``{}``). A tuple and a list with the same contents count as the same. Non-tensor
    leaves are compared with ``==`` (floats with a tolerance).
    """
    structural = structure_problems(actual, expected)
    changed = [path for path, _ in structural]
    expected_leaves = {p: v for p, v in flatten(expected) if not _under(p, changed)}
    problems = [f"{path}: {problem}" for path, problem in structural] + leaves_diff(
        {p: v for p, v in flatten(actual) if not _under(p, changed)},
        expected_leaves,
        rtol=rtol,
        atol=atol,
        check_dtype=check_dtype,
        check_device=check_device,
        equal_nan=equal_nan,
    )
    if problems:
        raise AssertionError(
            format_problems(
                f"Outputs differ: {len(problems)} problem(s) across "
                f"{len(expected_leaves)} entries",
                problems,
            )
        )


def assert_finite(obj: Any, name: str = "tensor") -> None:
    """Assert that every floating-point tensor in ``obj`` has no NaN/Inf values."""
    problems = []
    for path, leaf in flatten(obj, name):
        if not isinstance(leaf, torch.Tensor) or not leaf.is_floating_point():
            continue
        finite = torch.isfinite(leaf)
        if finite.all():
            continue
        n_nan = int(torch.isnan(leaf).sum())
        n_inf = int(torch.isinf(leaf).sum())
        first = _unravel(int((~finite).reshape(-1).nonzero()[0]), leaf.shape)
        problems.append(
            f"{path} has {n_nan} NaN and {n_inf} Inf values (first at index {first})"
        )
    if problems:
        raise AssertionError("\n".join(problems))


# --------------------------------------------------------------------------- #
# Batch invariance
# --------------------------------------------------------------------------- #
def batch_invariance_problems(
    model,
    inputs: Any,
    *,
    batched: Any = None,
    rtol: float | None = None,
    atol: float | None = None,
    check_dtype: bool = True,
    check_device: bool = True,
) -> list[str]:
    """Compare ``model(batch)`` against per-sample runs; return problem descriptions.

    Raises ``BatchInvarianceNotApplicable`` if there is nothing to compare.
    """
    size = find_batch_size(inputs)
    if size is None or size < 2:
        raise BatchInvarianceNotApplicable("needs inputs with a batch dimension >= 2")
    with torch.inference_mode():
        if batched is None:
            batched = call_model(model, inputs)
        try:
            singles = [call_model(model, slice_batch(inputs, i, size)) for i in range(size)]
        except Exception as exc:  # e.g. BatchNorm in training mode refuses a batch of one
            training = [n or "<root>" for n, m in getattr(model, "named_modules", list)() if m.training]
            hint = (f" The model is in training mode ({', '.join(training[:3])}); call model.eval()."
                    if training else "")
            raise AssertionError(
                f"Batch invariance: the model fails on a single sample: {type(exc).__name__}: "
                f"{str(exc).splitlines()[0] if str(exc) else ''}.{hint}"
            ) from exc
    expected = batch_aligned(batched, size)
    if not expected:
        raise BatchInvarianceNotApplicable("no batch-aligned tensor outputs to compare")
    return leaves_diff(
        concat_batch_outputs(singles),
        expected,
        rtol=rtol,
        atol=atol,
        check_dtype=check_dtype,
        check_device=check_device,
    )


def assert_batch_invariant(model, inputs: Any, **tolerances) -> None:
    """Assert ``model(batch)[i] ~ model(batch[i:i+1])[0]`` for every sample ``i``.

    ``inputs`` may be a tensor, a tuple of positional args, or a dict of kwargs
    (tensors whose leading dim equals the batch size are sliced). Outputs may be
    arbitrary pytrees. The model is expected to be in eval mode already.
    """
    try:
        problems = batch_invariance_problems(model, inputs, **tolerances)
    except BatchInvarianceNotApplicable as exc:
        raise ValueError(f"assert_batch_invariant {exc}") from None
    if problems:
        raise AssertionError(
            format_problems(
                "Batch invariance violated: per-sample results differ from the batched run",
                problems,
            )
        )
