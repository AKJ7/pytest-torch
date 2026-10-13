"""Tensor/pytree assertions with readable failure messages."""
from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from typing import Any

import torch

from .utils import (
    batch_aligned,
    call_model,
    concat_batch_outputs,
    find_batch_size,
    flatten,
    slice_batch,
)

# (rtol, atol) per dtype, mirroring torch.testing defaults.
_DEFAULT_TOLERANCES = {
    torch.float64: (1e-7, 1e-7),
    torch.float32: (1.3e-6, 1e-5),
    torch.float16: (1e-3, 1e-5),
    torch.bfloat16: (1.6e-2, 1e-5),
}
_FALLBACK_TOLERANCE = (1.3e-6, 1e-5)


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
def _tensor_diff(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float | None,
    atol: float | None,
    check_dtype: bool,
    check_device: bool,
) -> str | None:
    """Return a description of the mismatch, or None if the tensors are close."""
    if actual.shape != expected.shape:
        return f"Shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}"
    if check_dtype and actual.dtype != expected.dtype:
        return f"Dtype mismatch: {actual.dtype} != {expected.dtype}"
    if check_device and actual.device != expected.device:
        return f"Device mismatch: {actual.device} != {expected.device}"

    default_rtol, default_atol = _DEFAULT_TOLERANCES.get(
        actual.dtype, _FALLBACK_TOLERANCE
    )
    rtol = default_rtol if rtol is None else rtol
    atol = default_atol if atol is None else atol

    if actual.numel() == 0:
        return None

    a = actual.detach().to("cpu", torch.float64)
    e = expected.detach().to("cpu", torch.float64)

    nan_a, nan_e = torch.isnan(a), torch.isnan(e)
    if not torch.equal(nan_a, nan_e):
        return (
            f"NaN positions differ: {int(nan_a.sum())} in actual, "
            f"{int(nan_e.sum())} in expected"
        )
    inf_a, inf_e = torch.isinf(a), torch.isinf(e)
    if not torch.equal(inf_a, inf_e) or not torch.equal(a[inf_a], e[inf_e]):
        return "Inf positions or signs differ"

    finite = torch.isfinite(a) & torch.isfinite(e)
    diff = (a - e).abs()
    bad = finite & (diff > atol + rtol * e.abs())
    if not bad.any():
        return None

    tiny = torch.finfo(torch.float64).tiny
    rel = torch.where(bad, diff / e.abs().clamp_min(tiny), torch.zeros_like(diff))
    masked = torch.where(finite, diff, torch.full_like(diff, -1.0))
    idx = _unravel(int(masked.reshape(-1).argmax()), diff.shape)

    return (
        f"{int(bad.sum())}/{a.numel()} elements differ beyond tolerance "
        f"(rtol={rtol:g}, atol={atol:g})\n"
        f"max abs error: {diff[idx].item():.3e} at index {idx} "
        f"(actual={a[idx].item():.6g}, expected={e[idx].item():.6g})\n"
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
) -> None:
    """Assert two tensors are close, reporting *where* and *how badly* they differ."""
    message = _tensor_diff(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        check_dtype=check_dtype,
        check_device=check_device,
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
            return None
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
        )
        if message is not None:
            problems.append(f"{path}: {message}")
    return problems


def assert_outputs_close(
    actual: Any,
    expected: Any,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    check_dtype: bool = True,
    check_device: bool = True,
) -> None:
    """Pytree-aware ``assert_tensors_close`` for dicts, tuples, dataclasses, ModelOutputs.

    Reports the path of every mismatching entry (e.g. ``output['aux'][0]``), plus
    missing/unexpected entries. Non-tensor leaves are compared with ``==``
    (floats with a tolerance).
    """
    expected_leaves = dict(flatten(expected))
    problems = leaves_diff(
        dict(flatten(actual)),
        expected_leaves,
        rtol=rtol,
        atol=atol,
        check_dtype=check_dtype,
        check_device=check_device,
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
        singles = [call_model(model, slice_batch(inputs, i, size)) for i in range(size)]
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
