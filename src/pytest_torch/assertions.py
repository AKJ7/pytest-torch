"""Tensor assertions with readable failure messages."""
from __future__ import annotations

import torch

# (rtol, atol) per dtype, mirroring torch.testing defaults.
_DEFAULT_TOLERANCES = {
    torch.float64: (1e-7, 1e-7),
    torch.float32: (1.3e-6, 1e-5),
    torch.float16: (1e-3, 1e-5),
    torch.bfloat16: (1.6e-2, 1e-5),
}
_FALLBACK_TOLERANCE = (1.3e-6, 1e-5)


def _unravel(flat_index: int, shape) -> tuple:
    idx = []
    for dim in reversed(tuple(shape)):
        flat_index, rem = divmod(flat_index, dim)
        idx.append(rem)
    return tuple(reversed(idx))


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
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    if check_dtype and actual.dtype != expected.dtype:
        raise AssertionError(f"Dtype mismatch: {actual.dtype} != {expected.dtype}")
    if check_device and actual.device != expected.device:
        raise AssertionError(f"Device mismatch: {actual.device} != {expected.device}")

    default_rtol, default_atol = _DEFAULT_TOLERANCES.get(
        actual.dtype, _FALLBACK_TOLERANCE
    )
    rtol = default_rtol if rtol is None else rtol
    atol = default_atol if atol is None else atol

    if actual.numel() == 0:
        return

    a = actual.detach().to("cpu", torch.float64)
    e = expected.detach().to("cpu", torch.float64)

    nan_a, nan_e = torch.isnan(a), torch.isnan(e)
    if not torch.equal(nan_a, nan_e):
        raise AssertionError(
            f"NaN positions differ: {int(nan_a.sum())} in actual, "
            f"{int(nan_e.sum())} in expected"
        )
    inf_a, inf_e = torch.isinf(a), torch.isinf(e)
    if not torch.equal(inf_a, inf_e) or not torch.equal(a[inf_a], e[inf_e]):
        raise AssertionError("Inf positions or signs differ")

    finite = torch.isfinite(a) & torch.isfinite(e)
    diff = (a - e).abs()
    bad = finite & (diff > atol + rtol * e.abs())
    if not bad.any():
        return

    tiny = torch.finfo(torch.float64).tiny
    rel = torch.where(bad, diff / e.abs().clamp_min(tiny), torch.zeros_like(diff))
    masked = torch.where(finite, diff, torch.full_like(diff, -1.0))
    idx = _unravel(int(masked.reshape(-1).argmax()), diff.shape)

    raise AssertionError(
        f"{int(bad.sum())}/{a.numel()} elements differ beyond tolerance "
        f"(rtol={rtol:g}, atol={atol:g})\n"
        f"  max abs error: {diff[idx].item():.3e} at index {idx} "
        f"(actual={a[idx].item():.6g}, expected={e[idx].item():.6g})\n"
        f"  max rel error: {rel.max().item():.3e}"
    )


def assert_finite(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Assert a tensor contains no NaN/Inf values."""
    finite = torch.isfinite(tensor)
    if finite.all():
        return
    n_nan = int(torch.isnan(tensor).sum())
    n_inf = int(torch.isinf(tensor).sum())
    first = _unravel(int((~finite).reshape(-1).nonzero()[0]), tensor.shape)
    raise AssertionError(
        f"{name} has {n_nan} NaN and {n_inf} Inf values (first at index {first})"
    )


def assert_batch_invariant(model, x: torch.Tensor, **tolerances) -> None:
    """Assert ``model(x)[i] ~ model(x[i:i+1])[0]`` for every sample ``i``.

    The model is expected to be in eval mode already.
    """
    with torch.inference_mode():
        batched = model(x)
        single = torch.cat([model(x[i : i + 1]) for i in range(x.shape[0])])
    assert_tensors_close(single, batched, **tolerances)
