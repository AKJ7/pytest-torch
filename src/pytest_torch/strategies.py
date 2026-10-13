"""Hypothesis strategies for tensors. Needs ``pip install 'pytest-torch[hypothesis]'``.

    from hypothesis import given
    from pytest_torch.strategies import tensors

    @given(x=tensors(shape=(4, 16)))
    def test_model_is_finite(x): ...

Note: Hypothesis rejects function-scoped fixtures in ``@given`` tests by default; build
the model at module scope or suppress ``HealthCheck.function_scoped_fixture``.
"""
from __future__ import annotations

import hypothesis.strategies as st
import torch

_FLOAT_KINDS = ("uniform", "zeros", "constant", "extremes", "tiny")


def shapes(
    *, min_dims: int = 1, max_dims: int = 3, min_side: int = 1, max_side: int = 8
) -> st.SearchStrategy:
    """Tuples of positive ints (tensor shapes)."""
    return st.lists(
        st.integers(min_side, max_side), min_size=min_dims, max_size=max_dims
    ).map(tuple)


@st.composite
def tensors(
    draw,
    shape=None,
    *,
    dtype: torch.dtype = torch.float32,
    low: float = -10.0,
    high: float = 10.0,
    device: str = "cpu",
    edge_cases: bool = True,
    allow_nan: bool = False,
    allow_inf: bool = False,
    noncontiguous: bool = False,
):
    """Tensors of ``dtype`` with values in ``[low, high]``.

    ``shape`` may be a tuple, a strategy of shapes, or ``None`` (random 1-3 dims).
    With ``edge_cases=True`` the data is sometimes all zeros, constant, at the extreme
    values, or tiny; ``allow_nan`` / ``allow_inf`` additionally inject one non-finite
    value. With ``noncontiguous=True`` the memory layout is drawn too: contiguous,
    non-contiguous (like a transposed tensor) or, for 4-D shapes, channels_last.
    Data comes from a drawn seed, so examples are cheap to generate and replay.
    """
    tensor = draw(_tensor_values(shape, dtype, low, high, device, edge_cases, allow_nan, allow_inf))
    if noncontiguous:
        layouts = ["contiguous", "noncontiguous"] + (["channels_last"] if tensor.dim() == 4 else [])
        layout = draw(st.sampled_from(layouts))
        if layout != "contiguous":
            from .invariance import _relayout

            tensor = _relayout(tensor, layout)
    return tensor


@st.composite
def _tensor_values(draw, shape, dtype, low, high, device, edge_cases, allow_nan, allow_inf):
    if shape is None:
        shp = draw(shapes())
    elif isinstance(shape, st.SearchStrategy):
        shp = tuple(draw(shape))
    else:
        shp = tuple(shape)
    seed = draw(st.integers(0, 2**31 - 1))
    generator = torch.Generator().manual_seed(seed)

    if dtype == torch.bool:
        return torch.randint(0, 2, shp, generator=generator).to(device=device, dtype=dtype)
    if not dtype.is_floating_point:
        data = torch.randint(int(low), int(high) + 1, shp, generator=generator)
        return data.to(device=device, dtype=dtype)

    kinds = list(_FLOAT_KINDS) if edge_cases else ["uniform"]
    if allow_nan or allow_inf:
        kinds.append("nonfinite")
    kind = draw(st.sampled_from(kinds))
    span = float(high) - float(low)

    def uniform() -> torch.Tensor:
        return float(low) + span * torch.rand(shp, generator=generator, dtype=torch.float64)

    if kind == "zeros":
        data = torch.zeros(shp, dtype=torch.float64)
    elif kind == "constant":
        value = draw(st.floats(low, high, allow_nan=False, allow_infinity=False))
        data = torch.full(shp, value, dtype=torch.float64)
    elif kind == "extremes":
        signs = (torch.rand(shp, generator=generator, dtype=torch.float64) > 0.5).double() * 2 - 1
        data = signs * max(abs(float(low)), abs(float(high)))
    elif kind == "tiny":
        data = 1e-6 * (uniform() - float(low)) / (span or 1.0) * 2 - 1e-6
    elif kind == "nonfinite":
        data = uniform()
        flat = data.reshape(-1)
        if flat.numel():
            bad = [v for v, ok in ((float("nan"), allow_nan), (float("inf"), allow_inf),
                                   (-float("inf"), allow_inf)) if ok]
            index = int(torch.randint(0, flat.numel(), (1,), generator=generator))
            flat[index] = draw(st.sampled_from(bad))
    else:
        data = uniform()
    return data.to(device=device, dtype=dtype)


@st.composite
def padded_batches(
    draw,
    *,
    vocab_size: int,
    batch_size: int | st.SearchStrategy | None = None,
    max_len: int = 16,
    min_len: int = 1,
    pad_id: int = 0,
    side: str = "right",
    ids_key: str = "input_ids",
    mask_key: str = "attention_mask",
):
    """``{"input_ids", "attention_mask"}`` batches with ragged, padded sequences.

    ``batch_size`` is an int or a strategy (default: 1 to 4). Lengths cover the edge cases on purpose: every sequence of length ``min_len``, one full
    sequence among short ones, all full (no padding), and arbitrary mixes. Real tokens are
    drawn from ``[0, vocab_size)`` excluding ``pad_id``; ``side`` is ``"right"`` or ``"left"``.
    """
    if side not in ("right", "left"):
        raise ValueError("side must be 'right' or 'left'")
    if not 1 <= min_len <= max_len:
        raise ValueError("need 1 <= min_len <= max_len")
    if batch_size is None:
        batch_size = st.integers(1, 4)
    batch = draw(batch_size) if isinstance(batch_size, st.SearchStrategy) else int(batch_size)
    pattern = draw(st.sampled_from(["shortest", "one_full", "all_full", "mixed"]))
    if pattern == "shortest":
        lengths = [min_len] * batch
    elif pattern == "one_full":
        lengths = [max_len] + [min_len] * (batch - 1)
    elif pattern == "all_full":
        lengths = [max_len] * batch
    else:
        lengths = draw(st.lists(st.integers(min_len, max_len), min_size=batch, max_size=batch))
    seed = draw(st.integers(0, 2**31 - 1))
    generator = torch.Generator().manual_seed(seed)

    ids = torch.full((batch, max_len), pad_id, dtype=torch.long)
    mask = torch.zeros(batch, max_len, dtype=torch.long)
    tokens = [t for t in range(vocab_size) if t != pad_id]
    if not tokens:
        raise ValueError("vocab_size must leave at least one token besides pad_id")
    table = torch.tensor(tokens)
    for row, length in enumerate(lengths):
        values = table[torch.randint(0, len(tokens), (length,), generator=generator)]
        span = slice(0, length) if side == "right" else slice(max_len - length, max_len)
        ids[row, span] = values
        mask[row, span] = 1
    return {ids_key: ids, mask_key: mask}
