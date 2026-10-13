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
):
    """Tensors of ``dtype`` with values in ``[low, high]``.

    ``shape`` may be a tuple, a strategy of shapes, or ``None`` (random 1-3 dims).
    With ``edge_cases=True`` the data is sometimes all zeros, constant, at the extreme
    values, or tiny; ``allow_nan`` / ``allow_inf`` additionally inject one non-finite
    value. Data comes from a drawn seed, so examples are cheap to generate and replay.
    """
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
