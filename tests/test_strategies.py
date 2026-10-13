import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hypothesis")
from hypothesis import given, settings
from torch import nn

from pytest_torch.strategies import shapes, tensors

MODEL = nn.Linear(8, 4).eval()  # module scope: Hypothesis dislikes function-scoped fixtures


@settings(max_examples=25, deadline=None)
@given(x=tensors(shape=(4, 8)))
def test_requested_shape_dtype_and_range(x):
    assert x.shape == (4, 8)
    assert x.dtype == torch.float32
    assert torch.isfinite(x).all()
    assert x.abs().max() <= 10.0 + 1e-5


@settings(max_examples=25, deadline=None)
@given(shape=shapes(min_dims=2, max_dims=3, max_side=5))
def test_shape_bounds(shape):
    assert 2 <= len(shape) <= 3
    assert all(1 <= side <= 5 for side in shape)


@settings(max_examples=25, deadline=None)
@given(x=tensors(shape=(3, 3), dtype=torch.long, low=0, high=5))
def test_integer_tensors_stay_in_range(x):
    assert x.dtype == torch.long
    assert int(x.min()) >= 0 and int(x.max()) <= 5


@settings(max_examples=25, deadline=None)
@given(x=tensors(shape=shapes(min_dims=2, max_dims=2, min_side=1, max_side=4), allow_nan=True, allow_inf=True))
def test_non_finite_values_are_allowed_when_requested(x):
    assert x.dim() == 2 and x.dtype == torch.float32


@settings(max_examples=25, deadline=None)
@given(x=tensors(shape=(6, 8)))
def test_linear_layer_is_finite_on_edge_cases(x):
    with torch.no_grad():
        assert torch.isfinite(MODEL(x)).all()
