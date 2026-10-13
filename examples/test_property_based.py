"""Property-based tests: let Hypothesis find the inputs you didn't think of.

Run: pytest examples/test_property_based.py -v   (needs: pip install "pytest-torch[hypothesis]")
"""
import pytest

pytest.importorskip("hypothesis")

import torch
from hypothesis import given, settings, strategies as st
from torch import nn

from pytest_torch import assert_finite
from pytest_torch.strategies import shapes, tensors


class StableSoftmaxHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 5)

    def forward(self, x):
        # Try `self.proj(x).exp() / self.proj(x).exp().sum(-1, keepdim=True)`: Hypothesis
        # finds an input at the extremes that overflows to inf / nan.
        return self.proj(x).softmax(-1)


class Normalizer(nn.Module):
    """Standardize the last dimension, guarding against zero variance."""

    def forward(self, x):
        # Try removing `+ 1e-6`: an all-zeros or constant input gives 0 / 0 = nan.
        return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True, unbiased=False) + 1e-6)


# Hypothesis rejects function-scoped fixtures in @given tests: build models at module scope.
torch.manual_seed(0)
HEAD = StableSoftmaxHead().eval()


@settings(max_examples=50, deadline=None)
@given(x=tensors(shape=(4, 16), low=-1e4, high=1e4))  # zeros, constants, extremes, tiny values
def test_probabilities_are_valid_on_edge_cases(x):
    with torch.inference_mode():
        probs = HEAD(x)
    assert_finite(probs, "probs")
    assert torch.allclose(probs.sum(-1), torch.ones(4), atol=1e-5)
    assert (probs >= 0).all()


@settings(max_examples=50, deadline=None)
@given(x=tensors(shape=shapes(min_dims=1, max_dims=3, max_side=6), low=-100, high=100))
def test_normalizer_handles_any_shape(x):
    out = Normalizer()(x)
    assert out.shape == x.shape
    assert_finite(out, "normalized")


@settings(max_examples=30, deadline=None)
@given(ids=tensors(shape=st.tuples(st.integers(1, 4), st.integers(1, 12)), dtype=torch.long, low=0, high=99))
def test_embedding_accepts_every_valid_token(ids):
    emb = nn.Embedding(100, 8)
    assert emb(ids).shape == (*ids.shape, 8)


@settings(max_examples=30, deadline=None)
@given(x=tensors(shape=(2, 16), allow_nan=True))
def test_nan_inputs_are_rejected_loudly(x):
    # A guard that refuses bad inputs instead of propagating NaN downstream.
    def guarded(t):
        if not torch.isfinite(t).all():
            raise ValueError("non-finite input")
        return HEAD(t)

    if torch.isfinite(x).all():
        guarded(x)
    else:
        with pytest.raises(ValueError, match="non-finite"):
            guarded(x)
