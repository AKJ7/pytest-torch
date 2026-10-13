"""Unusual inputs: memory layouts, ragged padding, length-1 and full-length sequences.

Run: pytest examples/test_unusual_inputs.py -v   (the Hypothesis tests need pytest-torch[hypothesis])
"""
import pytest
import torch
from torch import nn

from pytest_torch import assert_layout_invariant, assert_mask_invariant


class PatchEncoder(nn.Module):
    """Image -> flattened patch features."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=4, stride=4)
        self.proj = nn.Linear(8 * 4 * 4, 16)

    def forward(self, images):
        feats = self.conv(images)
        # Try `feats.view(feats.shape[0], -1)`: fine for contiguous inputs, but a
        # channels_last batch makes `view` raise; the layout test catches it.
        return self.proj(feats.reshape(feats.shape[0], -1))


def test_any_memory_layout_gives_the_same_output(torch_seed, torch_device, make_tensor):
    model = PatchEncoder().to(torch_device).eval()
    # Same values, three layouts: contiguous, non-contiguous (like a transposed tensor)
    # and channels_last (what many vision pipelines produce).
    assert_layout_invariant(model, make_tensor(2, 3, 16, 16))


class MeanPoolEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(30, 8)

    def forward(self, input_ids, attention_mask):
        h = self.emb(input_ids)
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        # Try `h.mean(1)`: padding leaks into the pooled vector.
        return {"pooled": (h * m).sum(1) / m.sum(1).clamp_min(1)}


hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402

from pytest_torch.strategies import padded_batches, tensors  # noqa: E402

torch.manual_seed(0)
ENCODER = MeanPoolEncoder().eval()  # module scope: Hypothesis rejects function-scoped fixtures


@settings(max_examples=40, deadline=None)
@given(batch=padded_batches(vocab_size=30, max_len=12, side="right"))
def test_padding_never_leaks(batch):
    # Batches cover every sequence at length 1, one full sequence among short ones,
    # no padding at all, and random mixes.
    if bool(batch["attention_mask"].all()):
        return  # nothing is masked: nothing can leak
    assert_mask_invariant(ENCODER, batch)


@settings(max_examples=40, deadline=None)
@given(x=tensors(shape=(2, 3, 16, 16), noncontiguous=True, low=-1, high=1))
def test_finite_for_any_layout_and_edge_values(x):
    model = PatchEncoder().eval()
    with torch.inference_mode():
        assert torch.isfinite(model(x)).all()
