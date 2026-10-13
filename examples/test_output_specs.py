"""Output specifications: shapes, dtypes, finite values, ranges and probabilities in one place.

Run: pytest examples/test_output_specs.py -v
"""
import pytest
import torch
from torch import nn

from pytest_torch import Probabilities, TensorSpec, assert_output_spec


class Tagger(nn.Module):
    """Token tagger: per-token logits and probabilities, plus a pooled confidence score."""

    def __init__(self, vocab=50, dim=16, tags=5):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, tags)

    def forward(self, input_ids):
        logits = self.head(self.emb(input_ids))
        probs = logits.softmax(-1)
        # Try `.sum(-1)` instead of `.mean(-1)` for confidence: it exceeds 1 and the range check fails.
        return {"logits": logits, "probs": probs, "confidence": probs.max(-1).values.mean(-1)}


# One spec, reused by every test below. "B" and "T" are symbolic: they must be the same
# size everywhere they appear, whatever that size is.
SPEC = {
    "logits": TensorSpec(("B", "T", 5), dtype=torch.float32),
    # Try `.softmax(1)` in forward(): the probabilities no longer sum to 1 along the tag dim.
    "probs": Probabilities(("B", "T", 5)),
    "confidence": TensorSpec(("B",), min=0.0, max=1.0),
}


@pytest.fixture
def model(torch_seed, torch_device):
    return Tagger().to(torch_device).eval()


def test_output_matches_spec(model, make_tensor):
    ids = make_tensor(4, 12, dtype=torch.long, low=0, high=50)
    with torch.inference_mode():
        dims = assert_output_spec(model(ids), SPEC, B=4)
    assert dims == {"B": 4, "T": 12}  # the bound sizes come back, e.g. for later checks


@pytest.mark.parametrize("batch, length", [(1, 1), (3, 7), (8, 64)])
def test_spec_holds_for_any_batch_and_length(model, make_tensor, batch, length):
    ids = make_tensor(batch, length, dtype=torch.long, low=0, high=50)
    with torch.inference_mode():
        assert_output_spec(model(ids), SPEC, B=batch, T=length)


def test_log_probabilities_and_any_leading_dims(make_tensor):
    scores = make_tensor(2, 3, 7)
    # `...` matches any number of leading dims; log=True checks logsumexp == 0.
    assert_output_spec(scores.log_softmax(-1), Probabilities((..., 7), log=True))
