"""Invariance checks for a sequence model. Run: pytest examples/test_sequence_model.py -v"""
import pytest
import torch
from torch import nn

from pytest_torch import (
    assert_mask_invariant,
    assert_padding_invariant,
    assert_permutation_equivariant,
)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50, 8)
        self.proj = nn.Linear(8, 8)

    def forward(self, input_ids, attention_mask):
        h = self.proj(self.emb(input_ids))
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        # Try replacing this with `h.mean(1)`: the padding and mask tests below fail.
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1)
        return {"hidden": h, "pooled": pooled}


@pytest.fixture
def model(torch_seed, torch_device):
    return Encoder().to(torch_device).eval()


@pytest.fixture
def batch(make_tensor):
    # make_tensor draws on the CPU and moves to the device: identical on cpu/cuda/mps.
    ids = make_tensor(3, 6, dtype=torch.long, low=1, high=50)
    mask = torch.ones_like(ids)
    mask[1, 4:] = 0
    mask[2, 3:] = 0
    return {"input_ids": ids, "attention_mask": mask}


def test_padding_does_not_change_real_tokens(model, batch):
    assert_padding_invariant(model, batch)


def test_masked_tokens_do_not_leak(model, batch):
    assert_mask_invariant(model, batch)


def test_per_token_outputs_are_permutation_equivariant(model, batch):
    assert_permutation_equivariant(model, batch)
