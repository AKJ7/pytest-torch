"""Shared fixtures: a tiny GPT trained to memorize a small corpus."""
import pytest
import torch
from torch.nn import functional as F

from .model import CORPUS, TinyGPT, encode


@pytest.fixture(scope="session")
def corpus_ids():
    return encode(CORPUS)


@pytest.fixture(scope="session")
def trained_lm(corpus_ids):
    """Trained until it reproduces the corpus (a few seconds on CPU)."""
    torch.manual_seed(0)
    model = TinyGPT()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    window = 32
    starts = torch.arange(0, len(corpus_ids) - window - 1)
    for _ in range(300):
        idx = starts[torch.randint(0, len(starts), (16,))]
        x = torch.stack([corpus_ids[i : i + window] for i in idx])
        y = torch.stack([corpus_ids[i + 1 : i + window + 1] for i in idx])
        loss = F.cross_entropy(model(x).logits.flatten(0, 1), y.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()
