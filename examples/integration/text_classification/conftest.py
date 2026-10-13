"""Shared fixtures: what a real project's tests/conftest.py would provide."""
import pytest
import torch
from torch.nn import functional as F

from .model import NEGATIVE, NEUTRAL, POSITIVE, TextClassifier, Tokenizer, make_dataset


@pytest.fixture(scope="session")
def tokenizer():
    return Tokenizer(NEGATIVE + NEUTRAL + POSITIVE)


@pytest.fixture(scope="session")
def train_data():
    return make_dataset(600, seed=0)


@pytest.fixture(scope="session")
def val_loader(tokenizer):
    data = make_dataset(200, seed=1)
    batches = []
    for i in range(0, len(data), 32):  # last batch is smaller: metrics must weight by size
        texts, labels = zip(*data[i : i + 32])
        batches.append((tokenizer(list(texts)), torch.tensor(labels)))
    return batches


@pytest.fixture(scope="session")
def trained_model(tokenizer, train_data):
    """A briefly trained model (a few seconds on CPU), shared by the accuracy tests."""
    torch.manual_seed(0)
    model = TextClassifier(len(tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    texts, labels = zip(*train_data)
    inputs, targets = tokenizer(list(texts)), torch.tensor(labels)
    for _ in range(3):
        for i in range(0, len(texts), 32):
            batch = {k: v[i : i + 32] for k, v in inputs.items()}
            loss = F.cross_entropy(model(**batch).logits, targets[i : i + 32])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()
